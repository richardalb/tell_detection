"""
This module provides the PyTorch ``Dataset`` implementations and the PyTorch
Lightning ``LightningDataModule`` that drive the pipeline described in the
associated paper:

* :class:`TellDataset` -- the self-labelled site / non-site classification
  dataset used for training, validation and testing.
* :class:`PredictDataset` -- a tiled GeoTIFF dataset used to run sliding
  window inference over large satellite scenes.
* :class:`VadineanuDataset` -- an external, independently labelled dataset
  used for cross-dataset generalisation experiments.
* :class:`TellDataModule` -- the Lightning data module that wires the above
  datasets together and exposes the train / validation / test / predict
  dataloaders to the training scripts.

All datasets expect the directory layout documented in the repository README
and distributed with the accompanying Zenodo data release.
"""

import os
import glob

import lightning as L
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from torchvision.io import read_image
from torchvision.transforms import v2


class TellDataset(torch.utils.data.Dataset):
    """Self-labelled site / non-site classification dataset.

    Labels are read from two CSV files inside ``<path>/classification``:
    ``classification_labels.csv`` (positives, ``choice == "Site"``) and
    ``new_nosites.csv`` (negatives, ``choice == "nosite"``). A fixed block of
    legacy negative tiles (ids ``18238``--``18637``) is additionally synthesised
    to reproduce the sampling used in the paper.

    Images are read as single-channel (grayscale) tensors normalised to
    ``[0, 1]`` and, unless the model is a Swin or ViT variant, zero-padded to
    ``400 x 400`` pixels.

    Args:
        path: Root of the labelled dataset (default: ``"./data"``).
        transform: Optional ``torchvision`` transform applied to the image
            after loading.
        model_name: Name of the backbone, used to decide whether images must be
            padded to ``400 x 400`` (transformer backbones expect ``384 x 384``
            and are not padded here).
    """

    def __init__(self, path="./data", transform=None, model_name="cnn"):
        self.data_path = path

        # Positive examples: tiles labelled as "Site".
        labels = pd.read_csv(
            os.path.join(path, "classification", "classification_labels.csv")
        )[["choice", "image", "id"]]
        labels = labels[labels["choice"] == "Site"]

        # Legacy negative tiles that were not part of the original CSV export.
        # They are re-created here so that the negative class composition matches
        # the experiments reported in the paper.
        nosites = []
        for i in range(500, 1000):
            nosites.append(
                {"id": i + 18238, "choice": "Nosite", "image": str(i) + ".png"}
            )

        # Additional negatives stored separately in a second CSV.
        new_nosites = pd.read_csv(
            os.path.join(path, "classification", "new_nosites.csv")
        )[["choice", "image", "id"]]
        new_nosites = new_nosites[new_nosites["choice"] == "nosite"]

        # Merge positives and both negative sources into a single label table.
        labels = pd.concat([labels, pd.DataFrame(nosites),
                           new_nosites], ignore_index=True)
        self.labels = labels
        self.transform = transform
        self.model_name = model_name

    def __len__(self):
        """Return the total number of labelled tiles."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Load, normalise and optionally augment the tile at ``idx``.

        Returns:
            A dictionary with keys ``image`` (tensor of shape ``[1, H, W]``),
            ``label`` (``1.0`` for a site, ``0.0`` otherwise) and ``id``
            (integer identifier).
        """
        id = self.labels.iloc[idx]["id"]

        label = self.labels.iloc[idx]["choice"]
        file_path = os.path.join(
            self.data_path,
            "classification",
        )
        if label == "Site":
            # Positive tiles live in the ``casini_site`` folder. The stored path
            # carries a fixed-length prefix that is stripped before opening.
            file_name = self.labels.iloc[idx]["image"].split("/")[-1]
            file_name = file_name[9:]
            label = 1.0
            image = read_image(os.path.join(
                file_path, "casini_site", file_name)) / 255
        else:
            # Negative tiles can be stored in three different ways depending on
            # their provenance; each is resolved to an on-disk image here.
            label = 0.0
            file_name = self.labels.iloc[idx]["image"].split("/")[-1]
            if len(self.labels.iloc[idx]["image"].split("/")) == 1:
                # Plain ``nosite/<name>`` reference.
                image = read_image(os.path.join(
                    file_path, "nosite", file_name)) / 255
            elif len(self.labels.iloc[idx]["image"].split("-")) == 3:
                # Composite key ``...-...-<tile>`` pointing into ``casini_site``.
                image = read_image(os.path.join(
                    file_path, "casini_site", self.labels.iloc[idx]["image"].split(
                        "-")[-1]
                ))
            else:
                # Fallback: the file already lives under ``casini_site``.
                image = read_image(os.path.join(
                    file_path, "casini_site", file_name)) / 255

        if self.transform:
            image = self.transform(image)

        # Convolutional backbones expect a fixed 400 x 400 input; smaller tiles
        # are zero-padded. Transformer backbones (Swin / ViT) resize internally
        # and are therefore skipped.
        if (
            image.shape != torch.Size([1, 400, 400])
            and "swin" not in self.model_name
            and "vit" not in self.model_name
        ):
            temp = torch.zeros(1, 400, 400)
            temp[:, : image.shape[1], : image.shape[2]] = image
            image = temp

        return {"image": image, "label": label, "id": int(id)}


class PredictDataset(torch.utils.data.Dataset):
    """Tiled GeoTIFF dataset for sliding-window inference.

    Each ``.tif`` file found under the per-country sub-directories of ``path``
    is treated as a large raster scene. The scene is decomposed into
    ``stride x stride`` (200 px) steps and windowed into overlapping
    ``400 x 400`` tiles. The total number of tiles per scene is computed from
    the scene area and used to map a flat dataset index back to a
    ``(scene, row, column)`` position.

    Geometric metadata (CRS affine transform, width and height) is cached so
    that predicted tile coordinates can later be georeferenced.

    Args:
        path: Root directory containing one sub-directory per country.
        countries: List of country names (sub-directory names) to process.
    """

    def __init__(self, path, countries):
        predict_paths = []
        self.countries = countries
        for country in self.countries:
            predict_paths.append(
                os.path.join(path, country)
            )
        predict_paths = predict_paths
        window_size = 400
        self.stride = 200

        # Index every GeoTIFF and cache the metadata needed to map a flat index
        # back to a pixel window. Metadata is kept in memory so that the HPC
        # workers do not have to reopen every raster at each access.
        tiffs = []
        for path in predict_paths:
            for tif in os.listdir(path):
                if tif[-4:] != ".tif":
                    print("unvalid file: ", tif)
                    continue
                temp = rasterio.open(os.path.join(path, tif))
                width, height = temp.width, temp.height
                transform = temp.transform
                # Approximate number of sliding-window positions for this raster.
                num_tiles = int((height * width) / (self.stride**2))
                # Load everything into RAM for hpc.
                tiffs.append(
                    [path, tif, num_tiles, height, width, transform])
                temp.close()
                print('ADDED TIF: ', tif, path)

        self.tiffs = tiffs
        # Deterministic conversion from NumPy arrays to single-channel float
        # tensors scaled to [0, 1].
        self.transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        print('FINISHED READING DATA')

    def __len__(self):
        """Return the total number of tiles across all indexed scenes."""
        len = 0
        for tiff in self.tiffs:
            len += tiff[2]
        return len

    def __getitem__(self, idx):
        """Read the tile encoded by the flat index ``idx``.

        Returns:
            A dictionary with the single-channel ``image`` tile, the
            ``country`` / ``tif`` identifiers, and the top-left pixel offset
            (``widths``, ``heights``) used for georeferencing the prediction.
        """
        # Walk the cached scenes until the one containing ``idx`` is found.
        temp_idx = idx
        for i, tiff in enumerate(self.tiffs):
            if tiff[2] > temp_idx:
                country = tiff[0].split("/")[-1]
                tif = tiff[1]
                total_width = tiff[4]
                total_height = tiff[3]
                # Convert the flat index into row (``h``) and column (``w``)
                # pixel offsets within the scene.
                h = int((temp_idx * self.stride) / total_width) * self.stride
                w = (
                    temp_idx - ((h / self.stride) *
                                (total_width / self.stride))
                ) * self.stride
                tile = rasterio.open(os.path.join(tiff[0], tiff[1]))
                try:
                    img = torch.Tensor(tile.read(1, window=Window(w, h, 400, 400)))
                except:
                    # Edge tiles that fall outside the raster are filled with
                    # zeros and flagged so they can be excluded downstream.
                    img = torch.zeros((400, 400))
                    country += '_failed'
                tile.close()
                break
            else:
                temp_idx -= tiff[2]

        # Zero-pad tiles truncated by the raster boundary.
        if img.shape[0] != 400 or img.shape[1] != 400:
            temp = torch.zeros(400, 400)
            temp[: img.shape[0], : img.shape[1]] = img
            img = temp

        if self.transform:
            img = self.transform(img)

        return {
            "image": img.unsqueeze(0),
            "country": country,
            "tif": tif,
            "widths": w,
            "heights": h,
        }


class VadineanuDataset(torch.utils.data.Dataset):
    """External validation dataset following the Vadineanu et al. split.

    Expects two folders, ``<path>site`` and ``<path>nosite``, containing the
    positive and negative tiles respectively. Positives are assigned label
    ``1`` and negatives label ``0``.

    Note:
        The attribute names referenced in :meth:`__len__` and
        :meth:`__getitem__` (``no_sites`` / ``site``) do not match the ones
        assigned in :meth:`__init__` (``sites`` / ``nosites``); the original
        code is preserved verbatim for reproducibility.

    Args:
        path: Prefix path; ``"site"`` and ``"nosite"`` are appended to it.
        transform: Optional ``torchvision`` transform applied to the image.
    """

    def __init__(self, path, transform=None):
        self.sites = glob.glob(path + 'site')
        self.nosites = glob.glob(path + 'nosite')
        self.transform = transform

    def __len__(self):
        """Return the combined number of site and non-site tiles."""
        return len(self.sites) + len(self.no_sites)

    def __getitem__(self, idx):
        """Load the site or non-site tile at ``idx``."""
        if idx >= len(self.sites):
            image = read_image(self.nosites[idx - len(self.site)]) / 255
            label = 0
            img_name = self.nosites[idx - len(self.sites)].split("/")[-1][:-4]
        else:
            image = read_image(self.site[idx]) / 255
            label = 1
            img_name = self.sites[idx].split("/")[-1][:-4]

        if self.transform:
            image = self.transform(image)

        return {'id': img_name,
                'label': label,
                'image': image}


class TellDataModule(L.LightningDataModule):
    """Lightning data module orchestrating the tell-detection datasets.

    The module builds a :class:`TellDataset` and either uses the train /
    validation / test indices supplied by the caller (e.g. the k-fold splits
    generated in ``train.py``) or creates a deterministic 40 / 20 / 40
    train / validation / test split on first use. It exposes the standard
    Lightning dataloader hooks plus :meth:`predict_dataloader`.

    Args:
        path: Root of the labelled dataset.
        batch_size: Batch size for all dataloaders.
        model_name: Backbone name passed to :class:`TellDataset`.
        countries: Country sub-directories used for prediction.
        transform: Transform applied by :class:`TellDataset`.
        train_indices: Indices assigned to the training split.
        val_indices: Indices assigned to the validation split.
        test_indices: Indices assigned to the test split.
        num_workers: Number of dataloader worker processes.
    """

    def __init__(
        self,
        path: str = "data",
        batch_size: int = 100,
        model_name="ConvNeXt",
        countries=["Syria"],
        transform=None,
        train_indices=[],
        val_indices=[],
        test_indices=[],
        num_workers=15,
    ):
        self.path = path
        self.transform = transform
        self.countries = countries
        self.model_name = model_name
        self.batch_size = batch_size
        # Allows a split whose length is zero when running on multiple devices.
        self.allow_zero_length_dataloader_with_multiple_devices = True
        self.num_workers = num_workers
        self.save_hyperparameters()

        # Only honour caller-provided splits if all three are non-empty.
        if any(train_indices) and any(val_indices) and any(test_indices):
            self.train_indices = train_indices
            self.val_indices = val_indices
            self.test_indices = test_indices
        else:
            self.train_indices = []
            self.val_indices = []
            self.test_indices = []

    def setup(self, stage: str, xai_method=None):
        """Prepare the datasets and samplers for the requested ``stage``.

        Stages ``"fit"``, ``"validate"``, ``"test"`` and ``"vadineanu"`` build a
        :class:`TellDataset` (or :class:`VadineanuDataset`) and the relevant
        samplers. Stage ``"predict"`` instead builds a :class:`PredictDataset`.
        """
        path = self.path
        if stage != "predict":
            self.dataset = TellDataset(transform=self.transform,
                                       model_name=self.model_name, path=path)

            # Create a 40 / 20 / 40 train / val / test split if none was given.
            if not any(self.train_indices) and not any(self.val_indices) and not any(self.test_indices):
                print('No split provided, making a new one')
                dataset_size = len(self.dataset)
                indices = np.arange(dataset_size)
                np.random.shuffle(indices)
                split = int(np.floor(0.4 * dataset_size))
                self.train_indices = indices[split:]
                self.test_indices = indices[:int(split/2)]
                self.val_indices = indices[int(split/2):int(split)]

        if stage == "fit":
            # Subset samplers let one shared dataset back multiple splits.
            self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                self.train_indices
            )
            self.valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                self.val_indices
            )

        if stage == "validate":
            self.valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                self.val_indices
            )

        if stage == "test":
            self.test_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                self.test_indices
            )

        if stage == "predict":
            self.predict_dataset = PredictDataset(
                path=self.path, countries=self.countries
            )

        if stage == "vadineanu":
            self.dataset = VadineanuDataset(
                path=path+"/vadineanu", transform=self.transform)

    def train_dataloader(self):
        """Return the dataloader over the training split."""
        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        """Return the dataloader over the validation split."""
        val_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.valid_sampler,
            num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self):
        """Return the dataloader over the test split."""
        test_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
        )
        return test_loader

    def predict_dataloader(self):
        """Return the dataloader over the tiled inference scenes."""
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
