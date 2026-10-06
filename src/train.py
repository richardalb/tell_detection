"""
This script trains a :class:`~model.LitModel` on the self-labelled
site / non-site dataset using a stratified k-fold protocol. For every fold it
constructs a fresh :class:`~data_loader.TellDataModule` with the fold's
train / validation / test index sets, configures heavy augmentation, and runs
Lightning's :meth:`~lightning.pytorch.Trainer.fit`.

Example:
    ``python train.py -m resnetv2_50 -cn resnetv2_50 -lr 0.003 -e 100``

The checkpoints, TensorBoard logs and results are written under
``/tell_detection`` following the layout used on the HPC cluster.
"""

import argparse

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import loggers as pl_loggers
from torchvision.transforms import v2

from data_loader import TellDataModule, TellDataset
from model import LitModel


def prepare_model(model_name, lr):
    """Instantiate the Lightning model for ``model_name`` with learning rate ``lr``."""
    base_model = model_name
    print(base_model)
    base_model = LitModel(base_model, lr)
    return base_model


def main(args):
    """Train the model across ``args.n_folds`` folds.

    A single model instance is created up front and trained fold by fold. The
    dataset indices are shuffled once with a fixed seed so that the folds are
    reproducible, then rolled to produce, for each fold, a held-out test slice,
    a validation slice and the remaining training slice.

    Args:
        args: Parsed command-line arguments (see the ``__main__`` block).
    """
    device = "cuda"
    model = prepare_model(args.model_name, args.learning_rate)
    model = model.to(device)

    # Transformer backbones are resized to 384 x 384 and use the transformation
    # pipeline below; convolutional backbones use the identical pipeline in the
    # ``else`` branch. Both branches are kept explicit for readability.
    if 'swin' in args.model_name or 'vit' in args.model_name:
        transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.1),
                v2.RandomVerticalFlip(p=0.1),
                v2.RandomApply(torch.nn.ModuleList([
                    v2.RandomResizedCrop(
                        size=400, scale=(0.7, 1.0))
                ]), p=0.1),
                v2.RandomApply(torch.nn.ModuleList([
                    v2.RandomRotation(degrees=(0, 180))
                ]), p=0.1),
                v2.RandomApply(torch.nn.ModuleList([
                    v2.RandomAffine(degrees=(0, 180),
                                    translate=(0.1, 0.3),
                                    scale=(0.5, 0.75))
                ]), p=0.1),
                v2.RandomPerspective(distortion_scale=0.5, p=0.1),
                v2.RandomApply(torch.nn.ModuleList([
                    v2.ElasticTransform(),
                ]), p=0.1),
                v2.RandomApply(torch.nn.ModuleList([
                    v2.GaussianBlur(
                        kernel_size=9, sigma=(0.1, 2.0))
                ]), p=0.1),
                v2.RandomApply(torch.nn.ModuleList([
                    v2.GaussianNoise(
                        mean=0.0, sigma=0.1, clip=True)
                ]), p=0.1),
                v2.RandomSolarize(threshold=0.7, p=0.1),
                v2.RandomAdjustSharpness(sharpness_factor=0.67, p=0.1),
                v2.RandomEqualize(p=0.1),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Grayscale(num_output_channels=1),
                v2.RandomInvert(p=0.5),
                v2.Resize((384, 384)),
                v2.RandomErasing(p=0.1),
            ]
        )

    else:
        transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.1),
                v2.RandomVerticalFlip(p=0.1),
                v2.RandomApply(torch.nn.ModuleList([
                    v2.RandomResizedCrop(
                        size=400, scale=(0.7, 1.0))
                ]), p=0.1),
                v2.RandomApply(torch.nn.ModuleList([
                    v2.RandomRotation(degrees=(0, 180))
                ]), p=0.1),
                v2.RandomApply(torch.nn.ModuleList([
                    v2.RandomAffine(degrees=(0, 180),
                                    translate=(0.1, 0.3),
                                    scale=(0.5, 0.75))
                ]), p=0.1),
                v2.RandomPerspective(distortion_scale=0.5, p=0.1),
                v2.RandomApply(torch.nn.ModuleList([
                    v2.ElasticTransform(),
                ]), p=0.1),
                v2.RandomApply(torch.nn.ModuleList([
                    v2.GaussianBlur(
                        kernel_size=9, sigma=(0.1, 2.0))
                ]), p=0.1),
                v2.RandomApply(torch.nn.ModuleList([
                    v2.GaussianNoise(
                        mean=0.0, sigma=0.1, clip=True)
                ]), p=0.1),
                v2.RandomSolarize(threshold=0.7, p=0.1),
                v2.RandomAdjustSharpness(sharpness_factor=0.67, p=0.1),
                v2.RandomEqualize(p=0.1),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Grayscale(num_output_channels=1),
                v2.RandomInvert(p=0.5),
                v2.Resize((384, 384)),
                v2.RandomErasing(p=0.1),
            ]
        )

    # Build reproducible folds from a single shuffled index array.
    len_dataset = len(TellDataset(path=args.dataset_path))
    indices = np.arange(len_dataset)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_index = 1 / args.n_folds
    split = int(np.floor(fold_index * len_dataset))
    for k in range(args.n_folds):
        # Roll the shuffled indices so each fold uses a different held-out slice.
        rolled_indices = np.roll(indices, split)
        train_indices = rolled_indices[int(2 * split):]
        test_indices = indices[:split]
        val_indices = indices[split:int(2*split)]
        dataset = TellDataModule(path=args.dataset_path,
                                 batch_size=args.batch_size,
                                 transform=transform,
                                 model_name=args.model_name,
                                 train_indices=train_indices,
                                 val_indices=val_indices,
                                 test_indices=test_indices,
                                 num_workers=args.num_workers,)

        # Separate TensorBoard run per model / fold / learning rate.
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir="/tell_detection/logs/train/",
            name=args.model_name + '_fold_' + str(k) + '_lr_' + str(args.learning_rate),
            version=args.log_version,
            sub_dir='tb_logs'
        )

        num_epochs = args.epochs

        if args.long_training:
            # Long runs checkpoint every 10 epochs and keep all checkpoints.
            checkpoint_callback = ModelCheckpoint(
                save_top_k = -1,
                dirpath="/tell_detection/checkpoints/long/" + args.checkpoint_name,
                filename="{epoch:02d}-{val_loss:.5f}-" + "_lr_" + str(args.learning_rate),
                every_n_epochs=10,
            )
            callbacks = [checkpoint_callback]
        else:
            # Default protocol: keep the best 5 checkpoints and stop early when
            # the validation loss stops improving.
            checkpoint_callback = ModelCheckpoint(
                save_top_k=5,
                monitor="val_loss",
                mode="min",
                dirpath="/tell_detection/checkpoints/" + args.checkpoint_name,
                filename="{epoch:02d}-{val_loss:.5f}-fold_" + str(k) + '_lr_' + str(args.learning_rate),
            )
            early_stop_callback = EarlyStopping(
                monitor="val_loss", min_delta=0.01, patience=4, verbose=True, mode="min"
            )
            callbacks = [early_stop_callback, checkpoint_callback]

        trainer = L.Trainer(
            max_epochs=num_epochs,
            callbacks=callbacks,
            deterministic=True,
            logger=tb_logger,
        )
        trainer.fit(model, datamodule=dataset)


if __name__ == "__main__":
    print("Let's train!!!")
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default="vgg19.tv_in1k")
    parser.add_argument('-cn', '--checkpoint_name', required=True)
    parser.add_argument('-b', '--batch_size', default=10, type=int)
    parser.add_argument('-lv', '--log_version', type=int)
    parser.add_argument('-lr', '--learning_rate', type=float, required=True)
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-dpath', '--dataset_path',
                        default='./data')
    parser.add_argument('-nf', '--n_folds', default=5, type=int)
    parser.add_argument('-nw', '--num_workers', default=8, type=int)
    parser.add_argument('-l', '--long_training', action="store_true")
    args = parser.parse_args()
    main(args)
