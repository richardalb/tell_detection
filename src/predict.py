"""
This script loads a trained checkpoint, tiles one or more country directories
of Corona satellite imagery with :class:`~data_loader.PredictDataset`, and runs
Lightning's prediction loop. Predictions are persisted batch by batch by
:meth:`~model.LitModel.predict_step` under
``/tell_detection/results/<country>/<model>/``.

Example:
    ``python predict.py -m resnetv2_50 -b 50 -cp best.ckpt --countries Syria``
"""

import argparse

import lightning as L

from data_loader import TellDataModule
from model import LitModel


def predict(countries, base_path, model_name, checkpoint, batch_size):
    """Run sliding-window inference over the requested countries.

    Args:
        countries: List of country sub-directories under ``base_path``.
        base_path: Root directory of the satellite imagery.
        model_name: Backbone name; must match the checkpoint's architecture.
        checkpoint: Path to the Lightning ``.ckpt`` file to load.
        batch_size: Number of tiles per inference batch.

    Returns:
        The list of prediction tensors returned by ``trainer.predict``.
    """
    dataset = TellDataModule(
        path=base_path, countries=countries, batch_size=batch_size, num_workers=7)
    # The "predict" stage builds a tiled PredictDataset rather than the
    # classification dataset.
    dataset.setup(stage="predict")
    predict_loader = dataset.predict_dataloader()
    # ``lr`` is required to construct the module but is unused for inference;
    # the weights are restored from the checkpoint.
    model = LitModel.load_from_checkpoint(
        checkpoint,
        model_name=model_name,
        lr=0.01,
        map_location="cpu",
    )
    trainer = L.Trainer(accelerator="gpu", devices=1, num_nodes=1)

    print("got to making predictions")
    return trainer.predict(model, predict_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-cp', '--checkpoint', default="/tell_detection/checkpoints/resnetv2_50/best.ckpt")
    parser.add_argument('-p', '--path', default="/tell_detection/data/corona_images")
    parser.add_argument('-b', '--batch_size', required=True, type=int)
    parser.add_argument('--countries', nargs='+')
    args = parser.parse_args()

    predict(countries=args.countries, base_path=args.path, model_name=args.model,
            checkpoint=args.checkpoint, batch_size=args.batch_size)
