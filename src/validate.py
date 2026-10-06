"""
This script provides three evaluation modes:

* :func:`validate` -- evaluate a checkpoint on the held-out self-labelled test
  split.
* :func:`vadineanu` -- evaluate a checkpoint on the external Vadineanu et al.
  dataset for cross-dataset generalisation. Results are logged to TensorBoard
  and summarised to a CSV.
* :func:`check_test_set` -- report the positive / negative class balance of the
  test split without running the model.

The mode is selected through the ``--vadineanu`` flag in the ``__main__`` block.
"""

import argparse
from torchvision.transforms import v2
import torch
import lightning as L
from model import LitModel
from data_loader import TellDataModule
import glob
import pandas as pd
from lightning.pytorch import loggers as pl_loggers


def prepare_model(model_name):
    """Instantiate a :class:`~model.LitModel` with a fixed learning rate.

    The learning rate is irrelevant for inference and is only required to build
    the module; the trained weights are restored later from the checkpoint.
    """
    base_model = model_name
    print(base_model)
    base_model = LitModel(base_model, 0.003)
    return base_model


def validate(args):
    """Evaluate a checkpoint on the self-labelled test split.

    Transformer backbones receive a ``384 x 384`` resize; convolutional
    backbones consume the native tile size.
    """
    device = "cuda"
    model = prepare_model(args.model_name)
    model = model.to(device)

    if 'vit' in args.model_name or 'swin' in args.model_name:
        transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Grayscale(num_output_channels=1),
                v2.Resize((384, 384)),
            ]
        )
    else:
        transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Grayscale(num_output_channels=1),
            ]
        )

    dataset = TellDataModule(path=args.dataset_path,
                             batch_size=args.batch_size,
                             transform=transform,
                             model_name=args.model_name)

    trainer = L.Trainer(
        deterministic=True,
    )
    # ``ckpt_path`` loads the trained weights before running the test loop.
    trainer.test(model=model, ckpt_path=args.checkpoint_path,
                 datamodule=dataset)


def vadineanu(args, checkpoint_path):
    """Evaluate a checkpoint on the external Vadineanu et al. dataset.

    Args:
        args: Parsed command-line arguments.
        checkpoint_path: Path to the ``.ckpt`` file to evaluate.

    Returns:
        The list of metric dictionaries returned by ``trainer.test``.
    """
    device = "cuda"
    model = prepare_model(args.model_name)
    model = model.to(device)

    if 'vit' in args.model_name or 'swin' in args.model_name:
        transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Grayscale(num_output_channels=1),
                v2.Resize((384, 384)),
            ]
        )
    else:
        transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Grayscale(num_output_channels=1),
            ]
        )

    dataset = TellDataModule(path=args.dataset_path,
                             batch_size=args.batch_size,
                             transform=transform,
                             model_name=args.model_name)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="/tell_detection/logs/validate/",
        name=args.model_name + '_validate_' + checkpoint_path.split('/')[-1],
        version=0,
        sub_dir='tb_logs'
    )

    # ``setup(stage="vadineanu")`` swaps the dataset for the external one.
    dataset.setup(stage="vadineanu")
    trainer = L.Trainer(
        deterministic=True,
        logger=tb_logger
    )
    return trainer.test(model=model, ckpt_path=checkpoint_path,
                 datamodule=dataset)


def check_test_set(args):
    """Print the number of positive and negative tiles in the test split.

    Useful as a sanity check on class balance before interpreting metrics.
    """
    if 'vit' in args.model_name or 'swin' in args.model_name:
        transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Grayscale(num_output_channels=1),
                v2.Resize((384, 384)),
            ]
        )
    else:
        transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Grayscale(num_output_channels=1),
            ]
        )

    dataset = TellDataModule(path=args.dataset_path,
                             batch_size=args.batch_size,
                             transform=transform,
                             model_name=args.model_name)

    dataset.setup("test")
    test_loader = dataset.test_dataloader()
    num_pos = 0
    num_neg = 0
    for i, item in enumerate(test_loader):
        num_pos += item["label"].sum()
        num_neg += args.batch_size - item["label"].sum()

    print('Num positives: ', num_pos, '\n')
    print('Num negatives: ', num_neg, '\n')


if __name__ == '__main__':
    print('Lets validate!')
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckp', '--checkpoint_path', required=True)
    parser.add_argument('-dpath', '--dataset_path', required=True)
    parser.add_argument('-b', '--batch_size', default=10, type=int)
    parser.add_argument('-m', '--model_name', required=True)
    parser.add_argument('-v', '--vadineanu', action="store_true")

    args = parser.parse_args()
    checkpoints = glob.glob(args.checkpoint_path + args.model_name + '/*.ckpt')
    results = []
    if args.vadineanu:
        # Evaluate every checkpoint of the given model on the external dataset
        # and aggregate the metrics into a single CSV.
        for checkpoint in checkpoints:
            scores = vadineanu(args, checkpoint)
            temp = {'checkpoint': checkpoint}
            for metric in scores[0].keys():
                temp[metric] = scores[0][metric]
            results.append(temp)

        result_df = pd.DataFrame(results)
        result_df.to_csv('/tell_detection/results/validate_' + args.model_name + '.csv')
    else:
        validate(args)
        check_test_set(args)
