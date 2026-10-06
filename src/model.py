"""
This module defines :class:`LitModel`, a thin Lightning wrapper around any
`timm <https://github.com/huggingface/pytorch-image-models>`_ backbone that
performs binary site / non-site classification. It handles the training,
validation, test and prediction loops, the optimiser / scheduler configuration
and the on-disk dumping of per-batch predictions.
"""

import geopandas as gpd
import lightning as L
import timm
import torch
import torch.optim as optim
import torchmetrics


class LitModel(L.LightningModule):
    """Lightning module wrapping a timm backbone for binary classification.

    Args:
        model_name: Any model name accepted by :func:`timm.create_model`
            (e.g. ``"resnetv2_50"``, ``"vit_base_patch16_384"``).
        lr: Learning rate used by the Adam optimiser.
    """

    def __init__(self, model_name, lr) -> None:
        super().__init__()
        # https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055-2/
        # ``in_chans=1`` because the maps are converted to single-channel input.
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=1, in_chans=1
        )
        self.lr = lr
        # Binary logits -> the loss applies the sigmoid internally.
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.model_name = model_name

    def training_step(self, batch, batch_idx):
        """Compute and log the training loss for one batch."""
        x = batch["image"]
        y = batch["label"]
        y_hat = self.model(x)
        loss = self.loss(y_hat.squeeze(), y.type(torch.float32))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute and log validation loss, recall, precision and F1."""
        x = batch["image"]
        y = batch["label"]
        y_hat = self.model(x)
        loss = self.loss(y_hat.squeeze(), y.type(torch.float32))

        # Precision, Recall, F1
        recall = torchmetrics.classification.BinaryRecall().to('cuda')
        precision = torchmetrics.classification.BinaryPrecision().to('cuda')
        f1 = torchmetrics.classification.BinaryF1Score().to('cuda')

        self.log("val_recall", recall(y_hat.squeeze(), y))
        self.log("val_precision", precision(y_hat.squeeze(), y))
        self.log("val_f1", f1(y_hat.squeeze(), y))
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """Compute and log test loss, recall, precision and F1."""
        x = batch["image"]
        y = batch["label"]
        y_hat = self.model(x)
        loss = self.loss(y_hat.squeeze(), y.type(torch.float32))

        # Precision, Recall, F1
        recall = torchmetrics.classification.BinaryRecall().to('cuda')
        precision = torchmetrics.classification.BinaryPrecision().to('cuda')
        f1 = torchmetrics.classification.BinaryF1Score().to('cuda')

        self.log("test_recall", recall(y_hat.squeeze(), y))
        self.log("test_precision", precision(y_hat.squeeze(), y))
        self.log("test_f1", f1(y_hat.squeeze(), y))
        self.log("test_loss", loss)

    def configure_optimizers(self):
        """Build the Adam optimiser and a one-cycle learning-rate schedule.

        The schedule is stepped every batch and spans the number of batches the
        trainer estimates for the whole run.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        stepping_batches = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Run inference on one tiled batch and persist the predictions.

        Each batch is written to a per-country CSV under
        ``/tell_detection/results`` so that the tiles can later be
        georeferenced and mosaicked into a probability raster.

        Returns:
            The raw logits produced by the model for the batch.
        """
        prediction = self.model(batch["image"])

        results = {
            "predictions": prediction.cpu().squeeze(),
            "country": batch["country"],
            "tif": batch["tif"],
            "widths": batch["widths"].cpu(),
            "heights": batch["heights"].cpu(),
        }
        temp_df = gpd.GeoDataFrame(results)
        temp_df.to_csv(
            "/tell_detection/results/" + batch["country"][0] + "/" + self.model_name +
            "/batch_" + str(batch_idx) + ".csv"
        )

        return prediction
