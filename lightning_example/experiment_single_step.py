import pytorch_lightning as pl
import torch
from pde_matching.data.trajectory import DataTrajectory
from pde_matching.envs import Env
from pde_matching.utils.metrics import TrajectoryMetrics
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import Optimizer
from torch_geometric.data.data import BaseData
from .experiment import Experiment  # Importing the base Experiment class


class ExperimentSingleStep(Experiment):
    def __init__(self, config: DictConfig, model: torch.nn.Module) -> None:
        super(ExperimentSingleStep, self).__init__(config, model)
        self.save_hyperparameters(config)

    def training_step(self, batch: BaseData, batch_idx: int):
        batch = [batch[0], batch[-1]]
        targets = self._batch_to_data_trajectory(batch)
        inputs = targets.clone()

        predictions = self.model(
            inputs, 
            mask=self._output_mask(inputs),
        )

        T = len(targets)
        outputs = self.model.loss(
            predictions[: T - 1], targets[1:], mask=self._loss_mask(targets[1:])
        )

        self.log(
            "train/loss",
            outputs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size,
        )

        train_metrics = {
            "train/mse_pos": outputs["mse_pos"],
            "train/mse_vel": outputs["mse_vel"],
        }

        self.log_dict(
            train_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.hparams.batch_size,
        )
        return outputs["loss"]
    
    def validation_step(self, batch, batch_idx):
        batch = [batch[0], batch[-1]]
        targets = self._batch_to_data_trajectory(batch)
        inputs = targets.clone()

        predictions = self.model(
            inputs, mask=self._output_mask(inputs)
        )

        T = len(targets)
        outputs = self.model.loss(
            predictions[: T - 1], targets[1:], mask=self._loss_mask(targets[1:])
        )

        val_metrics = {
            "val/loss": outputs["loss"],
            "val/mse_pos": outputs["mse_pos"],
            "val/mse_vel": outputs["mse_vel"],
        }

        self.log_dict(
            val_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.hparams.batch_size,
        )
        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
