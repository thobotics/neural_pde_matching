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


class Experiment(pl.LightningModule):
    def __init__(self, config: DictConfig, model: torch.nn.Module) -> None:
        super(Experiment, self).__init__()
        self.save_hyperparameters(config)

        self.model = model
        
        # Curriculum training
        self.is_curriculum = self.hparams.get("curriculum_training", {}).get("enabled", False)
        if self.is_curriculum:
            max_epochs = self.hparams.epochs
            end_epoch = self.hparams.curriculum_training.get("end_epoch", max_epochs // 3)
            min_T = self.hparams.curriculum_training.get("start_timestep", 3)
            step_size = self.hparams.curriculum_training.get("step_size", 1)
            curriculum_Ts = torch.arange(
                min_T, self.hparams.dataset["window_size"] + 1, step=step_size
            ).long().tolist()
            curriculum_Ts += [self.hparams.dataset["window_size"]]

            repeated_curriculum = []
            for i in range(len(curriculum_Ts)):
                tmp_arr = [curriculum_Ts[i]] * (end_epoch // len(curriculum_Ts) + 1)
                repeated_curriculum.extend(tmp_arr)

            curriculum_Ts = repeated_curriculum
            self.curriculum_Ts = curriculum_Ts + [
                self.hparams.dataset["window_size"]
            ] * (max_epochs - len(curriculum_Ts))

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer: Optimizer = instantiate(self.hparams.optimizer, params=params)
        if self.hparams.scheduler is not None:
            scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def forward(self, batch: DataTrajectory, env: Env, rollout: bool = False) -> Tensor:
        if rollout:
            return self.rollout(batch, env)
        else:
            return self.execute(batch, env)

    def execute(self, batch: DataTrajectory, env: Env) -> Tensor:
        return self.model(
            batch,
            mask=self._output_mask(batch),
            dt=env.dt,
        )

    def rollout(self, batch: DataTrajectory, env: Env) -> Tensor:
        predictions, _ = env.rollout(
            self.model,
            batch,
            mask=self._output_mask(batch),
            is_train=self.training,
        )
        return predictions

    def training_step(self, batch: BaseData, batch_idx: int):
        targets = self._batch_to_data_trajectory(batch)
        inputs = targets.clone()
        T = len(targets) if not self.is_curriculum else self.curriculum_Ts[self.current_epoch]

        predictions = self(
            inputs[:T], self.trainer.datamodule.train_data, self.hparams.rollout_train
        )

        outputs = self.model.loss(
            predictions[: T - 1], targets[1:T], mask=self._loss_mask(targets[1:T])
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

        # Compute the commulative and nth mse
        cum_mse = TrajectoryMetrics.cummulative_mse(outputs["mse_sample_wise"])

        step = self.hparams.val_per_steps
        cum_t = [t for t in range(step, T, step)] + [T - 1]
        for t in cum_t:
            train_metrics[f"train/cum_mse_{t:02d}"] = cum_mse[t - 1]

            nth_mse = TrajectoryMetrics.nth_mse(outputs["mse_sample_wise"], t - 1)
            train_metrics[f"train/nth_mse_{t:02d}"] = nth_mse

        self.log_dict(
            train_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.hparams.batch_size,
        )
        return outputs["loss"]

    def _validation_step(self, batch: BaseData, env: Env):
        targets = self._batch_to_data_trajectory(batch)
        
        # import time
        # start_time = time.time()
        predictions = self(
            targets, env, self.hparams.rollout_val
        )
        # forward_time = time.time() - start_time

        T = len(targets)
        mask = self._loss_mask(targets[1:])
        outputs = self.model.loss(predictions[: T - 1], targets[1:], mask=mask)

        # Create the validation metrics
        val_metrics = {
            "val/loss": outputs["loss"],
            "val/mse_pos": outputs["mse_pos"],
            "val/mse_vel": outputs["mse_vel"],
            # "val/forward_time": forward_time,
        }

        # print(f"Num nodes, edges: {targets.data_list[0].num_nodes, targets.data_list[0].num_edges}")
        # print(f"Validation forward time: {forward_time:.4f} seconds")

        # Compute the commulative and nth mse
        cum_mse = TrajectoryMetrics.cummulative_mse(outputs["mse_sample_wise"])

        step = self.hparams.val_per_steps
        cum_t = [t for t in range(step, T, step)] + [T - 1]
        for t in cum_t:
            val_metrics[f"val/cum_mse_{t:02d}"] = cum_mse[t - 1]

            nth_mse = TrajectoryMetrics.nth_mse(outputs["mse_sample_wise"], t - 1)
            val_metrics[f"val/nth_mse_{t:02d}"] = nth_mse

        # Compute additional metrics if available
        if hasattr(env, "additional_metrics"):
            additional_metrics = env.additional_metrics(
                predictions[: T - 1], targets[1:], mask=mask
            )
            for key, value in additional_metrics.items():
                val_metrics[f"val/{key}"] = value

        self.log_dict(
            val_metrics,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )
        return outputs["loss"]
    
    def validation_step(self, batch: BaseData, batch_idx: int):
        self._validation_step(batch, self.trainer.datamodule.val_data)

    def test_step(self, batch: BaseData, batch_idx: int):
        self._validation_step(batch, self.trainer.datamodule.test_data)

    def _batch_to_data_trajectory(self, batch: BaseData) -> DataTrajectory:
        return DataTrajectory.from_batch_list(
            batch,
            self.trainer.datamodule.input_keys,
            self.trainer.datamodule.output_keys,
            self.trainer.datamodule.standardizer,
        )

    def _loss_mask(self, mesh_trajectory: DataTrajectory) -> Tensor:
        return self.trainer.datamodule.loss_mask(mesh_trajectory.u_properties)

    def _output_mask(self, mesh_trajectory: DataTrajectory) -> Tensor:
        return self.trainer.datamodule.output_mask(mesh_trajectory.u_properties)

    def on_train_epoch_start(self) -> None:
        if self.hparams.dataset["shuffle"]:
            self.trainer.datamodule.train_data.shuffle()
            # self.trainer.datamodule.val_data.shuffle()
