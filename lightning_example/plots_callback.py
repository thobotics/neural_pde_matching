import copy
from typing import List

import pytorch_lightning as pl
import torch
from pde_matching.data.trajectory import DataTrajectory
from pde_matching.plots import Plots


class PlotsCallback(pl.Callback):
    def __init__(self, config):
        super().__init__()

        self.plots = Plots.build(config)
        self.ready = True
        self.n_images = self.plots.n_images
        self.plot_frequency = config.plot_at_every_n_val_step

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ready = False

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ready = True

    def _plot(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str = "val"):
        batch = self._get_batch(trainer, pl_module, stage).to("cpu")
        mask = pl_module._output_mask(batch).to("cpu")

        if stage == "val":
            env = trainer.datamodule.val_data
        elif stage == "test":
            env = trainer.datamodule.test_data
        else:
            raise ValueError(f"Unknown stage: {stage}")

        with torch.no_grad():
            pl_module.eval()

            # Move model to cpu to avoid cuda memory issues
            model = copy.deepcopy(pl_module.model).to("cpu")
            if pl_module.hparams.rollout_val:
                state_hat, state_dot_hat = env.rollout(
                    model, batch, mask=mask
                )
            else:
                state_hat = model(batch, mask=mask)[: len(batch) - 1]
                state_dot_hat = None

            state_hat = torch.cat(
                [batch.state_vector[0].unsqueeze(0), state_hat], dim=0
            )
            if state_dot_hat is not None:
                state_dot_hat = torch.cat(
                    [torch.zeros(1, *state_dot_hat.shape[1:]), state_dot_hat], dim=0
                )

        self.plots.plot(
            batch,
            (state_hat, state_dot_hat),
            mask,
            step=trainer.global_step,
            logger=trainer.logger,
            render_fn=env.render,
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if not self.ready or (trainer.current_epoch + 1) % (trainer.check_val_every_n_epoch * self.plot_frequency) != 0:
            return

        self._plot(trainer, pl_module, "val")

    def on_test_end(self, trainer, pl_module):
        if not self.ready:
            return

        self._plot(trainer, pl_module, "test")

    def _get_batch(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> DataTrajectory:
        batch = trainer.datamodule.get_interesting_batch(self.n_images, stage=stage)

        if batch is not None:
            batch = pl_module._batch_to_data_trajectory(batch)

        return batch
