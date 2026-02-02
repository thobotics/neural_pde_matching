import copy
from typing import List

import pytorch_lightning as pl
import torch
from pde_matching.data.trajectory import DataTrajectory
from pde_matching.plots import Plots
import numpy as np
import matplotlib.pyplot as plt


class PlotsSingleStepCallback(pl.Callback):
    def __init__(self, config):
        super().__init__()

        self.plots = Plots.build(config)
        self.ready = True
        self.n_images = self.plots.kwargs_animation['n_images']

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ready = False

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ready = True

    def _plot(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str = "val"):
        batch = self._get_batch(trainer, pl_module, stage=stage).to("cpu")
        mask = pl_module._output_mask(batch).to("cpu")

        with torch.no_grad():
            pl_module.eval()

            # Move model to cpu to avoid cuda memory issues
            model = copy.deepcopy(pl_module.model).to("cpu")
            predictions = model(batch, mask=mask)

        if stage == "val":
            render_fn = trainer.datamodule.val_data.render
        elif stage == "test":
            render_fn = trainer.datamodule.test_data.render
        else:
            raise ValueError(f"Unknown stage: {stage}")

        u_gt = batch[-1].u.detach().cpu().numpy()
        u_hat = predictions.detach().cpu().numpy()
        u_dot = batch[-1].u_dot.detach().cpu().numpy()
        properties = batch[-1].u_properties.cpu().detach().numpy()

        umin = np.min(u_gt, axis=(0, 1))
        umax = np.max(u_gt, axis=(0, 1))
        
        fig = render_fn(
            u=u_gt[0],
            u_dot=u_dot[0],
            u_hat=u_hat[0],
            u_dot_hat=u_dot[0],
            t=0,
            data_batch=batch[-1].data_list[0], 
            properties=properties[0], 
            umin=umin,
            umax=umax,
            udot_min=None,
            udot_max=None,
            n_images=self.n_images,
        )

        if trainer.logger is not None:
            if hasattr(trainer.logger, 'log_image'):
                # WandB logger
                trainer.logger.log_image(
                    key=f"prior_net_visualization", 
                    images=[fig],
                    step=trainer.global_step
                )
            elif hasattr(trainer.logger.experiment, 'log_figure'):
                # MLFlow logger
                trainer.logger.experiment.log_figure(
                    trainer.logger._run_id,
                    fig,
                    f"prior_net/{trainer.global_step:06d}_predictions.png"
                )

        fig.clear()
        plt.close(fig) 

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if not self.ready:
            return

        self._plot(trainer, pl_module, stage="val")

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if not self.ready:
            return

        self._plot(trainer, pl_module, stage="test")

    def _get_batch(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> DataTrajectory:
        batch = trainer.datamodule.get_interesting_batch(self.n_images, stage=stage)

        if batch is not None:
            batch = [batch[0], batch[-1]]
            batch = pl_module._batch_to_data_trajectory(batch)

        return batch
