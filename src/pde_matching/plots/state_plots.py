from abc import ABC, abstractmethod

import numpy as np
from pde_matching.data.trajectory import DataTrajectory
from matplotlib import pyplot as plt
from torch import Tensor


class StatePlots(ABC):
    def __init__(self, n_images) -> None:
        self.n_images = n_images

    def _create_figure(self, batch: DataTrajectory, prediction: Tensor, mask: Tensor):
        u = batch.u[:, mask].detach().cpu().numpy()
        u_dot = batch.u_dot[:, mask].detach().cpu().numpy()
        u_hat = prediction[:, mask][..., : u.shape[-1]].detach().cpu().numpy()
        u_hat_dot = prediction[:, mask][..., u.shape[-1] :].detach().cpu().numpy()

        fig, axs = plt.subplots(self.n_images, 2, figsize=(10, self.n_images * 3))
        gt_label = ["Ground Truth"] + [None] * (u.shape[-1] - 1)
        pred_label = ["Prediction"] + [None] * (u.shape[-1] - 1)

        for i in range(self.n_images):
            # Plotting position
            axs[i, 0].plot(
                np.arange(u.shape[0]),
                u[:, i, :],
                color="green",
                label=gt_label,
            )
            axs[i, 0].plot(
                np.arange(u.shape[0]),
                u_hat[:, i, :],
                color="orange",
                label=pred_label,
            )

            # Plotting velocity
            axs[i, 1].plot(
                np.arange(u.shape[0]),
                u_dot[:, i, :],
                color="green",
                label=gt_label,
            )

            if u_hat_dot.shape[-1] != 0:
                axs[i, 1].plot(
                    np.arange(u.shape[0]),
                    u_hat_dot[:, i, :],
                    color="orange",
                    label=pred_label,
                )

        axs[0, 0].set_title("Position")
        axs[0, 1].set_title("Velocity")
        fig.tight_layout()

        return fig

    @abstractmethod
    def plot(
        self,
        batch: DataTrajectory,
        u_hat: Tensor,
        mask: Tensor,
        step,
        logger,
        render_fn,
    ) -> None:
        pass


class WandbLoggerStatePlots(StatePlots):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def plot(
        self,
        batch: DataTrajectory,
        u_hat: Tensor,
        mask: Tensor,
        step,
        logger,
        render_fn,
    ) -> None:
        pass


class MLFlowStatePlots(StatePlots):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def plot(
        self,
        batch: DataTrajectory,
        u_hat: Tensor,
        mask: Tensor,
        step,
        logger,
        render_fn,
    ) -> None:
        figure = self._create_figure(batch, u_hat, mask)
        logger.experiment.log_figure(
            logger._run_id, figure, f"state/{step:06d}_state.png"
        )
        plt.close(figure)
