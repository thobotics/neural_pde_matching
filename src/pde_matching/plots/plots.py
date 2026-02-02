from abc import ABC, abstractmethod

from pde_matching.data.trajectory import DataTrajectory
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .animation_plots import AnimationPlot
from .state_plots import MLFlowStatePlots, StatePlots, WandbLoggerStatePlots


class Plots(ABC):
    models = {}

    def __init__(self, config: DictConfig) -> None:
        self.kwargs_animation = OmegaConf.to_container(config.animation)
        self.kwargs_state = OmegaConf.to_container(config.state)

    @classmethod
    def register(cls, model_name):
        def decorator(model_cls):
            cls.models[model_name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def build(cls, config: DictConfig):
        model_type = config.name
        model_cls = cls.models.get(model_type)
        if model_cls:
            return model_cls(config)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    @property
    def state_plots(self) -> StatePlots:
        return self._state_plots

    @property
    def animation_plots(self) -> AnimationPlot:
        return self._animation_plots

    @property
    def n_images(self) -> int:
        return max(self.state_plots.n_images, self.animation_plots.n_images)

    def plot(
        self,
        batch: DataTrajectory,
        u_hat: Tensor,
        mask: Tensor,
        step,
        logger,
        render_fn,
    ) -> None:
        self.state_plots.plot(batch, u_hat, mask, step, logger, render_fn)
        self.animation_plots.plot(batch, u_hat, step, logger, render_fn)


@Plots.register("mlflow")
class MLFlowPlots(Plots):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._state_plots = MLFlowStatePlots(**self.kwargs_state)
        self._animation_plots = AnimationPlot.create("mlflow", **self.kwargs_animation)


@Plots.register("wandb")
class WandbLoggerPlots(Plots):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._state_plots = WandbLoggerStatePlots(**self.kwargs_state)
        self._animation_plots = AnimationPlot.create("wandb", **self.kwargs_animation)
