import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from io import BytesIO

import torch
import wandb
import matplotlib.pyplot as plt

from .utils import render_figure


def create_animation(render_fn, trajectory, predictions, duration_ms=200, max_images=-1, skip_frames=1):
    positions_gt = trajectory.u.cpu().detach().numpy()
    velocities_gt = trajectory.u_dot.cpu().detach().numpy()
    properties = trajectory.u_properties.cpu().detach().numpy()
    
    if isinstance(predictions, tuple):
        positions_pred, _ = predictions
        predictions = positions_pred

    velocities_pred = None
    if positions_pred.shape[-1] > trajectory.u.shape[-1]:
        positions_pred, velocities_pred = trajectory.split(predictions)

    if velocities_pred is None:
        velocities_pred = positions_pred[1:] - positions_pred[:-1]
        velocities_pred = torch.cat(
            [velocities_pred, velocities_pred[-1].unsqueeze(0)],
            dim=0
        )
    
    positions_pred = positions_pred.cpu().detach().numpy()
    velocities_pred = velocities_pred.cpu().detach().numpy()
        
    
    position_min = positions_gt.min(axis=(0, 1))
    position_max = positions_gt.max(axis=(0, 1))
    velocity_min = velocities_gt.min(axis=(0, 1))
    velocity_max = velocities_gt.max(axis=(0, 1))
    
    frames = []
    num_frames = positions_gt.shape[0]

    for i in range(0, num_frames, skip_frames):
        fig = render_fn(
            positions_gt[i], velocities_gt[i], 
            positions_pred[i], velocities_pred[i],
            i, trajectory.data_list[i], properties[i],
            position_min, position_max,
            velocity_min, velocity_max,
            max_images
        )
        
        try:
            frames.append(render_figure(fig))
        finally:
            plt.close(fig)
    
    if not frames:
        raise ValueError("No frames to animate")
        
    buffer = BytesIO()
    frames[0].save(
        buffer, format="WebP", append_images=frames[1:],
        save_all=True, loop=0, duration=duration_ms,
        method=3, quality=95
    )
    
    return buffer.getvalue()


class AnimationPlot(ABC):
    registry = {}
    
    def __init__(self, fps=20, n_images=10, skip_frames=1, save_local_path=None):
        self.interval = int(1000 / fps)
        self.n_images = n_images
        self.skip_frames = skip_frames
        self.save_local_path = save_local_path

    @classmethod
    def register(cls, name):
        def decorator(impl_class):
            cls.registry[name] = impl_class
            return impl_class
        return decorator
    
    @classmethod
    def create(cls, logger_type, **kwargs):
        if logger_type not in cls.registry:
            available = ", ".join(cls.registry.keys())
            raise ValueError(f"Unknown logger type '{logger_type}'. Available: {available}")
        return cls.registry[logger_type](**kwargs)

    @abstractmethod
    def save_animation_to_logger_format(self, animation_bytes, format, step):
        pass

    @abstractmethod
    def log_animation_to_backend(self, batch, u_hat, step, logger, render_fn):
        pass

    def plot(self, batch, u_hat, step, logger, render_fn):
        return self.log_animation_to_backend(batch, u_hat, step, logger, render_fn)

    def _create_animation(self, render_fn, batch, u_hat, step):
        animation_bytes = create_animation(render_fn, batch, u_hat, self.interval, self.n_images, self.skip_frames)
        animation_object = self.save_animation_to_logger_format(animation_bytes, "gif", step)
        return {"animation": animation_object}


@AnimationPlot.register("wandb")
class WandbLoggerAnimationPlot(AnimationPlot):
    def save_animation_to_logger_format(self, animation_bytes, format, step):
        if self.save_local_path:
            file_path = Path(self.save_local_path).parent / f"wandb_animation_{step:06d}.{format}"
            with open(file_path, "wb") as f:
                f.write(animation_bytes)
            
            print(f"\nSaved animation to {file_path}")
            return wandb.Image(str(file_path))
        
        temp_file = tempfile.NamedTemporaryFile(
            prefix=f"wandb_animation_{step:06d}_", 
            suffix=f".{format}", 
            delete=False
        )
        with temp_file as f:
            f.write(animation_bytes)
        return wandb.Image(str(Path(temp_file.name)))

    def log_animation_to_backend(self, batch, u_hat, step, logger, render_fn):
        animation_dict = self._create_animation(render_fn, batch, u_hat, step)
        log_data = {"step": step}
        for key, animation_obj in animation_dict.items():
            log_data[f"image_{key}"] = animation_obj
        logger.log_metrics(log_data, step=step)

    def get_image_from_buffer(self, buffer, format, step):
        return self.save_animation_to_logger_format(buffer, format, step)


@AnimationPlot.register("mlflow")
class MLFlowAnimationPlot(AnimationPlot):
    def save_animation_to_logger_format(self, animation_bytes, format, step):
        temp_file = tempfile.NamedTemporaryFile(
            prefix=f"mlflow_animation_{step:06d}_",
            suffix=f".{format}",
            delete=False
        )
        with temp_file as f:
            f.write(animation_bytes)
        return str(Path(temp_file.name))

    def log_animation_to_backend(self, batch, u_hat, step, logger, render_fn):
        animation_dict = self._create_animation(render_fn, batch, u_hat, step)
        for key, file_path in animation_dict.items():
            logger.experiment.log_artifact(
                logger._run_id, file_path, f"image_{key}"
            )

    def get_image_from_buffer(self, buffer, format, step):
        return self.save_animation_to_logger_format(buffer, format, step)
