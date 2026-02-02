from io import BytesIO
from typing import Callable, List

import matplotlib.pyplot as plt
from PIL import Image
import torch
from pde_matching.data.trajectory import DataTrajectory


def render_figure(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="rgba")
    width, height = fig.canvas.get_width_height()
    image = Image.frombuffer("RGBA", (width, height), buffer.getbuffer(), "raw", "RGBA", 0, 1)
    return image


def render_worker(args):
    render_frame, frame = args
    fig = render_frame(*frame)
    try:
        return render_figure(fig)
    finally:
        plt.close(fig)


def render_animation(render_frame, frames, *, duration=200):
    imgs = []
    for frame in frames:
        imgs.append(render_worker((render_frame, frame)))
    
    if not imgs:
        raise ValueError("Cannot create animation from empty frame list")
        
    buffer = BytesIO()
    imgs[0].save(
        buffer, format="WebP", append_images=imgs[1:],
        save_all=True, loop=0, duration=duration,
        method=3, quality=95
    )
    return buffer.getvalue()


def animate_data_and_predictions(render_fn, trajectory, u_hat, interval=200, n_images=-1):
    from .animation_plots import create_animation
    return create_animation(render_fn, trajectory, u_hat, interval, n_images)
