import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from pde_matching.envs.visualizer import Visualizer


class ImpactPlateVisualizer(Visualizer):
    def __init__(
        self,
        output_mask_fn,
        n_columns=3,
        umin=None,
        umax=None,
        udot_min=None,
        udot_max=None,
        overlayed_ground_truth=False,
        plot_stress=True,
        stress_only=False,
        xy_min=(0.0, 0.0),
        xy_max=(30.0, 40.0),
        **kwargs
    ):
        self.output_mask_fn = output_mask_fn
        self.n_columns = n_columns
        self.overlayed_ground_truth = overlayed_ground_truth
        self.plot_stress = plot_stress
        self.stress_only = stress_only
        self.stress_dim_idx = 3  if not stress_only else 0
        self.xy_min = xy_min
        self.xy_max = xy_max
        if umin is not None:
            self.umin = np.array([v if v is not None else None for v in umin])
        else:
            self.umin = None
        if umax is not None:
            self.umax = np.array([v if v is not None else None for v in umax])
        else:
            self.umax = None
        if udot_min is not None:
            self.udot_min = np.array([v if v is not None else None for v in udot_min])
        else:
            self.udot_min = None
        if udot_max is not None:
            self.udot_max = np.array([v if v is not None else None for v in udot_max])
        else:   
            self.udot_max = None

    def visualize(self, u, u_dot, u_hat, u_dot_hat, t, data_batch, properties, 
                  umin, umax, udot_min, udot_max, n_images=-1, padding=0.025):
        from .impact_plate import NodeType

        u_tensor = torch.tensor(u)
        u_hat_tensor = torch.tensor(u_hat)
        mask = self.output_mask_fn(torch.tensor(properties))
        u_hat_tensor[..., :mask.shape[1]] = torch.where(
            mask,
            u_hat_tensor[..., :mask.shape[1]],
            u_tensor[..., :mask.shape[1]]
        )
        u_hat = u_hat_tensor.cpu().detach().numpy()

        # Simple data preparation
        if hasattr(data_batch, "batch") and data_batch.batch is not None:
            batch_idx = data_batch.batch.cpu().numpy()
            data_list = data_batch.to_data_list()
            u_list = [u[batch_idx == i] for i in range(n_images if n_images != -1 else len(data_list))]
            u_hat_list = [u_hat[batch_idx == i] for i in range(len(u_list))]
            u_dot_list = [u_dot[batch_idx == i] for i in range(len(u_list))]
            u_dot_hat_list = [u_dot_hat[batch_idx == i] for i in range(len(u_list))]
            properties_list = [properties[batch_idx == i] for i in range(len(u_list))]
        else:
            data_list = [data_batch]
            u_list = [u]
            u_hat_list = [u_hat]
            u_dot_list = [u_dot]
            u_dot_hat_list = [u_dot_hat]
            properties_list = [properties]

        num_images = len(u_list)

        # Helper functions
        def get_node_masks(props):
            """Get masks for normal and obstacle nodes only."""
            node_types = np.argmax(props[..., :NodeType.SIZE], axis=1)
            normal_mask = node_types == NodeType.NORMAL
            obstacle_mask = node_types == NodeType.OBSTACLE
            return normal_mask, obstacle_mask

        def get_stress_colors(u_data, faces, stress_min=None, stress_max=None):
            if len(faces) == 0 or not self.plot_stress:
                return ["orange"] * len(faces)
            
            # Extract face stresses
            face_stresses = np.array([np.mean(u_data[face, self.stress_dim_idx]) for face in faces])
            face_stresses = np.clip(face_stresses, 0, None)  # Ensure non-negative
            
            # Log scale
            face_stresses = (face_stresses + 1e-8) ** 0.4
            
            # Normalize using global bounds
            if stress_min is not None and stress_max is not None:
                stress_min_log = (stress_min + 1e-8) ** 0.4
                stress_max_log = (stress_max + 1e-8) ** 0.4
                normalized = (face_stresses - stress_min_log) / (stress_max_log - stress_min_log)
            else:
                normalized = (face_stresses - face_stresses.min()) / (face_stresses.max() - face_stresses.min())
            
            return cm.get_cmap("jet")(normalized)

        def create_2d_mesh_fast(ax, positions, faces, colors):
            """Create 2D mesh using vectorized operations for speed."""
            if len(faces) == 0:
                return
            
            # Extract 2D coordinates for all faces at once (vectorized)
            face_coords = positions[faces, :2]  # Shape: (n_faces, 3_or_4_vertices, 2)
            
            # Create all polygons at once
            polygons = [Polygon(coords, closed=True) for coords in face_coords]
            
            # Handle colors efficiently
            if isinstance(colors, str):
                face_colors = [colors] * len(faces)
            else:
                face_colors = colors
            
            # Single collection creation
            collection = PatchCollection(polygons, facecolors=face_colors, 
                                       edgecolors='black', linewidths=0.5, alpha=0.7)
            ax.add_collection(collection)

        # Create figure
        if not self.overlayed_ground_truth:
            fig = plt.figure(figsize=(6, 6 * num_images))
            axes = [fig.add_subplot(num_images, 2, i * 2 + j + 1) 
                   for i in range(num_images) for j in range(2)]
        else:
            fig = plt.figure(figsize=(8, 6 * num_images))
            axes = [fig.add_subplot(num_images, 1, i + 1) for i in range(num_images)]

        for i in range(num_images):
            data = data_list[i]
            normal_mask, obstacle_mask = get_node_masks(properties_list[i])
            
            # Get faces (cells) from data
            faces = data.cells.cpu().numpy() if hasattr(data, 'cells') else np.array([])
            
            if not self.overlayed_ground_truth:
                # Side-by-side: prediction | ground truth
                ax_pred, ax_gt = axes[i*2], axes[i*2+1]
                
                # Prediction plot
                if len(faces) > 0:
                    colors = get_stress_colors(u_dot_hat_list[i], faces, udot_min[self.stress_dim_idx], udot_max[self.stress_dim_idx])

                    if self.stress_only:
                        create_2d_mesh_fast(ax_pred, data_list[i].pos, faces, colors)
                    else:
                        create_2d_mesh_fast(ax_pred, u_hat_list[i], faces, colors)
                
                ax_pred.set_title(f"Prediction t={t:.2f}")
                ax_pred.set_aspect('equal')
                
                # Ground truth plot
                if len(faces) > 0:
                    colors = get_stress_colors(u_dot_list[i], faces, udot_min[self.stress_dim_idx], udot_max[self.stress_dim_idx])
                    create_2d_mesh_fast(ax_gt, data_list[i].pos, faces, colors)
                
                ax_gt.set_title(f"Ground Truth t={t:.2f}")
                ax_gt.set_aspect('equal')
                
            else:
                # Overlay mode
                ax = axes[i]
                
                # Prediction (colored)
                if len(faces) > 0:
                    colors = get_stress_colors(u_hat_list[i], faces, udot_min[3], udot_max[3])
                    create_2d_mesh_fast(ax, u_hat_list[i], faces, colors)
                    
                    # Ground truth wireframe
                    patches = []
                    for face in faces:
                        face_coords = u_list[i][face, :2]  # Only x, y
                        polygon = Polygon(face_coords, closed=True)
                        patches.append(polygon)
                    
                    if patches:
                        gt_collection = PatchCollection(patches, facecolors='none', 
                                                      edgecolors='black', linewidths=1.5, alpha=1.0)
                        ax.add_collection(gt_collection)
                
                ax.set_title(f"Overlay t={t:.2f}")
                ax.set_aspect('equal')

        # Set axis limits and styling
        if self.umin is not None and self.umax is not None:
            for j in range(2):  # Only x, y dimensions
                if self.umin[j] is not None:
                    umin[j] = self.umin[j]
                if self.umax[j] is not None:
                    umax[j] = self.umax[j]

        bounds = [(self.xy_min[0], self.xy_max[0]), (self.xy_min[1], self.xy_max[1])]

        for ax in axes:
            ax.set_xlim(*bounds[0])
            ax.set_ylim(*bounds[1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        title = "Impact Plate: Prediction vs Ground Truth" if not self.overlayed_ground_truth else "Impact Plate: Overlay"
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
