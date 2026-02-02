import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pde_matching.envs.visualizer import Visualizer
from matplotlib.tri import Triangulation


class CylinderFlowVisualizer(Visualizer):
    def __init__(self, 
        n_columns=3,
        umin=None,
        umax=None,
        overlayed_ground_truth=False,
        support_negative_velocity=False,
        xy_min=(0.0, 0.0),
        xy_max=(30.0, 40.0),
        **kwargs
    ):
        self.n_columns = n_columns
        self.overlayed_ground_truth = overlayed_ground_truth
        self.support_negative_velocity = support_negative_velocity
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

    def visualize(self, u, u_dot, u_hat, u_dot_hat, t, data_batch, properties, 
                  umin, umax, udot_min, udot_max, n_images=-1, padding=0.025):
        from .cylinder_flow import NodeType

        # Prepare data lists
        if hasattr(data_batch, "batch") and data_batch.batch is not None:
            batch_idx = data_batch.batch.cpu().numpy()
            data_list = data_batch.to_data_list()
            u_list = [u[batch_idx == i] for i in range(n_images if n_images != -1 else len(data_list))]
            u_hat_list = [u_hat[batch_idx == i] for i in range(len(u_list))]
            properties_list = [properties[batch_idx == i] for i in range(len(u_list))]
        else:
            data_list = [data_batch]
            u_list = [u]
            u_hat_list = [u_hat]
            properties_list = [properties]

        num_images = len(u_list)

        def get_node_masks(props):
            node_types = np.argmax(props[..., :NodeType.SIZE], axis=1)
            normal_mask = node_types == NodeType.NORMAL
            obstacle_mask = node_types == NodeType.OBSTACLE
            return normal_mask, obstacle_mask

        def plot_cylinder_flow(ax, data, u_state, properties, title_suffix, vel_min=None, vel_max=None):
            """Triangular mesh visualization for cylinder flow."""
            # Extract data - velocity is 2D
            velocity = u_state[:, :2]  # (N, 2) - velocity components
            velocity_magnitude = np.linalg.norm(velocity, axis=-1)  # (N,) - magnitude
            positions = data.pos.detach().cpu().numpy()
            triangles = data.cells.detach().cpu().numpy()
            
            normal_mask, obstacle_mask = get_node_masks(properties)
            
            # Create triangulation for plotting
            x = positions[:, 0]
            y = positions[:, 1]
            triang = Triangulation(x, y, triangles)
            
            # Choose colormap and limits for velocity magnitude
            cmap = 'viridis'
            if vel_min is not None and vel_max is not None:
                vmin = vel_min
                vmax = vel_max
            else:
                vmin = np.nanmin(velocity_magnitude)
                vmax = np.nanmax(velocity_magnitude)
            
            # Plot velocity magnitude field using tripcolor
            tcf = ax.tripcolor(triang, velocity_magnitude, shading='flat', cmap=cmap, vmin=vmin, vmax=vmax)

            # Highlight obstacle nodes (cylinder boundary)
            if np.any(obstacle_mask) and len(positions[obstacle_mask]) > 0:
                obstacle_coords = positions[obstacle_mask]
                ax.scatter(obstacle_coords[:, 0], obstacle_coords[:, 1], 
                          c='red', s=15, alpha=1.0, marker='o', edgecolors='black', linewidth=0.5)
            
            ax.set_title(f'{title_suffix} t={t:.2f}')
            ax.set_aspect('equal')
            
            # Set reasonable axis limits
            if len(positions) > 0:
                x_margin = (positions[:, 0].max() - positions[:, 0].min()) * 0.05
                y_margin = (positions[:, 1].max() - positions[:, 1].min()) * 0.05
                ax.set_xlim(positions[:, 0].min() - x_margin, positions[:, 0].max() + x_margin)
                ax.set_ylim(positions[:, 1].min() - y_margin, positions[:, 1].max() + y_margin)
            
            return tcf

        # Use provided umin/umax for consistent coloring across timesteps
        # For velocity, use magnitude bounds
        vel_min = 0.0
        vel_max = np.linalg.norm([umax[0], umax[1]]) if len(umax) >= 2 else 1.0

        if not self.overlayed_ground_truth:
            # Side-by-side: prediction | ground truth
            fig = plt.figure(figsize=(10, 2 * num_images))
            axes = [fig.add_subplot(num_images, 2, i * 2 + j + 1) 
                   for i in range(num_images) for j in range(2)]
            
            for i in range(num_images):
                ax_pred, ax_gt = axes[i * 2], axes[i * 2 + 1]
                data = data_list[i]
                
                tcf_pred = plot_cylinder_flow(ax_pred, data, u_hat_list[i], properties_list[i], 
                                            "Prediction", vel_min, vel_max)
                tcf_gt = plot_cylinder_flow(ax_gt, data, u_list[i], properties_list[i], 
                                          "Ground Truth", vel_min, vel_max)
                
                # Add colorbar to the right subplot
                # if i == 0:
                #     fig.colorbar(tcf_gt, ax=ax_gt, shrink=0.8, label='Velocity Magnitude')
        else:
            # Overlay: prediction with ground truth  
            fig = plt.figure(figsize=(10, 10 * num_images))
            axes = [fig.add_subplot(num_images, 1, i + 1) for i in range(num_images)]
            
            for i, ax in enumerate(axes):
                data = data_list[i]
                positions = data.pos.detach().cpu().numpy()
                triangles = data.cells.detach().cpu().numpy()
                
                velocity_pred = u_hat_list[i]
                velocity_gt = u_list[i]
                
                vel_mag_pred = np.linalg.norm(velocity_pred, axis=-1)
                vel_mag_gt = np.linalg.norm(velocity_gt, axis=-1)
                
                normal_mask, obstacle_mask = get_node_masks(properties_list[i])
                
                # Create triangulation
                x = positions[:, 0]
                y = positions[:, 1]
                triang = Triangulation(x, y, triangles)
                
                # Plot prediction as filled contours
                tcf = ax.tripcolor(triang, vel_mag_pred, shading='flat', cmap='viridis', 
                                 alpha=0.7, vmin=vel_min, vmax=vel_max, label='Prediction')
                
                # Plot ground truth as contour lines
                ax.tricontour(triang, vel_mag_gt, levels=10, colors='red', alpha=0.8, linewidths=1)
                
                # Add velocity vectors for prediction
                if len(positions) > 500:
                    step = len(positions) // 100
                    subsample_idx = np.arange(0, len(positions), step)
                else:
                    subsample_idx = np.arange(len(positions))
                
                vector_mask = normal_mask[subsample_idx]
                if np.any(vector_mask):
                    sub_pos = positions[subsample_idx][vector_mask]
                    sub_vel_pred = velocity_pred[subsample_idx][vector_mask]
                    scale = (vel_max - vel_min) * 0.1 if vel_max > vel_min else 1.0
                    ax.quiver(sub_pos[:, 0], sub_pos[:, 1], 
                             sub_vel_pred[:, 0], sub_vel_pred[:, 1], 
                             scale=scale, scale_units='xy', angles='xy', 
                             alpha=0.6, color='blue', width=0.002)
                
                # Highlight obstacle nodes
                if np.any(obstacle_mask) and len(positions[obstacle_mask]) > 0:
                    obstacle_coords = positions[obstacle_mask]
                    ax.scatter(obstacle_coords[:, 0], obstacle_coords[:, 1], 
                             c='black', s=15, alpha=1.0, marker='s')
                
                ax.set_title(f'Overlay t={t:.2f}')
                ax.set_aspect('equal')
                
                if i == 0:
                    # Add custom legend
                    from matplotlib.patches import Patch
                    legend_elements = [Patch(facecolor='viridis', alpha=0.7, label='Prediction'),
                                     Patch(facecolor='red', alpha=0.8, label='Ground Truth (contours)')]
                    ax.legend(handles=legend_elements)
                
                # Set reasonable axis limits
                if len(positions) > 0:
                    x_margin = (positions[:, 0].max() - positions[:, 0].min()) * 0.05
                    y_margin = (positions[:, 1].max() - positions[:, 1].min()) * 0.05
                    ax.set_xlim(positions[:, 0].min() - x_margin, positions[:, 0].max() + x_margin)
                    ax.set_ylim(positions[:, 1].min() - y_margin, positions[:, 1].max() + y_margin)

        # bounds = [(self.xy_min[0], self.xy_max[0]), (self.xy_min[1], self.xy_max[1])]

        # Set labels
        for ax in axes:
            # ax.set_xlim(*bounds[0])
            # ax.set_ylim(*bounds[1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        title = "Cylinder Flow: Prediction vs Ground Truth" if not self.overlayed_ground_truth else "Cylinder Flow: Prediction with Ground Truth Overlay"
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
