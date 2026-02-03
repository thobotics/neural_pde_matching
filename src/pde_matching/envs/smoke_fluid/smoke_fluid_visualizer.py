import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pde_matching.envs.visualizer import Visualizer
from matplotlib import cm


class SmokeFluidVisualizer(Visualizer):
    def __init__(
        self,
        grid_shape,
        trajectory_length,
        plot_quiver=True,
        n_columns=3,
        umin=None,
        umax=None,
        overlayed_ground_truth=False,
        support_negative_density=False,
        use_log_colormap=False,
        log_scale=1.0,
        plot_as_images=False,
        local_initial_and_final_path=None,
        cmap='jet',
        transpose_grid=False,
        **kwargs
    ):
        self.grid_shape = grid_shape
        self.trajectory_length = trajectory_length
        self.n_columns = n_columns
        self.overlayed_ground_truth = overlayed_ground_truth
        self.support_negative_density = support_negative_density
        self.use_log_colormap = use_log_colormap
        self.log_scale = log_scale
        self.plot_as_images = plot_as_images
        self.local_initial_and_final_path = local_initial_and_final_path
        self.cmap = cmap
        self.transpose_grid = transpose_grid
        if umin is not None:
            self.umin = np.array([v if v is not None else None for v in umin])
        else:
            self.umin = None
        if umax is not None:
            self.umax = np.array([v if v is not None else None for v in umax])
        else:
            self.umax = None

    def _plot_scatter(self, ax, coords, density, fluid_mask, boundary_mask, cmap, vmin, vmax):
        """Scatter plot fallback."""
        if np.any(fluid_mask):
            fluid_coords = coords[fluid_mask]
            fluid_density = density[fluid_mask]
            
            if np.any(~np.isnan(fluid_density)) and np.any(abs(fluid_density) > 0):
                if self.use_log_colormap:
                    # Normalize to [0,1] range first, then apply log transform
                    if vmin is not None and vmax is not None:
                        fluid_density = (fluid_density - vmin) / (vmax - vmin)
                        fluid_density = np.clip(fluid_density, 0, 1)
                        fluid_density = (fluid_density + 1e-8) ** self.log_scale
                        vmin, vmax = 0, 1
                
                ax.scatter(fluid_coords[:, 0], fluid_coords[:, 1], 
                          c=fluid_density, cmap=cmap, s=50, alpha=1.0, vmin=vmin, vmax=vmax)
            else:
                ax.scatter(fluid_coords[:, 0], fluid_coords[:, 1], c='red', s=50, alpha=1.0)
        
        if np.any(boundary_mask):
            boundary_coords = coords[boundary_mask]
            ax.scatter(boundary_coords[:, 0], boundary_coords[:, 1], 
                      c='black', s=30, alpha=1.0, marker='s')

    def _save_initial_final_frames(self, u_list, u_hat_list, data_list, properties_list, umin, umax, save_dir):
        """Save initial and final frames as separate images."""
        from .smoke_fluid import NodeType
        
        def get_node_masks(props):
            node_types = np.argmax(props[..., :NodeType.SIZE], axis=1)
            fluid_mask = node_types == NodeType.FLUID
            boundary_mask = node_types == NodeType.BOUNDARY
            return fluid_mask, boundary_mask
        
        def create_grid_image(u_state, data, properties):
            density = u_state[:, 0]
            coords = data.pos.cpu().numpy()
            fluid_mask, boundary_mask = get_node_masks(properties)
            
            grid_image = np.full(self.grid_shape, np.nan)
            for i, (coord, density_val) in enumerate(zip(coords, density)):
                grid_x = int(np.clip(round(coord[0]), 0, self.grid_shape[1] - 1))
                grid_y = int(np.clip(round(coord[1]), 0, self.grid_shape[0] - 1))

                if not boundary_mask[i]:  # Skip boundary nodes
                    grid_image[grid_x, grid_y] = density_val
            return grid_image
        
        # Save data for first sample only
        if len(u_list) > 0 and self.plot_as_images:
            # Create grid images
            pred_img = create_grid_image(u_hat_list[0], data_list[0], properties_list[0])
            true_img = create_grid_image(u_list[0], data_list[0], properties_list[0])

            # Save raw data
            np.save(save_dir / "pred_full.npy", pred_img)
            np.save(save_dir / "true_full.npy", true_img)

            # Use same color scale for both
            vmin = np.nanmin(true_img)
            vmax = np.nanmax(true_img)

            # Save initial frames
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            im = ax.imshow(pred_img, cmap=self.cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax.set_title('Prediction')
            plt.colorbar(im, ax=ax)
            plt.savefig(save_dir / "pred.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            im = ax.imshow(true_img, cmap=self.cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax.set_title('Ground Truth')
            plt.colorbar(im, ax=ax)
            plt.savefig(save_dir / "true.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved analysis data to {save_dir}")

    def visualize(self, u, u_dot, u_hat, u_dot_hat, t, data_batch, properties, 
                  umin, umax, udot_min, udot_max, n_images=-1, padding=0.025):
        from .smoke_fluid import NodeType

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
            fluid_mask = node_types == NodeType.FLUID
            boundary_mask = node_types == NodeType.BOUNDARY
            return fluid_mask, boundary_mask

        def plot_smoke_simple(ax, data, u_state, properties, title_suffix, density_min=None, density_max=None):
            density = u_state[:, 0]
            coords = data.pos.cpu().numpy()
            fluid_mask, boundary_mask = get_node_masks(properties)
            
            cmap = self.cmap
            vmin, vmax = density_min, density_max

            if self.plot_as_images:
                # Grid/image plotting
                plot_density = density.copy()
                
                if self.use_log_colormap:
                    # Normalize to [0,1] range first, then apply log transform
                    if vmin is not None and vmax is not None:
                        plot_density = (plot_density - vmin) / (vmax - vmin)
                        plot_density = np.clip(plot_density, 0, 1)
                        plot_density = (plot_density + 1e-8) ** self.log_scale
                        vmin, vmax = 0, 1
                
                grid_shape = self.grid_shape
                
                # Create full grid image
                grid_image = np.full(grid_shape, np.nan)
                
                # Map each node to its grid position using coordinates
                for i, (coord, density_val) in enumerate(zip(coords, plot_density)):
                    grid_x = int(np.clip(round(coord[0]), 0, grid_shape[1] - 1))
                    grid_y = int(np.clip(round(coord[1]), 0, grid_shape[0] - 1))

                    if not boundary_mask[i]:  # Skip boundary nodes
                        if not self.transpose_grid:
                            grid_image[grid_x, grid_y] = density_val
                        else:
                            grid_image[grid_y, grid_x] = density_val

                # Plot as image
                extent = [0, grid_shape[1], 0, grid_shape[0]]
                ax.imshow(grid_image, cmap=cmap, vmin=vmin, vmax=vmax, 
                         extent=extent, origin='lower')
                
                # if np.any(boundary_mask):
                #     boundary_coords = coords[boundary_mask] + 1
                #     ax.scatter(boundary_coords[:, 0], boundary_coords[:, 1], 
                #               c='black', s=15, alpha=1.0, marker='s')
            else:
                # Scatter plot
                self._plot_scatter(ax, coords, density, fluid_mask, boundary_mask, cmap, vmin, vmax)
            
            ax.set_title(f'{title_suffix} t={t:.2f}')
            ax.set_aspect('equal')
            
            if len(coords) > 0:
                x_margin = (coords[:, 0].max() - coords[:, 0].min()) * 0.1
                y_margin = (coords[:, 1].max() - coords[:, 1].min()) * 0.1
                ax.set_xlim(coords[:, 0].min() - x_margin, coords[:, 0].max() + x_margin)
                ax.set_ylim(coords[:, 1].min() - y_margin, coords[:, 1].max() + y_margin)

        if not self.overlayed_ground_truth:
            # Side-by-side: prediction | ground truth
            fig = plt.figure(figsize=(12, 6 * num_images))
            axes = [fig.add_subplot(num_images, 2, i * 2 + j + 1) 
                   for i in range(num_images) for j in range(2)]
            
            for i in range(num_images):
                ax_pred, ax_gt = axes[i * 2], axes[i * 2 + 1]
                data = data_list[i]
                
                plot_smoke_simple(ax_pred, data, u_hat_list[i], properties_list[i], 
                                "Prediction", umin[0], umax[0])
                plot_smoke_simple(ax_gt, data, u_list[i], properties_list[i], 
                                "Ground Truth", umin[0], umax[0])
        else:
            raise NotImplementedError("Overlayed ground truth mode is not implemented yet.")

        # Set labels
        for ax in axes:
            ax.axis('off')  # This removes all axis elements (labels, ticks, etc.)

        title = "Prediction vs Ground Truth" if not self.overlayed_ground_truth else "Prediction with Ground Truth Overlay"
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save initial and final frames if path is specified
        if self.local_initial_and_final_path and (t == 0 or t == self.trajectory_length - 1):
            idx = "initial" if t == 0 else "final"
            save_dir = Path(self.local_initial_and_final_path) / f"sample_{idx}"
            save_dir.mkdir(parents=True, exist_ok=True)
            self._save_initial_final_frames(u_list, u_hat_list, data_list, properties_list, umin, umax, save_dir)
        
        return fig
