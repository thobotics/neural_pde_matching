import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pde_matching.envs.visualizer import Visualizer
from matplotlib import cm
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import art3d
from pathlib import Path

# Data indices configuration
VELOCITY_INDICES = slice(0, 3)     # u_dot[:, 0:3] - velocities
DISPLACEMENT_INDICES = slice(3, 6)  # u_dot[:, 3:6] - face displacements


class AbaqusPlateDeformationVisualizer(Visualizer):
    def __init__(
        self,
        trajectory_length,
        plot_2d=False,
        plot_ground_truth=True,
        plot_force_vector=True,
        plot_stress=True,
        plot_elev=30,
        plot_azim=-60,
        n_columns=3,
        umin=None,
        umax=None,
        overlayed_ground_truth=False,
        local_initial_and_final_path=None,
        **kwargs
    ):
        self.trajectory_length = trajectory_length
        self.plot_2d = plot_2d
        self.plot_ground_truth = plot_ground_truth
        self.plot_force_vector = plot_force_vector
        self.plot_stress = plot_stress
        self.plot_elev = plot_elev
        self.plot_azim = plot_azim
        self.n_columns = n_columns
        self.overlayed_ground_truth = overlayed_ground_truth
        self.local_initial_and_final_path = local_initial_and_final_path
        if umin is not None:
            self.umin = np.array([v if v is not None else None for v in umin])
        else:
            self.umin = None
        if umax is not None:
            self.umax = np.array([v if v is not None else None for v in umax])
        else:
            self.umax = None

    def _save_final_frame_2d(self, u_list, u_hat_list, data_list, properties_list, save_dir):
        """Save final frame as 2D plot with z-displacement as color."""
        from .abaqus_plate_deformation import NodeType
        
        def get_mesh_indices(props):
            node_types = np.argmax(props[..., :NodeType.SIZE], axis=1)
            return (node_types == NodeType.NORMAL) | (node_types == NodeType.BOUNDARY) | (node_types == NodeType.OBSTACLE)
        
        def extract_mesh_data(positions, mesh_idx, mesh_face):
            """Extract mesh data and return positions, faces, and z-values."""
            mesh_positions = positions[mesh_idx]
            faces_data = []
            z_values = []
            
            for face in mesh_face.T:
                face_vertices = mesh_positions[face]
                # Fix mesh ordering - ensure counter-clockwise ordering for proper visualization
                if mesh_face.shape[0] == 4:  # Quadrilateral faces
                    # Reorder vertices: [0,1,3,2] to get proper quad ordering
                    face_vertices = face_vertices[[0, 1, 3, 2]]
                
                # Store x,y coordinates for 2D polygon
                face_2d = face_vertices[:, :2]
                faces_data.append(face_2d)
                
                # Average z-displacement for face color
                face_z = np.mean(face_vertices[:, 2])
                z_values.append(face_z)
            
            return faces_data, np.array(z_values), mesh_positions
        
        def create_2d_mesh_plot(ax, positions, mesh_idx, mesh_face, title):
            faces_data, z_values, mesh_positions = extract_mesh_data(positions, mesh_idx, mesh_face)
            
            # Create polygon collection
            poly_collection = PolyCollection(faces_data, alpha=0.7)
            poly_collection.set_array(z_values)
            poly_collection.set_cmap('jet')
            
            ax.add_collection(poly_collection)
            
            # Set axis limits
            x_coords = mesh_positions[:, 0]
            y_coords = mesh_positions[:, 1]
            if len(x_coords) > 0:
                x_margin = (x_coords.max() - x_coords.min()) * 0.1
                y_margin = (y_coords.max() - y_coords.min()) * 0.1
                ax.set_xlim(x_coords.min() - x_margin, x_coords.max() + x_margin)
                ax.set_ylim(y_coords.min() - y_margin, y_coords.max() + y_margin)
            
            ax.set_aspect('equal')
            ax.set_title(title)
            return poly_collection, z_values
        
        # Save for first sample only
        if len(u_list) > 0:
            data = data_list[0]
            mesh_idx = get_mesh_indices(properties_list[0])
            
            # Extract data for numpy saving
            pred_faces, pred_z_values, pred_mesh_pos = extract_mesh_data(u_hat_list[0], mesh_idx, data.mesh_face.cpu())
            gt_faces, gt_z_values, gt_mesh_pos = extract_mesh_data(u_list[0], mesh_idx, data.mesh_face.cpu())
            
            # Save raw numpy data
            np.save(save_dir / "pred_mesh_positions.npy", pred_mesh_pos)
            np.save(save_dir / "pred_z_values.npy", pred_z_values)
            np.save(save_dir / "gt_mesh_positions.npy", gt_mesh_pos)
            np.save(save_dir / "gt_z_values.npy", gt_z_values)
            np.save(save_dir / "mesh_faces.npy", data.mesh_face.cpu().numpy())
            
            # Create side-by-side plot
            fig, (ax_pred, ax_gt) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Create plots
            pred_collection, _ = create_2d_mesh_plot(ax_pred, u_hat_list[0], mesh_idx, 
                                                   data.mesh_face.cpu(), 'Prediction')
            gt_collection, _ = create_2d_mesh_plot(ax_gt, u_list[0], mesh_idx,
                                                 data.mesh_face.cpu(), 'Ground Truth')
            
            # Use same colorbar scale for both plots
            all_z = np.concatenate([pred_z_values, gt_z_values])
            vmin, vmax = all_z.min(), all_z.max()
            pred_collection.set_clim(vmin, vmax)
            gt_collection.set_clim(vmin, vmax)
            
            # Add colorbars
            plt.colorbar(pred_collection, ax=ax_pred, label='Z displacement')
            plt.colorbar(gt_collection, ax=ax_gt, label='Z displacement')
            
            plt.tight_layout()
            plt.savefig(save_dir / "final_2d.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save separate prediction and ground truth plots
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            pred_collection, _ = create_2d_mesh_plot(ax, u_hat_list[0], mesh_idx, 
                                                   data.mesh_face.cpu(), 'Prediction')
            pred_collection.set_clim(vmin, vmax)
            plt.colorbar(pred_collection, ax=ax, label='Z displacement')
            plt.savefig(save_dir / "pred.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            gt_collection, _ = create_2d_mesh_plot(ax, u_list[0], mesh_idx,
                                                 data.mesh_face.cpu(), 'Ground Truth')
            gt_collection.set_clim(vmin, vmax)
            plt.colorbar(gt_collection, ax=ax, label='Z displacement')
            plt.savefig(save_dir / "true.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved 2D analysis data to {save_dir}")

    def visualize(self, u, u_dot, u_hat, u_dot_hat, t, data_batch, properties, 
                  umin, umax, udot_min, udot_max, n_images=-1, mesh_color="orange", collider_color="black", padding=0.025):
        from .abaqus_plate_deformation import NodeType

        # Prepare data lists
        if hasattr(data_batch, "batch") and data_batch.batch is not None:
            batch_idx = data_batch.batch.cpu().numpy()
            data_list = data_batch.to_data_list()
            u_list = [u[batch_idx == i] for i in range(n_images if n_images != -1 else len(data_batch))]
            u_dot_list = [u_dot[batch_idx == i] for i in range(len(u_list))]
            u_hat_list = [u_hat[batch_idx == i] for i in range(len(u_list))]
            u_dot_hat_list = [u_dot_hat[batch_idx == i] for i in range(len(u_list))]
            properties_list = [properties[batch_idx == i] for i in range(len(u_list))]
        else:
            data_list = [data_batch]
            u_list = [u]
            u_dot_list = [u_dot]
            u_hat_list = [u_hat]
            u_dot_hat_list = [u_dot_hat]
            properties_list = [properties]

        num_images = len(u_list)

        def get_mesh_indices(props):
            node_types = np.argmax(props[..., :NodeType.SIZE], axis=1)
            return (node_types == NodeType.NORMAL) | (node_types == NodeType.BOUNDARY) | (node_types == NodeType.OBSTACLE)

        def get_face_stress_colors(u_data, mesh_idx, mesh_face):
            face_stresses = []
            for face in mesh_face.T:
                face_vertices = u_data[mesh_idx][face, DISPLACEMENT_INDICES]
                stress_magnitude = np.mean(np.sqrt(np.sum(face_vertices**2, axis=1)))
                face_stresses.append(stress_magnitude)
            
            if not self.plot_stress:
                return mesh_color
            
            if np.max(face_stresses) == np.min(face_stresses):
                return cm.get_cmap("jet")(np.zeros(len(face_stresses)))
            
            normalized = (np.array(face_stresses) - np.min(face_stresses)) / (np.max(face_stresses) - np.min(face_stresses))
            return cm.get_cmap("jet")(normalized)

        def create_mesh_collection(positions, mesh_idx, mesh_face, colors, alpha=0.5, linewidths=0.5, edgecolors=(1,1,1,0.25)):
            mesh_coords = positions[mesh_idx, :3][mesh_face.T]
            if mesh_face.shape[0] == 4:
                mesh_coords = mesh_coords[:, [0, 1, 3, 2], :]
            return art3d.Poly3DCollection(mesh_coords, alpha=alpha, facecolors=colors, 
                                        edgecolors=edgecolors, linewidths=linewidths)

        def plot_force_vectors(ax, u_data, props):
            if not self.plot_force_vector:
                return
            force_direction = props[..., NodeType.SIZE:NodeType.SIZE + 3]
            collider_indices = force_direction[..., -1] != 0
            for collider, vector in zip(u_data[collider_indices], force_direction[collider_indices]):
                position = collider[:3]
                ax.quiver(*position, *(vector * 0.005), length=10., color=collider_color, 
                         alpha=0.8, arrow_length_ratio=0.15)
                ax.scatter(*position, color=collider_color, s=20, depthshade=True)

        if not self.overlayed_ground_truth:
            # Side-by-side: prediction | ground truth
            fig = plt.figure(figsize=(8, 4 * num_images))
            axes = [fig.add_subplot(num_images, 2, i * 2 + j + 1, projection="3d") 
                   for i in range(num_images) for j in range(2)]
            
            for i in range(num_images):
                ax_pred, ax_gt = axes[i * 2], axes[i * 2 + 1]
                data = data_list[i]
                data.mesh_face = data.mesh_face.cpu()
                mesh_idx = get_mesh_indices(properties_list[i])
                
                # Prediction plot
                plot_force_vectors(ax_pred, u_list[i], properties_list[i])
                pred_colors = get_face_stress_colors(u_hat_list[i], mesh_idx, data.mesh_face)
                pred_mesh = create_mesh_collection(u_hat_list[i], mesh_idx, data.mesh_face, pred_colors)
                ax_pred.add_collection3d(pred_mesh)
                ax_pred.set_title(f"Prediction t={t:.2f}")
                
                # Ground truth plot
                gt_colors = get_face_stress_colors(u_list[i], mesh_idx, data.mesh_face)
                gt_mesh = create_mesh_collection(u_list[i], mesh_idx, data.mesh_face, gt_colors)
                ax_gt.add_collection3d(gt_mesh)
                ax_gt.set_title(f"Ground Truth t={t:.2f}")
        else:
            # Overlay: wireframe ground truth + colored prediction
            fig = plt.figure(figsize=(6, 4 * num_images))
            axes = [fig.add_subplot(num_images, 1, i + 1, projection="3d") for i in range(num_images)]
            
            for i, ax in enumerate(axes):
                data = data_list[i]
                data.mesh_face = data.mesh_face.cpu()
                mesh_idx = get_mesh_indices(properties_list[i])
                
                plot_force_vectors(ax, u_list[i], properties_list[i])
                
                # Ground truth as wireframe (thick greyish-white)
                if self.plot_ground_truth:
                    gt_wireframe = create_mesh_collection(u_list[i], mesh_idx, data.mesh_face, 
                                                        colors="green", alpha=0, linewidths=2.0, 
                                                        edgecolors=(0.8, 0.8, 0.8, 0.9))
                    ax.add_collection3d(gt_wireframe)
                
                # Prediction with stress colors
                pred_colors = get_face_stress_colors(u_hat_list[i], mesh_idx, data.mesh_face)
                pred_mesh = create_mesh_collection(u_hat_list[i], mesh_idx, data.mesh_face, pred_colors)
                ax.add_collection3d(pred_mesh)
                ax.set_title(f"Prediction + Ground Truth t={t:.2f}")

        if self.umin is not None and self.umax is not None:
            for j in range(3):
                if self.umin[j] is not None:
                    umin[j] = self.umin[j]
                if self.umax[j] is not None:
                    umax[j] = self.umax[j]

        # Manually set axis limits based on umin and umax
        umin[2] = -20.0 if umin[2] == 0.0 else umin[2]
        umax[2] = 20.0 if umax[2] == 0.0 else umax[2]

        bounds = [(umin[i] + umin[i] * padding, umax[i] + umax[i] * padding) for i in range(3)]
        for ax in axes:
            ax.set_xlim(*bounds[0])
            ax.set_ylim(*bounds[1])
            ax.set_zlim(*bounds[2])
            ax.axis("off")
            ax.view_init(elev=self.plot_elev, azim=self.plot_azim)

        title = "Prediction vs Ground Truth" if not self.overlayed_ground_truth else "Prediction with Ground Truth Overlay"
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save 2D and 3D plots if path is specified and plotting 2D
        if self.local_initial_and_final_path and t <= self.trajectory_length - 2:
            save_dir = Path(self.local_initial_and_final_path) / "sample_final"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if self.plot_2d:
                # Save 2D plots with z-displacement as color
                self._save_final_frame_2d(u_list, u_hat_list, data_list, properties_list, save_dir)
            elif not self.overlayed_ground_truth and len(axes) >= 2:
                # Save separate 3D plots by recreating them
                data = data_list[0]
                mesh_idx = get_mesh_indices(properties_list[0])
                
                # Save prediction 3D plot
                pred_fig = plt.figure(figsize=(8, 6))
                pred_ax = pred_fig.add_subplot(111, projection="3d")
                
                # Recreate prediction mesh
                # plot_force_vectors(pred_ax, u_list[0], properties_list[0])
                pred_colors = get_face_stress_colors(u_hat_list[0], mesh_idx, data.mesh_face)
                pred_mesh = create_mesh_collection(u_hat_list[0], mesh_idx, data.mesh_face, pred_colors)
                pred_ax.add_collection3d(pred_mesh)
                
                # Copy axis limits and settings from original
                pred_ax.set_xlim(axes[0].get_xlim())
                pred_ax.set_ylim(axes[0].get_ylim()) 
                pred_ax.set_zlim(axes[0].get_zlim())
                pred_ax.view_init(elev=self.plot_elev, azim=self.plot_azim)
                # pred_ax.set_title("Prediction 3D")
                pred_ax.axis("off")
                plt.tight_layout()
                plt.savefig(save_dir / f"pred_3d_{t}.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                # Save ground truth 3D plot  
                gt_fig = plt.figure(figsize=(8, 6))
                gt_ax = gt_fig.add_subplot(111, projection="3d")
                
                # Recreate ground truth mesh
                plot_force_vectors(gt_ax, u_list[0], properties_list[0])
                gt_colors = get_face_stress_colors(u_list[0], mesh_idx, data.mesh_face)
                gt_mesh = create_mesh_collection(u_list[0], mesh_idx, data.mesh_face, gt_colors)
                gt_ax.add_collection3d(gt_mesh)
                
                # Copy axis limits and settings from original
                gt_ax.set_xlim(axes[1].get_xlim())
                gt_ax.set_ylim(axes[1].get_ylim())
                gt_ax.set_zlim(axes[1].get_zlim()) 
                gt_ax.view_init(elev=self.plot_elev, azim=self.plot_azim)
                # gt_ax.set_title("Ground Truth 3D")
                gt_ax.axis("off")
                plt.tight_layout()
                plt.savefig(save_dir / f"true_3d_{t}.png", dpi=150, bbox_inches='tight')
                plt.close()
        
        return fig
