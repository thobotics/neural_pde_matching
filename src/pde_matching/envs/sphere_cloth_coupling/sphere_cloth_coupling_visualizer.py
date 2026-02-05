import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pde_matching.envs.visualizer import Visualizer
from matplotlib import cm
from mpl_toolkits.mplot3d import art3d
import meshio


class SphereClothCouplingVisualizer(Visualizer):
    def __init__(
        self,
        trajectory_length,
        plot_elev=30,
        plot_azim=-60,
        n_columns=3,
        umin=None,
        umax=None,
        overlayed_ground_truth=False,
        local_initial_and_final_path=None,
        **kwargs,
    ):
        self.trajectory_length = trajectory_length
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

    def visualize(self, u, u_dot, u_hat, u_dot_hat, t, data_batch, properties, 
                  umin, umax, udot_min, udot_max, n_images=-1, mesh_color="orange", sphere_color="blue", padding=0.025):
        from .sphere_cloth_coupling import NodeType

        # Prepare data lists
        if hasattr(data_batch, "batch") and data_batch.batch is not None:
            batch_idx = data_batch.batch.cpu().numpy()
            data_list = data_batch.to_data_list()
            u_list = [u[batch_idx == i] for i in range(n_images if n_images != -1 else len(data_list))]
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

        def get_node_masks(props):
            node_types = np.argmax(props[..., :NodeType.SIZE], axis=1)
            cloth_mask = node_types == NodeType.CLOTH
            cloth_corner_mask = node_types == NodeType.CLOTH_CORNER
            sphere_center_mask = node_types == NodeType.SPHERE_CENTER
            # For cloth mesh visualization, we need both cloth and corners for proper indexing
            all_cloth_mask = cloth_mask | cloth_corner_mask
            return all_cloth_mask, cloth_mask, cloth_corner_mask, sphere_center_mask

        def create_mesh_collection(positions, faces, colors, alpha=0.7, linewidths=1.0, edgecolors="#5a5a5a"):
            if len(faces) == 0:
                return None
            mesh_coords = positions[faces]
            return art3d.Poly3DCollection(mesh_coords, alpha=alpha, facecolors=colors, 
                                        edgecolors=edgecolors, linewidths=linewidths)

        def save_separate_vtk_files(save_dir, prefix, cloth_positions, cloth_faces, sphere_positions, sphere_faces, corner_positions):
            """Save cloth, sphere, and corners as separate VTK files."""
            saved_files = []
            
            # Save cloth mesh
            if len(cloth_positions) > 0 and len(cloth_faces) > 0:
                cloth_cells = [("triangle", cloth_faces)]
                cloth_mesh = meshio.Mesh(points=cloth_positions, cells=cloth_cells)
                cloth_path = save_dir / f"{prefix}_cloth.vtk"
                meshio.write(cloth_path, cloth_mesh)
                saved_files.append(cloth_path)
            
            # Save sphere mesh
            if len(sphere_positions) > 0 and len(sphere_faces) > 0:
                sphere_cells = [("triangle", sphere_faces)]
                sphere_mesh = meshio.Mesh(points=sphere_positions, cells=sphere_cells)
                sphere_path = save_dir / f"{prefix}_sphere.vtk"
                meshio.write(sphere_path, sphere_mesh)
                saved_files.append(sphere_path)
            
            # Save corner points as vertices
            if len(corner_positions) > 0:
                corner_cells = [("vertex", np.arange(len(corner_positions)).reshape(-1, 1))]
                corner_mesh = meshio.Mesh(points=corner_positions, cells=corner_cells)
                corner_path = save_dir / f"{prefix}_corners.vtk"
                meshio.write(corner_path, corner_mesh)
                saved_files.append(corner_path)
            
            return saved_files

        def get_cloth_colors(faces, positions, color="orange", use_colormap=True, z_min=9.0, z_max=10.5):
            if len(faces) == 0:
                return []
            
            # Calculate height (z-coordinate) for each face
            face_heights = []
            for face in faces:
                face_vertices = positions[face]
                avg_height = np.mean(face_vertices[:, 2])  # z-coordinate is index 2
                face_heights.append(avg_height)
            
            face_heights = np.array(face_heights)
            if use_colormap is False:
                return [color] * len(faces)
            
            # Normalize heights to [0, 1] for colormap
            normalized = (face_heights - z_min) / (z_max - z_min)
            return cm.get_cmap("jet")(normalized)
        
        def get_sphere_colors(positions, faces, colormap="plasma"):
            if len(faces) == 0:
                return []
            
            face_stresses = []
            for face in faces:
                face_vertices = positions[face]
                # Simple stress metric: variance of face vertices
                stress = np.var(face_vertices, axis=0).sum()
                face_stresses.append(stress)
            
            if np.max(face_stresses) == np.min(face_stresses):
                return cm.get_cmap(colormap)(np.zeros(len(face_stresses)))
            
            normalized = (np.array(face_stresses) - np.min(face_stresses)) / (np.max(face_stresses) - np.min(face_stresses))
            return cm.get_cmap(colormap)(normalized)

        if not self.overlayed_ground_truth:
            # Side-by-side: prediction | ground truth
            fig = plt.figure(figsize=(12, 6 * num_images))
            axes = [fig.add_subplot(num_images, 2, i * 2 + j + 1, projection="3d") 
                   for i in range(num_images) for j in range(2)]
            
            for i in range(num_images):
                ax_pred, ax_gt = axes[i * 2], axes[i * 2 + 1]
                data = data_list[i]
                
                all_cloth_mask, cloth_mask, cloth_corner_mask, sphere_center_mask = get_node_masks(properties_list[i])
                
                # Get faces for cloth
                cloth_faces = data.cloth_faces.cpu().numpy() if hasattr(data, 'cloth_faces') else np.array([])
                sphere_faces = data.sphere_faces.cpu().numpy() if hasattr(data, 'sphere_faces') else np.array([])
                
                # Prediction plot
                if len(cloth_faces) > 0 and np.any(all_cloth_mask):
                    cloth_sphere_positions = u_hat_list[i]
                    cloth_sphere_positions[cloth_corner_mask] = u_list[i][cloth_corner_mask]  # Set corners to ground truth for stability
                    cloth_positions = cloth_sphere_positions[all_cloth_mask]
                    cloth_colors = get_cloth_colors(cloth_faces, cloth_positions, "orange", use_colormap=False)
                    cloth_mesh_pred = create_mesh_collection(cloth_positions, cloth_faces, cloth_colors, alpha=0.4)
                    if cloth_mesh_pred:
                        ax_pred.add_collection3d(cloth_mesh_pred)
                
                # Sphere meshes (visualization only, use stored sphere mesh positions)
                if len(sphere_faces) > 0 and hasattr(data, 'sphere_mesh_pos'):
                    sphere_mesh_pos = data.sphere_mesh_pos.clone().cpu().numpy()
                    num_spheres = getattr(data, 'num_spheres', 3)
                    p_sphere_per_sphere = getattr(data, 'p_sphere_per_sphere', len(sphere_mesh_pos) // num_spheres)
                    
                    center_positions = u_hat_list[i][sphere_center_mask]
                    sphere_mesh_pos -= sphere_mesh_pos.mean(axis=0, keepdims=True) 
                    sphere_mesh_pos += center_positions 

                    for sphere_idx in range(num_spheres):
                        start_idx = sphere_idx * p_sphere_per_sphere
                        end_idx = (sphere_idx + 1) * p_sphere_per_sphere
                        sphere_mesh = sphere_mesh_pos[start_idx:end_idx]
                        sphere_colors = "#33f3f6"  # Gray color for GT
                        sphere_mesh_pred = create_mesh_collection(sphere_mesh, sphere_faces, sphere_colors, alpha=1.0, linewidths=0.5, edgecolors="#4A4949")
                        if sphere_mesh_pred:
                            ax_pred.add_collection3d(sphere_mesh_pred)
                
                # Sphere centers (multiple points from graph)
                # if np.any(sphere_center_mask):
                #     center_positions = u_hat_list[i][sphere_center_mask]
                #     colors = ['red', 'blue', 'green'][:len(center_positions)]
                #     for idx, (center_pos, color) in enumerate(zip(center_positions, colors)):
                #         ax_pred.scatter(*center_pos.reshape(1, -1).T, color=color, s=150, alpha=0.8, 
                #                       label=f'Sphere {idx+1} Center')
                
                # Cloth corner nodes (highlight them)
                if np.any(cloth_corner_mask):
                    corner_pos = u_list[i][cloth_corner_mask]
                    ax_pred.scatter(*corner_pos.T, color='green', s=80, alpha=0.8, marker='s', label='Cloth Corners')
                
                ax_pred.set_title(f"Prediction t={t:.2f}")
                
                # Ground truth plot
                if len(cloth_faces) > 0 and np.any(all_cloth_mask):
                    cloth_positions = u_list[i][all_cloth_mask]
                    cloth_colors = get_cloth_colors(cloth_faces, cloth_positions, "orange", use_colormap=False)
                    cloth_mesh_gt = create_mesh_collection(cloth_positions, cloth_faces, cloth_colors, alpha=0.4)
                    if cloth_mesh_gt:
                        ax_gt.add_collection3d(cloth_mesh_gt)
                
                # Sphere meshes (visualization only, use stored sphere mesh positions)
                if len(sphere_faces) > 0 and hasattr(data, 'sphere_mesh_pos'):
                    sphere_mesh_pos = data.sphere_mesh_pos.clone().cpu().numpy()
                    num_spheres = getattr(data, 'num_spheres', 3)
                    p_sphere_per_sphere = getattr(data, 'p_sphere_per_sphere', len(sphere_mesh_pos) // num_spheres)
                    
                    for sphere_idx in range(num_spheres):
                        start_idx = sphere_idx * p_sphere_per_sphere
                        end_idx = (sphere_idx + 1) * p_sphere_per_sphere
                        sphere_mesh = sphere_mesh_pos[start_idx:end_idx]
                        # sphere_colors = get_sphere_colors(sphere_mesh, sphere_faces, "plasma")
                        sphere_colors = "#fb0000"  # Gray color for GT
                        sphere_mesh_gt = create_mesh_collection(sphere_mesh, sphere_faces, sphere_colors, alpha=1.0, linewidths=0.5, edgecolors="#4A4949")
                        if sphere_mesh_gt:
                            ax_gt.add_collection3d(sphere_mesh_gt)
                
                # Sphere centers (multiple points from graph)
                # if np.any(sphere_center_mask):
                #     center_positions = u_list[i][sphere_center_mask]
                #     colors = ['red', 'blue', 'green'][:len(center_positions)]
                #     for idx, (center_pos, color) in enumerate(zip(center_positions, colors)):
                #         ax_gt.scatter(*center_pos.reshape(1, -1).T, color=color, s=150, alpha=0.8, 
                #                     label=f'Sphere {idx+1} Center')
                
                # Cloth corner nodes (highlight them)
                if np.any(cloth_corner_mask):
                    corner_pos = u_list[i][cloth_corner_mask]
                    ax_gt.scatter(*corner_pos.T, color='green', s=80, alpha=0.8, marker='s', label='Cloth Corners')
                
                ax_gt.set_title(f"Ground Truth t={t:.2f}")
        else:
            # Overlay: wireframe ground truth + colored prediction
            fig = plt.figure(figsize=(8, 6 * num_images))
            axes = [fig.add_subplot(num_images, 1, i + 1, projection="3d") for i in range(num_images)]
            
            for i, ax in enumerate(axes):
                data = data_list[i]
                all_cloth_mask, cloth_mask, cloth_corner_mask, sphere_center_mask = get_node_masks(properties_list[i])
                
                cloth_faces = data.cloth_faces.cpu().numpy() if hasattr(data, 'cloth_faces') else np.array([])
                sphere_faces = data.sphere_faces.cpu().numpy() if hasattr(data, 'sphere_faces') else np.array([])
                
                # Prediction (colored)
                if len(cloth_faces) > 0 and np.any(all_cloth_mask):
                    cloth_positions = u_hat_list[i][all_cloth_mask]
                    cloth_colors = get_cloth_colors(cloth_faces, cloth_positions, "orange")
                    cloth_mesh_pred = create_mesh_collection(cloth_positions, cloth_faces, cloth_colors, alpha=0.7)
                    if cloth_mesh_pred:
                        ax.add_collection3d(cloth_mesh_pred)
                
                # Sphere meshes prediction (visualization only)
                if len(sphere_faces) > 0 and hasattr(data, 'sphere_mesh_pos'):
                    sphere_mesh_pos = data.sphere_mesh_pos.cpu().numpy()
                    num_spheres = getattr(data, 'num_spheres', 3)
                    p_sphere_per_sphere = getattr(data, 'p_sphere_per_sphere', len(sphere_mesh_pos) // num_spheres)
                    
                    for sphere_idx in range(num_spheres):
                        start_idx = sphere_idx * p_sphere_per_sphere
                        end_idx = (sphere_idx + 1) * p_sphere_per_sphere
                        sphere_mesh = sphere_mesh_pos[start_idx:end_idx]
                        sphere_colors = get_sphere_colors(sphere_mesh, sphere_faces, "plasma")
                        sphere_mesh_pred = create_mesh_collection(sphere_mesh, sphere_faces, sphere_colors, alpha=0.7)
                        if sphere_mesh_pred:
                            ax.add_collection3d(sphere_mesh_pred)
                
                # Ground truth (wireframe)
                if len(cloth_faces) > 0 and np.any(all_cloth_mask):
                    cloth_positions = u_list[i][all_cloth_mask]
                    cloth_mesh_gt = create_mesh_collection(cloth_positions, cloth_faces, 'none', 
                                                         alpha=1.0, linewidths=1.0, edgecolors='black')
                    if cloth_mesh_gt:
                        ax.add_collection3d(cloth_mesh_gt)
                
                # Sphere meshes ground truth (wireframe)
                if len(sphere_faces) > 0 and hasattr(data, 'sphere_mesh_pos'):
                    sphere_mesh_pos = data.sphere_mesh_pos.cpu().numpy()
                    num_spheres = getattr(data, 'num_spheres', 3)
                    p_sphere_per_sphere = getattr(data, 'p_sphere_per_sphere', len(sphere_mesh_pos) // num_spheres)
                    
                    for sphere_idx in range(num_spheres):
                        start_idx = sphere_idx * p_sphere_per_sphere
                        end_idx = (sphere_idx + 1) * p_sphere_per_sphere
                        sphere_mesh = sphere_mesh_pos[start_idx:end_idx]
                        sphere_mesh_gt = create_mesh_collection(sphere_mesh, sphere_faces, 'none',
                                                              alpha=1.0, linewidths=1.0, edgecolors='gray')
                        if sphere_mesh_gt:
                            ax.add_collection3d(sphere_mesh_gt)
                
                # Sphere centers
                if np.any(sphere_center_mask):
                    center_pos_pred = u_hat_list[i][sphere_center_mask]
                    center_pos_gt = u_list[i][sphere_center_mask]
                    colors_pred = ['red', 'blue', 'green'][:len(center_pos_pred)]
                    colors_gt = ['darkred', 'darkblue', 'darkgreen'][:len(center_pos_gt)]
                    
                    for idx, (pred_pos, gt_pos, color_pred, color_gt) in enumerate(zip(center_pos_pred, center_pos_gt, colors_pred, colors_gt)):
                        ax.scatter(*pred_pos.reshape(1, -1).T, color=color_pred, s=150, alpha=0.8, 
                                 label=f'Pred Sphere {idx+1}')
                        ax.scatter(*gt_pos.reshape(1, -1).T, color=color_gt, s=100, alpha=0.6, marker='x', 
                                 label=f'GT Sphere {idx+1}')
                
                # Cloth corners  
                if np.any(cloth_corner_mask):
                    corner_pos_pred = u_hat_list[i][cloth_corner_mask]
                    corner_pos_gt = u_list[i][cloth_corner_mask] 
                    ax.scatter(*corner_pos_pred.T, color='green', s=80, alpha=0.8, marker='s', label='Pred Corners')
                    ax.scatter(*corner_pos_gt.T, color='darkgreen', s=60, alpha=0.6, marker='+', label='GT Corners')
                
                ax.set_title(f"Overlay t={t:.2f}")
                if i == 0:  # Add legend only to first subplot
                    ax.legend()

        # Set axis limits
        if self.umin is not None and self.umax is not None:
            for j in range(3):
                if self.umin[j] is not None:
                    umin[j] = self.umin[j]
                if self.umax[j] is not None:
                    umax[j] = self.umax[j]

        # Adjust bounds
        # umin[2] = min(umin[2], -1.0) if umin[2] != 0.0 else -5.0
        # umax[2] = max(umax[2], 15.0) if umax[2] != 0.0 else 15.0
        umin[2] = 7.0 if self.trajectory_length == 100 else 9.0
        umax[2] = 10.5

        bounds = [(umin[i] - abs(umin[i]) * padding, umax[i] + abs(umax[i]) * padding) for i in range(3)]
        
        for ax in axes:
            ax.set_xlim(*bounds[0])
            ax.set_ylim(*bounds[1])
            ax.set_zlim(*bounds[2])
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            ax.set_axis_off()
            ax.view_init(elev=self.plot_elev, azim=self.plot_azim)

        title = "Sphere-Cloth Coupling: Prediction vs Ground Truth" if not self.overlayed_ground_truth else "Sphere-Cloth Coupling: Prediction with Ground Truth Overlay"
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # if self.local_initial_and_final_path and t == self.trajectory_length - 2:
        if self.local_initial_and_final_path:
            save_dir = Path(self.local_initial_and_final_path) / "sample_final"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save 3D plots if not in overlay mode
            if not self.overlayed_ground_truth and len(axes) >= 2:
                data = data_list[0]
                all_cloth_mask, cloth_mask, cloth_corner_mask, sphere_center_mask = get_node_masks(properties_list[0])
                
                # Save prediction 3D plot
                pred_fig = plt.figure(figsize=(6, 6))
                pred_ax = pred_fig.add_subplot(111, projection="3d")
                
                # Recreate prediction visualization using existing logic
                cloth_faces = data.cloth_faces.cpu().numpy() if hasattr(data, 'cloth_faces') else np.array([])
                sphere_faces = data.sphere_faces.cpu().numpy() if hasattr(data, 'sphere_faces') else np.array([])
                
                if len(cloth_faces) > 0 and np.any(all_cloth_mask):
                    cloth_sphere_positions = u_hat_list[0]
                    cloth_sphere_positions[cloth_corner_mask] = u_list[0][cloth_corner_mask]  # Set corners to ground truth for stability
                    cloth_positions = cloth_sphere_positions[all_cloth_mask]
                    cloth_colors = get_cloth_colors(cloth_faces, cloth_positions, "orange")
                    cloth_mesh_pred = create_mesh_collection(cloth_positions, cloth_faces, cloth_colors, alpha=0.4)
                    if cloth_mesh_pred:
                        pred_ax.add_collection3d(cloth_mesh_pred)
                
                if len(sphere_faces) > 0 and hasattr(data, 'sphere_mesh_pos'):
                    sphere_mesh_pos = data.sphere_mesh_pos.clone().cpu().numpy()
                    num_spheres = getattr(data, 'num_spheres', 3)
                    p_sphere_per_sphere = getattr(data, 'p_sphere_per_sphere', len(sphere_mesh_pos) // num_spheres)
                    
                    center_positions = u_hat_list[0][sphere_center_mask]
                    sphere_mesh_pos -= sphere_mesh_pos.mean(axis=0, keepdims=True) 
                    sphere_mesh_pos += center_positions 

                    for sphere_idx in range(num_spheres):
                        start_idx = sphere_idx * p_sphere_per_sphere
                        end_idx = (sphere_idx + 1) * p_sphere_per_sphere
                        sphere_mesh = sphere_mesh_pos[start_idx:end_idx]
                        sphere_colors = "#33f3f6"  # Gray color for GT
                        sphere_mesh_pred = create_mesh_collection(sphere_mesh, sphere_faces, sphere_colors, alpha=1.0, linewidths=0.5, edgecolors="#4A4949")
                        if sphere_mesh_pred:
                            pred_ax.add_collection3d(sphere_mesh_pred)
                
                # if np.any(sphere_center_mask):
                #     center_positions = u_hat_list[0][sphere_center_mask]
                #     colors = ['red', 'blue', 'green'][:len(center_positions)]
                #     for idx, (center_pos, color) in enumerate(zip(center_positions, colors)):
                #         pred_ax.scatter(*center_pos.reshape(1, -1).T, color=color, s=150, alpha=1.0)
                
                # Cloth corner nodes (highlight them)
                if np.any(cloth_corner_mask):
                    corner_pos = u_list[0][cloth_corner_mask]  # Use ground truth for corners
                    pred_ax.scatter(*corner_pos.T, color='green', s=80, alpha=0.8, marker='s')
                

                if t == self.trajectory_length - 2:
                    # Copy axis limits and settings from original
                    pred_ax.set_xlim(axes[0].get_xlim())
                    pred_ax.set_ylim(axes[0].get_ylim())
                    pred_ax.set_zlim([9, 10.5])  # Fixed z-limits for better visibility
                    # pred_ax.set_zlim(axes[0].get_zlim())
                    pred_ax.view_init(elev=self.plot_elev, azim=self.plot_azim)
                    pred_ax.axis("off")
                    plt.tight_layout()
                    plt.savefig(save_dir / "pred_3d.png", dpi=150, bbox_inches='tight')
                    plt.close()
                
                # Export prediction to VTK
                cloth_positions_pred = cloth_sphere_positions[all_cloth_mask] if np.any(all_cloth_mask) else []
                corner_positions_pred = u_hat_list[0][cloth_corner_mask] if np.any(cloth_corner_mask) else []
                sphere_mesh_pred = sphere_mesh_pos if len(sphere_faces) > 0 and hasattr(data, 'sphere_mesh_pos') else []
                
                saved_files = save_separate_vtk_files(
                    save_dir, f"pred_{t}",
                    cloth_positions_pred, cloth_faces,
                    sphere_mesh_pred, sphere_faces,
                    corner_positions_pred
                )
                
                # Save ground truth 3D plot
                gt_fig = plt.figure(figsize=(6, 6))
                gt_ax = gt_fig.add_subplot(111, projection="3d")
                
                # Recreate ground truth visualization
                if len(cloth_faces) > 0 and np.any(all_cloth_mask):
                    cloth_positions = u_list[0][all_cloth_mask]
                    cloth_colors = get_cloth_colors(cloth_faces, cloth_positions, "orange")
                    cloth_mesh_gt = create_mesh_collection(cloth_positions, cloth_faces, cloth_colors, alpha=0.4)
                    if cloth_mesh_gt:
                        gt_ax.add_collection3d(cloth_mesh_gt)
                
                if len(sphere_faces) > 0 and hasattr(data, 'sphere_mesh_pos'):
                    sphere_mesh_pos = data.sphere_mesh_pos.clone().cpu().numpy()
                    num_spheres = getattr(data, 'num_spheres', 3)
                    p_sphere_per_sphere = getattr(data, 'p_sphere_per_sphere', len(sphere_mesh_pos) // num_spheres)
                    
                    for sphere_idx in range(num_spheres):
                        start_idx = sphere_idx * p_sphere_per_sphere
                        end_idx = (sphere_idx + 1) * p_sphere_per_sphere
                        sphere_mesh = sphere_mesh_pos[start_idx:end_idx]
                        # sphere_colors = get_sphere_colors(sphere_mesh, sphere_faces, "plasma")
                        sphere_colors = "#fb0000"  # Gray color for GT
                        sphere_mesh_gt = create_mesh_collection(sphere_mesh, sphere_faces, sphere_colors, alpha=1.0, linewidths=0.5, edgecolors="#4A4949")
                        if sphere_mesh_gt:
                            gt_ax.add_collection3d(sphere_mesh_gt)
                
                # if np.any(sphere_center_mask):
                #     center_positions = u_list[0][sphere_center_mask]
                #     colors = ['red', 'blue', 'green'][:len(center_positions)]
                #     for idx, (center_pos, color) in enumerate(zip(center_positions, colors)):
                #         gt_ax.scatter(*center_pos.reshape(1, -1).T, color=color, s=150, alpha=1.0)
                
                # Cloth corner nodes (highlight them)
                if np.any(cloth_corner_mask):
                    corner_pos = u_list[0][cloth_corner_mask]
                    gt_ax.scatter(*corner_pos.T, color='green', s=80, alpha=0.8, marker='s')
                
                if t == self.trajectory_length - 2:
                    # Copy axis limits and settings from original
                    gt_ax.set_xlim(axes[1].get_xlim())
                    gt_ax.set_ylim(axes[1].get_ylim())
                    gt_ax.set_zlim([9, 10.5])
                    # gt_ax.set_zlim(axes[1].get_zlim())
                    gt_ax.view_init(elev=self.plot_elev, azim=self.plot_azim)
                    gt_ax.axis("off")
                    plt.tight_layout()
                    plt.savefig(save_dir / "true_3d.png", dpi=150, bbox_inches='tight')
                    plt.close()
                
                # Export ground truth to VTK
                cloth_positions_gt = u_list[0][all_cloth_mask] if np.any(all_cloth_mask) else []
                corner_positions_gt = u_list[0][cloth_corner_mask] if np.any(cloth_corner_mask) else []
                sphere_mesh_gt = data.sphere_mesh_pos.cpu().numpy() if hasattr(data, 'sphere_mesh_pos') else []
                
                saved_files_gt = save_separate_vtk_files(
                    save_dir, f"true_{t}",
                    cloth_positions_gt, cloth_faces,
                    sphere_mesh_gt, sphere_faces,
                    corner_positions_gt
                )
                
                print(f"Saved 3D plots and VTK files to {save_dir}")
                print(f"VTK files: {[f.name for f in saved_files + saved_files_gt]}")
            

        return fig
