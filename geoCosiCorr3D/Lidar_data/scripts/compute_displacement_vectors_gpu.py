#!/usr/bin/env python3
"""
GPU-accelerated displacement vector computation for LiDAR point clouds using ICP.
Converts MATLAB ICP workflow to Python with CUDA acceleration.

Requirements:
    pip install open3d numpy scipy cupy-cuda11x matplotlib
    
Usage:


    python compute_displacement_vectors_gpu.py
"""

import numpy as np
import open3d as o3d
import os
import time
from scipy.spatial.transform import Rotation
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU (CuPy) available for acceleration")
except ImportError:
    print("CuPy not available, falling back to CPU")
    GPU_AVAILABLE = False

class LidarDisplacementGPU:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            print("Using GPU acceleration")
        else:
            print("Using CPU processing")
    
    def load_point_cloud(self, file_path):
        """Load PCD file and return Open3D point cloud."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PCD file not found: {file_path}")
        
        pcd = o3d.io.read_point_cloud(file_path)
        if pcd.is_empty():
            raise ValueError(f"Point cloud is empty: {file_path}")
        
        print(f"Loaded {len(pcd.points)} points from {os.path.basename(file_path)}")
        return pcd
    
    def normalize_point_cloud(self, pcd):
        """Normalize point cloud by subtracting minimum coordinates."""
        points = np.asarray(pcd.points)
        
        if self.use_gpu:
            points_gpu = cp.asarray(points)
            min_coords = cp.min(points_gpu, axis=0)
            normalized_points = points_gpu - min_coords
            normalized_points_cpu = cp.asnumpy(normalized_points)
            min_coords_cpu = cp.asnumpy(min_coords)
        else:
            min_coords_cpu = np.min(points, axis=0)
            normalized_points_cpu = points - min_coords_cpu
        
        # Create new point cloud with normalized coordinates
        normalized_pcd = o3d.geometry.PointCloud()
        normalized_pcd.points = o3d.utility.Vector3dVector(normalized_points_cpu)
        
        return normalized_pcd, min_coords_cpu
    
    def create_grid(self, pcd, grid_size):
        """Create grid boundaries for processing."""
        points = np.asarray(pcd.points)
        
        if self.use_gpu:
            points_gpu = cp.asarray(points)
            max_coords = cp.max(points_gpu, axis=0)
            max_coords_cpu = cp.asnumpy(max_coords)
        else:
            max_coords_cpu = np.max(points, axis=0)
        
        x_max, y_max = max_coords_cpu[0], max_coords_cpu[1]
        
        x_grid = np.arange(0, x_max + grid_size, grid_size)
        y_grid = np.arange(0, y_max + grid_size, grid_size)
        
        return x_grid, y_grid, max_coords_cpu
    
    def filter_points_in_grid(self, pcd, x_min, x_max, y_min, y_max):
        """Filter points within grid cell boundaries."""
        points = np.asarray(pcd.points)
        
        if self.use_gpu:
            points_gpu = cp.asarray(points)
            mask = ((points_gpu[:, 0] >= x_min) & (points_gpu[:, 0] < x_max) & 
                   (points_gpu[:, 1] >= y_min) & (points_gpu[:, 1] < y_max))
            indices = cp.where(mask)[0]
            indices_cpu = cp.asnumpy(indices)
        else:
            mask = ((points[:, 0] >= x_min) & (points[:, 0] < x_max) & 
                   (points[:, 1] >= y_min) & (points[:, 1] < y_max))
            indices_cpu = np.where(mask)[0]
        
        if len(indices_cpu) > 0:
            sub_pcd = pcd.select_by_index(indices_cpu)
            return sub_pcd, len(indices_cpu)
        else:
            return None, 0
    
    def icp_registration(self, source_pcd, target_pcd):
        """Perform ICP registration between two point clouds."""
        # Set ICP parameters for better convergence
        threshold = 2.0  # Distance threshold
        trans_init = np.eye(4)  # Initial transformation
        
        # Use point-to-point ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            target_pcd, source_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # Extract transformation matrix
        transformation_matrix = reg_p2p.transformation
        rmse = reg_p2p.inlier_rmse
        
        # Extract translation vector
        translation_vector = transformation_matrix[:3, 3]
        
        # Extract rotation matrix and convert to Euler angles
        rotation_matrix = transformation_matrix[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        rotation_angles = rotation.as_euler('xyz', degrees=True)
        
        # Invert to match MATLAB convention (source to target vs target to source)
        rotation_angles = -rotation_angles
        
        return translation_vector, rotation_angles, rmse
    
    def compute_displacement_vectors(self, source_pcd_path, target_pcd_path, 
                                   grid_size, file_suffix):
        """Main function to compute displacement vectors."""
        print(f"\nProcessing: {os.path.basename(source_pcd_path)} -> {os.path.basename(target_pcd_path)}")
        print(f"Grid size: {grid_size}")
        
        # Load point clouds
        source_old = self.load_point_cloud(source_pcd_path)
        target_old = self.load_point_cloud(target_pcd_path)
        
        # Normalize point clouds (subtract minimum coordinates)
        source, min_coords = self.normalize_point_cloud(source_old)
        target, _ = self.normalize_point_cloud(target_old)
        
        # Create grid
        x_grid, y_grid, max_coords = self.create_grid(source, grid_size)
        
        # Initialize result arrays
        grid_centers = []
        displacement_vectors = []
        rotation_vectors = []
        rmse_array = []
        
        total_cells = (len(x_grid) - 1) * (len(y_grid) - 1)
        processed_cells = 0
        
        print(f"Processing {total_cells} grid cells...")
        
        start_time = time.time()
        
        # Process each grid cell
        for i in range(len(x_grid) - 1):
            for j in range(len(y_grid) - 1):
                print(f'Processing grid cell: [i, j] = [{i}, {j}] ({processed_cells+1}/{total_cells})')
                
                x_min, x_max = x_grid[i], x_grid[i + 1]
                y_min, y_max = y_grid[j], y_grid[j + 1]
                
                # Filter points in current grid cell
                source_sub, source_count = self.filter_points_in_grid(source, x_min, x_max, y_min, y_max)
                target_sub, target_count = self.filter_points_in_grid(target, x_min, x_max, y_min, y_max)
                
                # Only process if both subclouds have sufficient points
                if source_count > 3 and target_count > 3:
                    try:
                        # Perform ICP registration
                        translation_vector, rotation_angles, rmse = self.icp_registration(
                            source_sub, target_sub)
                        
                        # Calculate grid center
                        source_points = np.asarray(source_sub.points)
                        grid_center = np.mean(source_points, axis=0)
                        
                        # Store results
                        grid_centers.append(grid_center)
                        displacement_vectors.append(translation_vector)
                        rotation_vectors.append(rotation_angles)
                        rmse_array.append(rmse)
                        
                        print(f"  -> Successful registration: RMSE = {rmse:.4f}")
                        
                    except Exception as e:
                        print(f"  -> Registration failed: {e}")
                else:
                    print(f"  -> Insufficient points: source={source_count}, target={target_count}")
                
                processed_cells += 1
        
        elapsed_time = time.time() - start_time
        print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        print(f"Successfully processed {len(grid_centers)} grid cells")
        
        # Convert to numpy arrays
        if grid_centers:
            grid_centers = np.array(grid_centers)
            displacement_vectors = np.array(displacement_vectors)
            rotation_vectors = np.array(rotation_vectors)
            rmse_array = np.array(rmse_array)
        else:
            print("Warning: No successful registrations found!")
            grid_centers = np.array([])
            displacement_vectors = np.array([])
            rotation_vectors = np.array([])
            rmse_array = np.array([])
        
        # Save results
        output_file = f'trial_dispVec_site1_{file_suffix}.npz'
        np.savez(output_file,
                 gridCenters=grid_centers,
                 displacementVectors=displacement_vectors,
                 rotationVectors=rotation_vectors,
                 rmseArray=rmse_array,
                 x_max=max_coords[0],
                 x_min=min_coords[0],
                 y_max=max_coords[1],
                 y_min=min_coords[1],
                 z_min=min_coords[2])
        
        print(f"Results saved to: {output_file}")
        
        return {
            'grid_centers': grid_centers,
            'displacement_vectors': displacement_vectors,
            'rotation_vectors': rotation_vectors,
            'rmse_array': rmse_array,
            'bounds': {
                'x_max': max_coords[0], 'x_min': min_coords[0],
                'y_max': max_coords[1], 'y_min': min_coords[1],
                'z_min': min_coords[2]
            }
        }
    
    def visualize_displacement_vectors(self, results, file_suffix):
        """Create visualization of displacement vectors."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("Matplotlib not available for visualization")
            return
        
        if len(results['grid_centers']) == 0:
            print("No data to visualize")
            return
        
        grid_centers = results['grid_centers']
        displacement_vectors = results['displacement_vectors']
        
        # Create 2D displacement plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # XY displacement vectors
        ax1.quiver(grid_centers[:, 0], grid_centers[:, 1], 
                  displacement_vectors[:, 0], displacement_vectors[:, 1],
                  angles='xy', scale_units='xy', scale=1, alpha=0.7)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'XY Displacement Vectors - {file_suffix}')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Z displacement as color map
        scatter = ax2.scatter(grid_centers[:, 0], grid_centers[:, 1], 
                            c=displacement_vectors[:, 2], cmap='RdBu_r', s=50)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'Z Displacement - {file_suffix}')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Z Displacement (m)')
        
        plt.tight_layout()
        plot_filename = f'dispVec_site1_{file_suffix}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Displacement plot saved to: {plot_filename}")
        plt.close()


def main():
    """Main execution function."""
    # Initialize processor
    processor = LidarDisplacementGPU(use_gpu=True)
    
    # Define file paths
    base_path = "/home/bozhouzh/Geospatial-COSICorr3D/geoCosiCorr3D/Lidar_data/Site1"
    
    ptcloud_files = {
        '2014': os.path.join(base_path, 'upsampled_subset_ptCloud_site1_2014.pcd'),
        '2016': os.path.join(base_path, 'upsampled_subset_ptCloud_site1_2016.pcd'),
        # Add other years as needed
        # '2018': os.path.join(base_path, 'upsampled_subset_ptCloud_site1_2018.pcd'),
        # '2019': os.path.join(base_path, 'upsampled_subset_ptCloud_site1_2019.pcd'),
        # '2020': os.path.join(base_path, 'upsampled_subset_ptCloud_site1_2020.pcd'),
        # '2021': os.path.join(base_path, 'upsampled_subset_ptCloud_site1_2021.pcd'),
    }
    
    # Grid size parameter
    grid_size = 10.0 
    
    # Process displacement vectors
    print("=== LiDAR Displacement Vector Computation ===")
    
    # From 2014 to 2016
    if '2014' in ptcloud_files and '2016' in ptcloud_files:
        if os.path.exists(ptcloud_files['2014']) and os.path.exists(ptcloud_files['2016']):
            results = processor.compute_displacement_vectors(
                ptcloud_files['2014'], 
                ptcloud_files['2016'], 
                grid_size, 
                '2014to2016'
            )
            processor.visualize_displacement_vectors(results, '2014to2016')
        else:
            print("Warning: Required PCD files not found for 2014-2016 comparison")
    
    # Add more year combinations as needed
    # if '2014' in ptcloud_files and '2018' in ptcloud_files:
    #     results = processor.compute_displacement_vectors(
    #         ptcloud_files['2014'], ptcloud_files['2018'], grid_size, '2014to2018')
    #     processor.visualize_displacement_vectors(results, '2014to2018')
    
    print("\n=== Processing Complete ===")


if __name__ == "__main__":
    main()
