#!/usr/bin/env python3
"""
Memory-efficient GPU-accelerated displacement vector computation for large LiDAR point clouds.
Processes data in chunks to handle files that don't fit in memory.

Requirements:
    pip install open3d numpy scipy cupy-cuda11x matplotlib
    
Usage:
    python compute_displacement_chunked.py
"""

import numpy as np
import open3d as o3d
import os
import time
from scipy.spatial.transform import Rotation
import gc
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU (CuPy) available for acceleration")
except ImportError:
    print("CuPy not available, falling back to CPU")
    GPU_AVAILABLE = False

class LidarDisplacementChunked:
    def __init__(self, use_gpu=True):  
        self.use_gpu = use_gpu and GPU_AVAILABLE
        #self.max_points_per_chunk = max_points_per_chunk
        
        if self.use_gpu:
            print("Using GPU acceleration")
            # Check GPU memory
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            print(f"GPU memory: {free_bytes/1024**3:.1f}GB free / {total_bytes/1024**3:.1f}GB total")
        else:
            print("Using CPU processing")
    
    def load_point_cloud_chunked(self, file_path):
        """Load PCD file in chunks to manage memory."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PCD file not found: {file_path}")
        
        print(f"Loading {os.path.basename(file_path)}...")
        
        # Load all points
        pcd_full = o3d.io.read_point_cloud(file_path)
        total_points = len(pcd_full.points)
        print(f"Total points: {total_points:,}")
        
        points = np.asarray(pcd_full.points)
        
        # Clean up
        del pcd_full
        gc.collect()
        
        print(f"Loaded all {len(points):,} points")
        return points, len(points)
    
    def convert_feet_to_meters(self, points):
        """Convert coordinates from feet to meters."""
        return points * 0.3048
    
    def normalize_point_cloud(self, points):
        """Normalize point cloud by subtracting minimum coordinates."""
        if self.use_gpu:
            #points_gpu = cp.asarray(points)
            points_gpu = cp.asarray(points, dtype=cp.float32)
            min_coords = cp.min(points_gpu, axis=0)
            normalized = points_gpu - min_coords
            return cp.asnumpy(normalized), cp.asnumpy(min_coords)
        else:
            min_coords = np.min(points, axis=0)
            normalized = points - min_coords
            return normalized, min_coords
    
    def create_grid(self, points1, points2, grid_size):
        """Create 2D grid centers over XY (CPU-only; GPU-safe)."""
        p1_min = np.min(points1[:, :2], axis=0)
        p1_max = np.max(points1[:, :2], axis=0)
        p2_min = np.min(points2[:, :2], axis=0)
        p2_max = np.max(points2[:, :2], axis=0)

        mins = np.minimum(p1_min, p2_min)
        maxs = np.maximum(p1_max, p2_max)

        spans = np.maximum(maxs - mins, np.array([grid_size, grid_size], dtype=np.float32))
        nx, ny = np.ceil(spans / grid_size).astype(int)

        xs = mins[0] + (np.arange(nx, dtype=np.float32) + 0.5) * grid_size
        ys = mins[1] + (np.arange(ny, dtype=np.float32) + 0.5) * grid_size
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        centers = np.stack((X, Y), axis=-1).reshape(-1, 2).astype(np.float32, copy=False)

        print(f"Created grid with {centers.shape[0]} cells ({nx} x {ny})")
        return centers

    def filter_points_in_grid(self, points, grid_center, grid_size):
        """Filter points within a grid cell."""
        half_size = grid_size / 2
        
        if self.use_gpu:
            points_gpu = cp.asarray(points)
            center_gpu = cp.asarray(grid_center)
            
            mask = (
                (points_gpu[:, 0] >= center_gpu[0] - half_size) &
                (points_gpu[:, 0] <= center_gpu[0] + half_size) &
                (points_gpu[:, 1] >= center_gpu[1] - half_size) &
                (points_gpu[:, 1] <= center_gpu[1] + half_size)
            )
            
            filtered = points_gpu[mask]
            return cp.asnumpy(filtered)
        else:
            mask = (
                (points[:, 0] >= grid_center[0] - half_size) &
                (points[:, 0] <= grid_center[0] + half_size) &
                (points[:, 1] >= grid_center[1] - half_size) &
                (points[:, 1] <= grid_center[1] + half_size)
            )
            return points[mask]
    
    def icp_registration(self, source_points, target_points):
        """Perform ICP registration between point clouds."""
        if len(source_points) < 3 or len(target_points) < 3:
            return None, None, False
        
        # Create Open3D point clouds
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        
        # ICP registration
        threshold = 2.0  # meters
        trans_init = np.eye(4)
        
        try:
            reg_result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            if reg_result.fitness > 0.1:  # Good registration
                transformation = reg_result.transformation
                # Extract translation (displacement)
                displacement = transformation[:3, 3]
                return displacement, transformation, True
            else:
                return None, None, False
                
        except Exception as e:
            print(f"ICP failed: {e}")
            return None, None, False
    
    def compute_displacement_vectors(self, file1_path, file2_path, grid_size, output_suffix):
        """Compute displacement vectors between two point clouds."""
        print("=== LiDAR Displacement Vector Computation (pt-pt registration) ===")
        print(f"\nProcessing: {os.path.basename(file1_path)} -> {os.path.basename(file2_path)}")
        print(f"Grid size: {grid_size}")
        #print(f"Max points per chunk: {self.max_points_per_chunk:,}")
        
        start_time = time.time()
        
        # Load point clouds
        points1, count1 = self.load_point_cloud_chunked(file1_path)
        points2, count2 = self.load_point_cloud_chunked(file2_path)
        
        points1 = points1.astype(np.float32, copy=False) # added newly ZZ
        points2 = points2.astype(np.float32, copy=False) # added newly ZZ

        # Convert from feet to meters
        print("Converting coordinates from feet to meters...")
        points1 = self.convert_feet_to_meters(points1)
        points2 = self.convert_feet_to_meters(points2)
        
        # Normalize coordinates
        print("Normalizing coordinates...")
        points1_norm, offset1 = self.normalize_point_cloud(points1)
        points2_norm, offset2 = self.normalize_point_cloud(points2)
        
        # Create spatial grid
        print("Creating spatial grid...")
        grid_centers = self.create_grid(points1_norm, points2_norm, grid_size)
        
        # Compute displacements for each grid cell
        displacements = []
        valid_centers = []
        transformations = []
        
        print(f"Processing {len(grid_centers)} grid cells...")
        
        for i, center in enumerate(grid_centers):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(grid_centers)} ({100*i/len(grid_centers):.1f}%)")
            
            # Filter points in this grid cell
            cell_points1 = self.filter_points_in_grid(points1_norm, center, grid_size)
            cell_points2 = self.filter_points_in_grid(points2_norm, center, grid_size)
            
            #if len(cell_points1) > 0 and len(cell_points2) > 0:  # Minimum points for reliable ICP
            displacement, transformation, success = self.icp_registration(cell_points1, cell_points2)
                
            if success:
                displacements.append(displacement)
                valid_centers.append(center)
                transformations.append(transformation)
        
        print(f"Successfully processed {len(displacements)} out of {len(grid_centers)} grid cells")
        
        if len(displacements) == 0:
            print("No valid displacements computed!")
            return None
        
        # Convert to arrays
        displacements = np.array(displacements)
        valid_centers = np.array(valid_centers)
        
        # Compute statistics
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        
        results = {
            'grid_centers': valid_centers,
            'displacements': displacements,
            'displacement_magnitudes': displacement_magnitudes,
            'transformations': transformations,
            'grid_size': grid_size,
            'processing_time': time.time() - start_time,
            'total_cells': len(grid_centers),
            'valid_cells': len(displacements),
            'offset1': offset1,
            'offset2': offset2,
            'original_points1': count1,
            'original_points2': count2
        }
        
        print(f"\nResults:")
        print(f"  Valid cells: {len(displacements)}/{len(grid_centers)} ({100*len(displacements)/len(grid_centers):.1f}%)")
        print(f"  Mean displacement: {np.mean(displacement_magnitudes):.3f} m")
        print(f"  Max displacement: {np.max(displacement_magnitudes):.3f} m")
        print(f"  Processing time: {results['processing_time']:.1f} seconds")
        
        # Save results
        output_dir = "/home/bozhouzh/CEC/CEC/geoCosiCorr3D/Lidar_results"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"displacement_results_{output_suffix}.npz")
        np.savez(output_file, **results)
        print(f"Results saved to: {output_file}")
        
        return results

def main():
    """Main function to run displacement computation."""
    
    # Initialize processor with smaller chunks for memory efficiency
    processor = LidarDisplacementChunked(use_gpu=True)  # 3M points
    
    # File paths
    base_path = "/home/bozhouzh/CEC/CEC/geoCosiCorr3D/Lidar_Data/Site_1"
    file1 = os.path.join(base_path, "upsampled_subset_ptCloud_site1_2018.pcd")
    file2 = os.path.join(base_path, "upsampled_subset_ptCloud_site1_2019.pcd")

    # Compute displacement vectors
    results = processor.compute_displacement_vectors(
        file1, file2, 
        grid_size=3,  # Larger grid for faster processing
        output_suffix='_2018to2019'
    )
    
    if results:
        print("=== Processing Complete ===")
    else:
        print("=== Processing Failed ===")

if __name__ == "__main__":
    main()
