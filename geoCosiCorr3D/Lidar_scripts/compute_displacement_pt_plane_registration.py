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
            points_gpu = cp.asarray(points)
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
    '''
    def icp_registration(self, src_pts, tgt_pts, grid_size=None):
        # Quick sanity
        if len(src_pts) < 50 or len(tgt_pts) < 50:
            return None, None, False

        spcd = o3d.geometry.PointCloud()
        tpcd = o3d.geometry.PointCloud()
        spcd.points = o3d.utility.Vector3dVector(src_pts)
        tpcd.points = o3d.utility.Vector3dVector(tgt_pts)

        # Downsample per cell for stability & speed
        #voxel = max((grid_size or 1.0)/20.0, 0.05)
        #if len(spcd.points) > 2000: spcd = spcd.voxel_down_sample(voxel)
        #if len(tpcd.points) > 2000: tpcd = tpcd.voxel_down_sample(voxel)

        # ---- Stage 1: coarse point-to-point (to get a good init) ----
        thr_coarse = min(1.0 * (grid_size or 2.0), 3.0)
        reg_pp = o3d.pipelines.registration.registration_icp(
            spcd, tpcd, thr_coarse, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40)
        )
        T_init = reg_pp.transformation

        # ---- Normals on TARGET only (used by point-to-plane) ----
        tpcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        # Orient consistently (simpler & fast)
        tpcd.orient_normals_towards_camera_location(tpcd.get_center())

        # ---- Stage 2: refine with point-to-plane ----
        thr_fine = min(0.5 * (grid_size or 2.0), 2.0)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

        reg = o3d.pipelines.registration.registration_icp(
            spcd, tpcd, thr_fine, T_init,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
        )

        ok = (reg.fitness > 0.20) and (reg.inlier_rmse < 0.3 * thr_fine)
        if not ok:
            return None, None, False

        T = reg.transformation
        return T[:3, 3], T, True
    '''
    
    
    def icp_registration(self, source_points, target_points):
        """Perform ICP registration between point clouds (point-to-plane)."""
        # Need enough points to estimate surface normals reliably
        if len(source_points) < 50 or len(target_points) < 50:
            return None, None, False

        # Create Open3D point clouds
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)

        # Estimate normals for point-to-plane ICP
        try:
            # Use adaptive radius based on point density
            bbox_diag = np.linalg.norm(np.max(source_points, axis=0) - np.min(source_points, axis=0))
            normal_radius = min(0.5, bbox_diag / 20)  # Adaptive radius, max 0.5m
            
            source_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=50)
            )
            target_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=50)
            )
            
            # Orient normals consistently (important for point-to-plane ICP)
            source_pcd.orient_normals_consistent_tangent_plane(k=30)
            target_pcd.orient_normals_consistent_tangent_plane(k=30)
            
        except Exception as e:
            print(f"Normal estimation failed: {e}")
            return None, None, False

        # ICP registration (point-to-plane)
        threshold = 1.0  # Reduced threshold for better precision
        trans_init = np.eye(4)
        try:
            reg_result = o3d.pipelines.registration.registration_icp(
                source_pcd,
                target_pcd,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=100
                )
            )
            
            # More strict fitness requirement for point-to-plane
            if reg_result.fitness > 0.3 and reg_result.inlier_rmse < threshold:
                transformation = reg_result.transformation
                displacement = transformation[:3, 3]
                return displacement, transformation, True
            else:
                return None, None, False
                
        except Exception as e:
            print(f"ICP failed: {e}")
            return None, None, False
    
    def compute_displacement_vectors(self, file1_path, file2_path, grid_size, output_suffix):
        """Compute displacement vectors between two point clouds."""
        print("=== LiDAR Displacement Vector Computation (pt-plane registration) ===")
        print(f"\nProcessing: {os.path.basename(file1_path)} -> {os.path.basename(file2_path)}")
        print(f"Grid size: {grid_size}")
        #print(f"Max points per chunk: {self.max_points_per_chunk:,}")
        
        start_time = time.time()
        
        # Load point clouds
        points1, count1 = self.load_point_cloud_chunked(file1_path)
        points2, count2 = self.load_point_cloud_chunked(file2_path)
        
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
            
            if len(cell_points1) > 50 and len(cell_points2) > 50:  # Minimum points for reliable point-to-plane ICP
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
    processor = LidarDisplacementChunked(use_gpu=True) 
    
    # File paths
    base_path = "/home/bozhouzh/CEC/CEC/geoCosiCorr3D/Lidar_Data"
    file1 = os.path.join(base_path, "upsampled_subset_ptCloud_site1_2016.pcd")
    file2 = os.path.join(base_path, "upsampled_subset_ptCloud_site1_2018.pcd")
    
    # Compute displacement vectors
    results = processor.compute_displacement_vectors(
        file1, file2, 
        grid_size=3,  # Larger grid for faster processing
        output_suffix='pt_plane_2016to2018'
    )
    
    if results:
        print("=== Processing Complete ===")
    else:
        print("=== Processing Failed ===")

if __name__ == "__main__":
    main()
