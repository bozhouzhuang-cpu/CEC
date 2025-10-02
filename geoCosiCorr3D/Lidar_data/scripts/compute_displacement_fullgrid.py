#!/usr/bin/env python3
"""
Full-grid displacement computation: saves a value for every grid cell.
Failed/insufficient ICP cells are filled with NaN (and success_mask=False).
Leaves compute_displacement_chunked.py unchanged.

Usage:
  python compute_displacement_fullgrid.py
"""

import numpy as np
import open3d as o3d
import os
import time
import gc

try:
	import cupy as cp
	GPU_AVAILABLE = True
	print("GPU (CuPy) available for acceleration")
except ImportError:
	print("CuPy not available, falling back to CPU")
	GPU_AVAILABLE = False

class LidarDisplacementFullGrid:
	def __init__(self, use_gpu=True, max_points_per_chunk=50000000):
		self.use_gpu = use_gpu and GPU_AVAILABLE
		self.max_points_per_chunk = max_points_per_chunk
		if self.use_gpu:
			print("Using GPU acceleration")
			free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
			print(f"GPU memory: {free_bytes/1024**3:.1f}GB free / {total_bytes/1024**3:.1f}GB total")
		else:
			print("Using CPU processing")

	def load_point_cloud_chunked(self, file_path):
		if not os.path.exists(file_path):
			raise FileNotFoundError(f"PCD file not found: {file_path}")
		print(f"Loading {os.path.basename(file_path)} in chunks...")
		pcd_full = o3d.io.read_point_cloud(file_path)
		total_points = len(pcd_full.points)
		print(f"Total points: {total_points:,}")
		if total_points <= self.max_points_per_chunk:
			points = np.asarray(pcd_full.points)
			return points, total_points
		sample_ratio = self.max_points_per_chunk / total_points
		print(f"Sampling {sample_ratio:.1%} of points ({self.max_points_per_chunk:,} points)")
		step = max(1, int(1 / sample_ratio))
		indices = np.arange(0, total_points, step)[:self.max_points_per_chunk]
		pcd_sampled = pcd_full.select_by_index(indices)
		points = np.asarray(pcd_sampled.points)
		del pcd_full, pcd_sampled
		gc.collect()
		print(f"Loaded {len(points):,} sampled points")
		return points, len(points)

	def convert_feet_to_meters(self, points):
		return points * 0.3048

	def normalize_point_cloud(self, points):
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
		all_points = np.vstack([points1, points2])
		if self.use_gpu:
			all_points_gpu = cp.asarray(all_points)
			min_coords = cp.min(all_points_gpu, axis=0)
			max_coords = cp.max(all_points_gpu, axis=0)
			min_coords = cp.asnumpy(min_coords)
			max_coords = cp.asnumpy(max_coords)
		else:
			min_coords = np.min(all_points, axis=0)
			max_coords = np.max(all_points, axis=0)
		x_range = np.arange(min_coords[0], max_coords[0] + grid_size, grid_size)
		y_range = np.arange(min_coords[1], max_coords[1] + grid_size, grid_size)
		grid_centers = []
		for x in x_range[:-1]:
			for y in y_range[:-1]:
				grid_centers.append([x + grid_size / 2, y + grid_size / 2])
		print(f"Created grid with {len(grid_centers)} cells ({len(x_range)-1} x {len(y_range)-1})")
		return np.array(grid_centers)

	def filter_points_in_grid(self, points, grid_center, grid_size):
		half_size = grid_size / 2
		if self.use_gpu:
			pts = cp.asarray(points)
			c = cp.asarray(grid_center)
			mask = (
				(pts[:, 0] >= c[0] - half_size) & (pts[:, 0] <= c[0] + half_size) &
				(pts[:, 1] >= c[1] - half_size) & (pts[:, 1] <= c[1] + half_size)
			)
			return cp.asnumpy(pts[mask])
		else:
			mask = (
				(points[:, 0] >= grid_center[0] - half_size) & (points[:, 0] <= grid_center[0] + half_size) &
				(points[:, 1] >= grid_center[1] - half_size) & (points[:, 1] <= grid_center[1] + half_size)
			)
			return points[mask]

	def icp_registration(self, source_points, target_points, threshold=1.0):
		if len(source_points) < 10 or len(target_points) < 10:
			return None, None, False
		source_pcd = o3d.geometry.PointCloud()
		source_pcd.points = o3d.utility.Vector3dVector(source_points)
		target_pcd = o3d.geometry.PointCloud()
		target_pcd.points = o3d.utility.Vector3dVector(target_points)
		try:
			reg_result = o3d.pipelines.registration.registration_icp(
				source_pcd, target_pcd, threshold, np.eye(4),
				o3d.pipelines.registration.TransformationEstimationPointToPoint()
			)
			if reg_result.fitness > 0.1:
				T = reg_result.transformation
				disp = T[:3, 3]
				return disp, T, True
			else:
				return None, None, False
		except Exception as e:
			print(f"ICP failed: {e}")
			return None, None, False

	def compute_displacement_vectors(self, file1_path, file2_path, grid_size, output_suffix):
		print("=== LiDAR Displacement Vector Computation (Full Grid) ===")
		print(f"\nProcessing: {os.path.basename(file1_path)} -> {os.path.basename(file2_path)}")
		print(f"Grid size: {grid_size}")
		print(f"Max points per chunk: {self.max_points_per_chunk:,}")
		start_time = time.time()
		pts1, count1 = self.load_point_cloud_chunked(file1_path)
		pts2, count2 = self.load_point_cloud_chunked(file2_path)
		print("Converting coordinates from feet to meters...")
		pts1 = self.convert_feet_to_meters(pts1)
		pts2 = self.convert_feet_to_meters(pts2)
		print("Normalizing coordinates...")
		pts1n, off1 = self.normalize_point_cloud(pts1)
		pts2n, off2 = self.normalize_point_cloud(pts2)
		print("Creating spatial grid...")
		centers = self.create_grid(pts1n, pts2n, grid_size)
		ncells = len(centers)
		# Preallocate full-grid outputs
		displacements = np.full((ncells, 3), np.nan)
		transformations = np.full((ncells, 4, 4), np.nan, dtype=float)
		success_mask = np.zeros(ncells, dtype=bool)
		print(f"Processing {ncells} grid cells...")
		for i, c in enumerate(centers):
			if i % 100 == 0:
				print(f"  Progress: {i}/{ncells} ({100*i/ncells:.1f}%)")
			cell1 = self.filter_points_in_grid(pts1n, c, grid_size)
			cell2 = self.filter_points_in_grid(pts2n, c, grid_size)
			ok = False
			disp = None
			T = None
			if len(cell1) > 20 and len(cell2) > 20:
				disp, T, ok = self.icp_registration(cell1, cell2, threshold=3.0)
			if ok:
				displacements[i] = disp
				transformations[i] = T
				success_mask[i] = True
		# Stats
		magnitudes = np.linalg.norm(displacements, axis=1)
		mean_mag = np.nanmean(magnitudes)
		max_mag = np.nanmax(magnitudes)
		successful = int(success_mask.sum())
		print(f"Successfully processed {successful} out of {ncells} grid cells")
		print(f"  Mean displacement: {mean_mag:.3f} m")
		print(f"  Max displacement:  {max_mag:.3f} m")
		# Save results (full grid)
		out_dir = "/home/bozhouzh/Geospatial-COSICorr3D/geoCosiCorr3D/Lidar_results/Site1"
		os.makedirs(out_dir, exist_ok=True)
		out_file = os.path.join(out_dir, f"displacement_results_fullgrid_{output_suffix}.npz")
		np.savez(out_file,
			grid_centers=centers,
			displacements=displacements,
			displacement_magnitudes=magnitudes,
			transformations=transformations,
			success_mask=success_mask,
			grid_size=grid_size,
			processing_time=time.time() - start_time,
			total_cells=ncells,
			successful_cells=successful,
			valid_cells=ncells,
			offset1=off1,
			offset2=off2,
			original_points1=count1,
			original_points2=count2
		)
		print(f"Results saved to: {out_file}")
		return {
			'grid_centers': centers,
			'displacements': displacements,
			'displacement_magnitudes': magnitudes,
			'transformations': transformations,
			'success_mask': success_mask,
			'grid_size': grid_size,
			'processing_time': time.time() - start_time,
			'total_cells': ncells,
			'successful_cells': successful,
			'valid_cells': ncells,
			'offset1': off1,
			'offset2': off2,
			'original_points1': count1,
			'original_points2': count2
		}

def main():
	processor = LidarDisplacementFullGrid(use_gpu=True, max_points_per_chunk=100000000)
	base_path = '/home/bozhouzh/Geospatial-COSICorr3D/geoCosiCorr3D/Lidar_data/Site1'
	file1 = os.path.join(base_path, 'upsampled_subset_ptCloud_site1_2014.pcd')
	file2 = os.path.join(base_path, 'upsampled_subset_ptCloud_site1_2016.pcd')
	res = processor.compute_displacement_vectors(
		file1, file2,
		grid_size=3.048,
		output_suffix='2014to2016'
	)
	if res:
		print("=== Full-grid Processing Complete ===")
	else:
		print("=== Full-grid Processing Failed ===")

if __name__ == "__main__":
	main()
