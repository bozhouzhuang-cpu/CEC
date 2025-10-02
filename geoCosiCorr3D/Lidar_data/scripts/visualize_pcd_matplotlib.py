#!/usr/bin/env python3
"""
PCD visualizer using matplotlib (WSL-friendly alternative).

Usage:
  python visualize_pcd_matplotlib.py /absolute/path/to/pointcloud.pcd

If no path is provided, it defaults to the Site1 sample.

Features:
  - Works in WSL without X11 setup
  - Interactive 3D plots with matplotlib
  - Automatic downsampling for large point clouds
  - Color-coded height visualization
"""

import argparse
import os
import sys
import numpy as np

DEFAULT_PCD = (
    "/home/bozhouzh/Geospatial-COSICorr3D/geoCosiCorr3D/Lidar_data/Site1/"
    "sub_pc_site1_2014.pcd"
)


def parse_pcd_file(pcd_path):
    """Parse PCD file and extract point coordinates."""
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"PCD file not found: {pcd_path}")
    
    points = []
    data_section = False
    point_count = 0
    
    print(f"Reading PCD file: {pcd_path}")
    
    with open(pcd_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            
            if line.startswith('POINTS'):
                point_count = int(line.split()[1])
                print(f"Expected points: {point_count}")
                continue
                
            if line == 'DATA ascii':
                data_section = True
                continue
            elif line == 'DATA binary':
                print("Binary PCD files not supported yet. Use ASCII format.")
                sys.exit(1)
                
            if data_section and line:
                try:
                    coords = line.split()
                    if len(coords) >= 3:
                        x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                        points.append([x, y, z])
                except (ValueError, IndexError):
                    continue
                    
            # Progress indicator for large files
            if data_section and len(points) % 1000000 == 0 and len(points) > 0:
                print(f"Loaded {len(points):,} points...")
    
    if not points:
        raise ValueError("No valid points found in PCD file")
    
    return np.array(points)


def downsample_points(points, max_points=100000):
    """Downsample points if there are too many for visualization."""
    if len(points) <= max_points:
        return points
    
    # Random downsampling
    indices = np.random.choice(len(points), max_points, replace=False)
    downsampled = points[indices]
    print(f"Downsampled from {len(points):,} to {len(downsampled):,} points")
    return downsampled


def visualize_with_matplotlib(points, title="PCD Visualization", save_path=None):
    """Create 3D visualization using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not found. Installing...")
        os.system("pip install matplotlib")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color points by height (Z value) for better visualization
    colors = z
    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=0.5, alpha=0.6)
    
    # Add colorbar
    plt.colorbar(scatter, label='Height (Z)', shrink=0.5)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    # Print statistics
    print(f"\nPoint Cloud Statistics:")
    print(f"Points: {len(points):,}")
    print(f"X range: {x.min():.2f} to {x.max():.2f}")
    print(f"Y range: {y.min():.2f} to {y.max():.2f}")
    print(f"Z range: {z.min():.2f} to {z.max():.2f}")
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D visualization saved to: {save_path}")
    else:
        print("\nVisualization ready! Close the plot window to continue.")
        plt.show()
    
    plt.close()


def create_2d_views(points, title="PCD 2D Views", save_path=None):
    """Create 2D views (top, side, front) for better understanding."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not found. Installing...")
        os.system("pip install matplotlib")
        import matplotlib.pyplot as plt
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{title} - 2D Projections")
    
    # Top view (X-Y)
    axes[0, 0].scatter(x, y, c=z, cmap='viridis', s=0.1, alpha=0.6)
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title('Top View (X-Y)')
    axes[0, 0].set_aspect('equal')
    
    # Side view (X-Z)
    axes[0, 1].scatter(x, z, c=y, cmap='plasma', s=0.1, alpha=0.6)
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Z (m)')
    axes[0, 1].set_title('Side View (X-Z)')
    
    # Front view (Y-Z)
    axes[1, 0].scatter(y, z, c=x, cmap='coolwarm', s=0.1, alpha=0.6)
    axes[1, 0].set_xlabel('Y (m)')
    axes[1, 0].set_ylabel('Z (m)')
    axes[1, 0].set_title('Front View (Y-Z)')
    
    # Height histogram
    axes[1, 1].hist(z, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_xlabel('Height (Z) (m)')
    axes[1, 1].set_ylabel('Point Count')
    axes[1, 1].set_title('Height Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"2D views saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize PCD using matplotlib (WSL-friendly)")
    parser.add_argument(
        "pcd_path",
        nargs="?",
        default=DEFAULT_PCD,
        help="Path to .pcd file (defaults to Site1 sample)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=100000,
        help="Maximum points to display (default: 100000)",
    )
    parser.add_argument(
        "--2d-views",
        action="store_true",
        dest="_2d_views",
        help="Show 2D projections instead of 3D plot",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Show both 3D and 2D visualizations",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plots as image files instead of showing them",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        # Load points
        points = parse_pcd_file(args.pcd_path)
        
        # Downsample if needed
        if len(points) > args.max_points:
            points = downsample_points(points, args.max_points)
        
        title = f"PCD: {os.path.basename(args.pcd_path)}"
        
        # Prepare save paths if needed
        save_3d = None
        save_2d = None
        if args.save:
            base_name = os.path.splitext(os.path.basename(args.pcd_path))[0]
            output_dir = os.path.dirname(args.pcd_path)
            save_3d = os.path.join(output_dir, f"{base_name}_3d_view.png")
            save_2d = os.path.join(output_dir, f"{base_name}_2d_views.png")
        
        # Show visualizations
        if args.both:
            visualize_with_matplotlib(points, title, save_3d)
            create_2d_views(points, title, save_2d)
        elif args._2d_views:
            create_2d_views(points, title, save_2d)
        else:
            visualize_with_matplotlib(points, title, save_3d)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
