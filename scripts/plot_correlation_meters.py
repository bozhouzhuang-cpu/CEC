#!/usr/bin/env python3
"""
Plot correlation results with axes in meters starting from zero.

This script reads a correlation output TIF file and creates a plot where:
- Axes are in meters (starting from 0)
- Displacement values are shown in pixels or meters
- Creates a high-quality publication-ready figure

Author: Auto-generated
Usage:
    python scripts/plot_correlation_meters.py <correlation_tif_file> [options]
    
Example:
    python scripts/plot_correlation_meters.py results/correlation.tif --vmin -3 --vmax 3 --dpi 300
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    import rasterio
except ImportError:
    print("Error: rasterio is required. Install it with: pip install rasterio")
    sys.exit(1)


def plot_correlation_in_meters(tif_path, output_path=None, vmin=-1, vmax=1, 
                                dpi=300, cmap='RdBu', show_displacement_in_meters=False):
    """
    Create a correlation plot with axes in meters and starting from zero.
    
    Parameters
    ----------
    tif_path : str
        Path to the correlation TIF file
    output_path : str, optional
        Path to save the output PNG. If None, saves next to the input file
    vmin : float
        Minimum value for colorbar
    vmax : float
        Maximum value for colorbar
    dpi : int
        Resolution of the output image
    cmap : str
        Matplotlib colormap name
    show_displacement_in_meters : bool
        If True, multiply displacement by pixel size to show in meters
    """
    
    # Read the correlation result
    print(f"Reading correlation file: {tif_path}")
    
    with rasterio.open(tif_path) as src:
        # Read displacement bands
        ew = src.read(1)  # East-West displacement
        ns = src.read(2)  # North-South displacement
        snr = src.read(3) if src.count >= 3 else None  # SNR if available
        bounds = src.bounds  # BoundingBox(left, bottom, right, top)
    
        # Lower-right corner (xmax, ymin)
        lower_right_x = bounds.right
        lower_right_y = bounds.bottom

        print(f"Lower-right corner UTM coordinates: X={lower_right_x}, Y={lower_right_y}")

        # Get pixel size in meters
        transform = src.transform
        pixel_size_x = abs(transform.a)  # meters/pixel
        pixel_size_y = abs(transform.e)  # meters/pixel
        
        print(f"Raster info:")
        print(f"  - Shape: {ew.shape}")
        print(f"  - Pixel size X: {pixel_size_x:.4f} m/pixel")
        print(f"  - Pixel size Y: {pixel_size_y:.4f} m/pixel")
        print(f"  - Transform: {transform}")
        
        # Create extent in meters starting from 0
        rows, cols = ew.shape
        extent_meters = [0, cols * pixel_size_x, 0, rows * pixel_size_y]
        
        print(f"  - Extent in meters: X=[0, {extent_meters[1]:.2f}], Y=[0, {extent_meters[3]:.2f}]")
        
        # Optionally convert displacement from pixels to meters
        if show_displacement_in_meters:
            ew_display = ew * pixel_size_x
            ns_display = ns * pixel_size_y
            disp_unit = 'm'
            print(f"  - Displaying displacement in meters")
        else:
            ew_display = ew
            ns_display = ns
            disp_unit = 'px'
            print(f"  - Displaying displacement in pixels")
        
        # Mask NaN values
        ew_display = np.ma.masked_invalid(ew_display)
        ns_display = np.ma.masked_invalid(ns_display)
        
        # Calculate statistics
        ew_mean = np.nanmean(ew_display)
        ew_std = np.nanstd(ew_display)
        ns_mean = np.nanmean(ns_display)
        ns_std = np.nanstd(ns_display)
        
        print(f"\nDisplacement statistics ({disp_unit}):")
        print(f"  - E-W: mean={ew_mean:.4f}, std={ew_std:.4f}")
        print(f"  - N-S: mean={ns_mean:.4f}, std={ns_std:.4f}")
    
    # Create the plot
    fig, axs = plt.subplots(1, 2, figsize=(5, 5))
    
    # Plot East-West displacement
    im1 = axs[0].imshow(ew_display, extent=extent_meters, cmap=cmap, 
                       vmin=vmin, vmax=vmax, origin='upper', interpolation='none')
    axs[0].set_xlabel('Relative Easting (m)', fontsize=12)
    axs[0].set_ylabel('Relative Northing (m)', fontsize=12)
    
    # Plot North-South displacement
    im2 = axs[1].imshow(ns_display, extent=extent_meters, cmap=cmap, 
                       vmin=vmin, vmax=vmax, origin='upper', interpolation='none')
    axs[1].set_xlabel('Relative Easting (m)', fontsize=12)
    # No y-axis label for the second plot
    
    # Add colorbars with titles
    BORDER_PAD = 1.2  # Increased to move colorbar more upward
    for ax, title_, im in zip(axs, [f"E-W (m)", f"N-S (m)"], [im1, im2]):
        axins = inset_axes(ax,
                          width="60%",
                          height="3%",
                          loc='upper center',
                          borderpad=-BORDER_PAD
                          )
        cbar = fig.colorbar(im, cax=axins, orientation="horizontal", extend='both')
        cbar.ax.set_title(title_, fontsize=12)
        cbar.ax.tick_params(labelsize=12, direction='out', top=True, bottom=False, labeltop=True, labelbottom=False)
    
    # Format axes
    for ax in axs:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(direction='in', which='minor', top=True, right=True)
        ax.tick_params(direction='in', which='major', top=True, right=True)
    
    # Remove y-tick labels from the right plot
    axs[1].set_yticklabels([])
    
    plt.tight_layout(pad=BORDER_PAD)
    
    # Determine output path
    if output_path is None:
        base_path = Path(tif_path).with_suffix('')
        output_path = f"{base_path}_meters.png"
    
    # Save the figure
    print(f"\nSaving plot to: {output_path}")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved successfully!")
    
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Plot correlation results with axes in meters starting from zero.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/plot_correlation_meters.py correlation.tif
  
  # With custom color range
  python scripts/plot_correlation_meters.py correlation.tif --vmin -3 --vmax 3
  
  # Show displacement in meters instead of pixels
  python scripts/plot_correlation_meters.py correlation.tif --displacement_in_meters
  
  # Custom output path and high DPI
  python scripts/plot_correlation_meters.py correlation.tif -o output.png --dpi 600
        """
    )
    
    parser.add_argument('tif_file', help='Path to the correlation TIF file')
    parser.add_argument('-o', '--output', help='Output PNG file path (default: same as input with _meters.png suffix)')
    parser.add_argument('--vmin', type=float, default=-1, help='Minimum value for colorbar (default: -1)')
    parser.add_argument('--vmax', type=float, default=1, help='Maximum value for colorbar (default: 1)')
    parser.add_argument('--dpi', type=int, default=300, help='Resolution of output image (default: 300)')
    parser.add_argument('--cmap', default='RdBu', help='Matplotlib colormap (default: RdBu)')
    parser.add_argument('--displacement_in_meters', action='store_true',
                       help='Show displacement in meters instead of pixels')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.tif_file):
        print(f"Error: File not found: {args.tif_file}")
        sys.exit(1)
    
    # Create the plot
    try:
        output_path = plot_correlation_in_meters(
            tif_path=args.tif_file,
            output_path=args.output,
            vmin=args.vmin,
            vmax=args.vmax,
            dpi=args.dpi,
            cmap=args.cmap,
            show_displacement_in_meters=args.displacement_in_meters
        )
        print(f"\n✓ Success! Plot saved to: {output_path}")
    except Exception as e:
        print(f"\n✗ Error creating plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

