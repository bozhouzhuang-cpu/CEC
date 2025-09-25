import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import geoCosiCorr3D
import geoCosiCorr3D.georoutines.geo_utils as geo_utils
import geoCosiCorr3D.georoutines.geoplt_misc as geo_plt
import geoCosiCorr3D.time_series.geoICA as geoICA
import matplotlib.gridspec as gridspec

print(geoCosiCorr3D.__version__)

def stack_correlation_bands(correlation_files, output_dir="dataset"):
    """
    Stack correlation results from multiple TIFs into EW and NS time series
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    ew_arrays = []
    ns_arrays = []
    
    # Read each correlation file and extract EW/NS bands
    for i, corr_file in enumerate(correlation_files):
        print(f"Processing {corr_file}...")
        
        # Load correlation result
        raster_info = geo_utils.cRasterInfo(corr_file)
        
        if raster_info.band_number < 2:
            print(f"Warning: {corr_file} has fewer than 2 bands, skipping...")
            continue
            
        # Extract EW (band 1) and NS (band 2) 
        ew_band = raster_info.raster_array[0]  # First band = EW displacement
        ns_band = raster_info.raster_array[1]  # Second band = NS displacement
        
        ew_arrays.append(ew_band)
        ns_arrays.append(ns_band)
        
        print(f"  EW shape: {ew_band.shape}, NS shape: {ns_band.shape}")
    
    if not ew_arrays:
        print("No valid correlation files found!")
        return None, None
    
    # Find common dimensions (intersection of all arrays)
    shapes = [arr.shape for arr in ew_arrays]
    min_height = min(shape[0] for shape in shapes)
    min_width = min(shape[1] for shape in shapes)
    
    print(f"Array shapes: {shapes}")
    print(f"Using common dimensions: ({min_height}, {min_width})")
    
    # Crop all arrays to common size
    ew_cropped = [arr[:min_height, :min_width] for arr in ew_arrays]
    ns_cropped = [arr[:min_height, :min_width] for arr in ns_arrays]
    
    # Stack arrays (each becomes a band in the output)
    ew_stack = np.stack(ew_cropped, axis=0)  # Shape: (n_times, height, width)
    ns_stack = np.stack(ns_cropped, axis=0)
    
    print(f"Stacked EW shape: {ew_stack.shape}")
    print(f"Stacked NS shape: {ns_stack.shape}")
    
    # Use georeferencing from the first file
    ref_raster = geo_utils.cRasterInfo(correlation_files[0])
    
    # Output filenames
    ew_output = os.path.join(output_dir, "EW_WV_Spot_MB_3DDA.tif")
    ns_output = os.path.join(output_dir, "NS_WV_Spot_MB_3DDA.tif")
    
    # Write EW time series
    geo_utils.cRasterInfo.write_raster(
        output_raster_path=ew_output,
        array_list=[ew_stack[i] for i in range(ew_stack.shape[0])],
        geo_transform=ref_raster.geo_transform,
        epsg_code=ref_raster.epsg_code,
        dtype="float32",
        descriptions=[f"EW_Time_{i+1}" for i in range(ew_stack.shape[0])]
    )
    
    # Write NS time series  
    geo_utils.cRasterInfo.write_raster(
        output_raster_path=ns_output,
        array_list=[ns_stack[i] for i in range(ns_stack.shape[0])],
        geo_transform=ref_raster.geo_transform,
        epsg_code=ref_raster.epsg_code,
        dtype="float32",
        descriptions=[f"NS_Time_{i+1}" for i in range(ns_stack.shape[0])]
    )
    
    print(f"Created EW time series: {ew_output}")
    print(f"Created NS time series: {ns_output}")
    
    return ew_output, ns_output

def plt_4_bands(raster_info: geo_utils.cRasterInfo,
               o_folder=None,
               arg=None,
               off=2.5):
    """
    Plot 4 bands side by side
    """
    fig = plt.figure(figsize=(19.20, 10.80))
    spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)
    axList = []
    imList = []

    for index, array_ in enumerate(raster_info.raster_array):
        ax = fig.add_subplot(spec[0, index])
        axList.append(ax)
        array_ = np.ma.masked_where(array_ < -100, array_)
        array_ = np.ma.masked_where(array_ > 100, array_)

        im = ax.imshow(array_, cmap="RdYlBu", vmin=-off, vmax=off)
        imList.append(im)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title(f"Band {index+1}")

    geo_plt.ColorBar_(ax=axList[-1], mapobj=imList[-1], size=12, vmin=-off, vmax=off,
                      cmap='RdBlu')
    
    if o_folder is not None:
        svgPath = os.path.join(o_folder, arg + ".svg")
        fig.savefig(svgPath, transparent=False, format="svg", dpi=600)
        print(f"Saved plot: {svgPath}")
    else:
        plt.show()

def run_ica_analysis(ew_fn, ns_fn, output_dir="results"):
    """
    Run ICA analysis on EW and NS displacement time series
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("RUNNING ICA ANALYSIS ON EAST-WEST DISPLACEMENTS")
    print("=" * 50)
    
    # EW ICA Analysis
    ew_ica = geoICA.geoICA(ew_fn, 4)
    ew_ica()
    
    print("Plotting original EW displacement time series:")
    plt_4_bands(raster_info=ew_ica.raster_info, arg="EW_orig_Disp", o_folder=output_dir, off=2.5)
    
    print("Plotting EW ICA components:")
    off = 0.5
    for i, mean_rec_fn in enumerate(ew_ica.mean_reconstructed_fns):
        fig, ax1 = plt.subplots(1, 1, constrained_layout=True)
        mapobj = ax1.imshow(np.ma.masked_invalid(geo_utils.cRasterInfo(mean_rec_fn).raster_array[0]),
                            vmin=-off, vmax=off, cmap="RdYlBu")
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        geo_plt.ColorBar_(ax=ax1, mapobj=mapobj, size=12, vmin=-off, vmax=off, cmap='RdBlu')
        ax1.set_title(f"EW ICA Component {i+1}: {Path(mean_rec_fn).stem}")
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"EW_ICA_Component_{i+1}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    print("=" * 50)
    print("RUNNING ICA ANALYSIS ON NORTH-SOUTH DISPLACEMENTS")
    print("=" * 50)
    
    # NS ICA Analysis
    ns_ica = geoICA.geoICA(ns_fn, 4)
    ns_ica()
    
    print("Plotting original NS displacement time series:")
    plt_4_bands(raster_info=ns_ica.raster_info, arg="NS_orig_Disp", o_folder=output_dir, off=2.5)
    
    print("Plotting NS ICA components:")
    for i, mean_rec_fn in enumerate(ns_ica.mean_reconstructed_fns):
        fig, ax1 = plt.subplots(1, 1, constrained_layout=True)
        mapobj = ax1.imshow(np.ma.masked_invalid(geo_utils.cRasterInfo(mean_rec_fn).raster_array[0]),
                            vmin=-off, vmax=off, cmap="RdYlBu")
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        geo_plt.ColorBar_(ax=ax1, mapobj=mapobj, size=12, vmin=-off, vmax=off, cmap='RdBlu')
        ax1.set_title(f"NS ICA Component {i+1}: {Path(mean_rec_fn).stem}")
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"NS_ICA_Component_{i+1}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    return ew_ica, ns_ica

if __name__ == "__main__":
    # Your 4 correlation TIF files
    correlation_files = [
        "/home/bozhouzh/Geospatial-COSICorr3D/tests/Gilroy/GIL_1_WV2_16OCT11191432-MS-050358698060_01_P001_VS_GIL_2_WV2_18JUN08192654-MS-050358698070_01_P001_spatial_wz_128_step_8.tif",
        "/home/bozhouzh/Geospatial-COSICorr3D/tests/Gilroy/GIL_1_WV2_16OCT11191432-MS-050358698060_01_P001_VS_GIL_3_WV2_19AUG04190453-MS-050358698080_01_P001_spatial_wz_128_step_8.tif",
        "/home/bozhouzh/Geospatial-COSICorr3D/tests/Gilroy/GIL_1_WV2_16OCT11191432-MS-050358698060_01_P001_VS_GIL_5_WV2_20MAY23191255-MS-050358905010_01_P001_spatial_wz_128_step_8.tif",
        "/home/bozhouzh/Geospatial-COSICorr3D/tests/Gilroy/GIL_1_WV2_16OCT11191432-MS-050358698060_01_P001_VS_GIL_6_WV2_21JUN04191908-MS-050358698100_01_P001_spatial_wz_128_step_8.tif"
    ]
    
    print("Step 1: Stacking correlation results...")
    ew_file, ns_file = stack_correlation_bands(correlation_files)
    
    if ew_file and ns_file:
        print("\nStep 2: Running ICA analysis...")
        ew_ica, ns_ica = run_ica_analysis(ew_file, ns_file)
        
        print("\nAnalysis complete! Check the 'results' folder for saved plots.")
        print(f"EW time series: {ew_file}")
        print(f"NS time series: {ns_file}")
    else:
        print("Failed to create time series files!")