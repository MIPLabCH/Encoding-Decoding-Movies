
#mask functions



import numpy as np
import matplotlib.pyplot as plt
import colorsys
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

def load_and_reshape_data(file_path):
    regions_2d = np.load(file_path, mmap_mode='r')
    regions_3d = regions_2d.reshape(91, 109, 91)
    return regions_3d

def get_unique_regions(data_3d):
    unique_regions = np.unique(data_3d)
    # Filter out zero (background)
    unique_regions = unique_regions[unique_regions > 0]
    return unique_regions

def analyze_regions_by_slice(data_3d, axis=1):
    """
    Determine which regions appear in each slice along the specified axis
    
    Parameters:
    data_3d: 3D numpy array of region IDs
    axis: Axis along which to take slices (0=sagittal, 1=coronal, 2=axial)
    
    Returns:
    List where each element contains the regions in that slice
    """
    num_slices = data_3d.shape[axis]
    regions_by_slice = []
    
    for i in range(num_slices):
        if axis == 0:
            slice_data = data_3d[i, :, :]
        elif axis == 1:
            slice_data = data_3d[:, i, :]
        else:  # axis == 2
            slice_data = data_3d[:, :, i]
        
        # Get unique regions in this slice (excluding 0/background)
        unique_in_slice = np.unique(slice_data)
        unique_in_slice = unique_in_slice[unique_in_slice > 0]
        
        regions_by_slice.append(unique_in_slice)
    
    return regions_by_slice

def calculate_region_extents(data_3d, all_regions=None):
    """
    Calculate the posterior-most and anterior-most slice for each region
    
    Parameters:
    data_3d: 3D numpy array of region IDs
    all_regions: Optional list of region IDs to analyze (if None, all regions are analyzed)
    
    Returns:
    Dictionary mapping region IDs to (posterior_slice, anterior_slice) indices
    """
    # If not provided, get all unique regions
    if all_regions is None:
        all_regions = get_unique_regions(data_3d)
    
    # Get regions by coronal slice
    coronal_regions = analyze_regions_by_slice(data_3d, axis=1)
    
    # Find posterior and anterior extents for each region
    region_extent = {}
    
    for region in all_regions:
        posterior_slice = None
        anterior_slice = None
        
        # Check each slice for this region
        for slice_idx in range(len(coronal_regions)):
            if region in coronal_regions[slice_idx]:
                # If first occurrence, set posterior slice
                if posterior_slice is None:
                    posterior_slice = slice_idx
                # Always update anterior slice to the latest occurrence
                anterior_slice = slice_idx
        
        if posterior_slice and anterior_slice:
            region_extent[region] = (posterior_slice, anterior_slice)
    
    return region_extent

def get_regions_passing_through_slice(region_extent, slice_idx):
    """
    Find all regions that pass through a given slice
    
    Parameters:
    region_extent: Dictionary mapping region IDs to (posterior, anterior) slice indices
    slice_idx: Slice index to check
    
    Returns:
    List of region IDs that pass through the specified slice
    """
    passing_regions = []
    
    for region, (posterior, anterior) in region_extent.items():
        if posterior <= slice_idx <= anterior:
            passing_regions.append(region)
    
    return passing_regions  #   <---

def highlight_regions_in_slice(data_3d, slice_idx, axis=1, title=None, figsize=(12, 10)):
    """
    Visualize all brain regions in axial view, with regions passing through
    a specified slice highlighted in color, and others in grayscale.
    
    Parameters:
    data_3d: 3D numpy array of region IDs
    slice_idx: The slice index to analyze
    axis: The axis for the slice (0=sagittal, 1=coronal, 2=axial)
    title: Optional custom title
    figsize: Figure size as (width, height) tuple
    
    Returns:
    numpy array of region IDs that pass through the specified slice
    """
    # Calculate region extents
    region_extent = calculate_region_extents(data_3d)
    
    # Get regions passing through this slice
    regions_in_slice = get_regions_passing_through_slice(region_extent, slice_idx)
    
    # Get slice name for title
    slice_name = ["Sagittal", "Coronal", "Axial"][axis]
    
    # Axial projection of the entire brain
    whole_brain_axial = np.max(data_3d, axis=2)
    
    # RGB
    colored_image = np.zeros((*whole_brain_axial.shape, 3))
    
    # Use gray for regions not in the slice
    other_regions_mask = (whole_brain_axial > 0)
    for i in range(3):  # RGB channels
        colored_image[:,:,i] = np.where(other_regions_mask, 0.3, 0)  # Dark gray
    
    
    
    # Get a colormap with enough colors for all regions in the slice
    colormap = cm.get_cmap('hsv', len(regions_in_slice) + 1)
    
    # Assign different colors to regions in the slice
    for i, region in enumerate(regions_in_slice):
        # Create a mask for this region in axial projection
        region_mask = np.max(data_3d == region, axis=2)
        color = colormap(i)[:3]  # RGB values of the color
        
        # Apply this color to the region
        for j in range(3):  # RGB channels
            colored_image[:,:,j] = np.where(region_mask, color[j], colored_image[:,:,j])
    
    # Create the visualization
    plt.figure(figsize=figsize)
    
    # Use custom title if provided, otherwise use default
    if title:
        plt.suptitle(title, fontsize=16)
    else:
        plt.suptitle(f"Brain Regions in Axial View\nRegions passing through {slice_name} slice {slice_idx} are highlighted", fontsize=16)
    
    # Main plot - axial view with highlighted regions
    plt.imshow(colored_image, origin='lower')
    plt.axis('off')
    
    # Add a dotted line to show the slice location
    if axis == 0:  # Sagittal slice (vertical line in axial view)
        plt.axvline(x=slice_idx, color='white', linestyle='--', linewidth=2)
    elif axis == 1:  # Coronal slice (horizontal line in axial view)
        plt.axvline(x=slice_idx, color='white', linestyle='--', linewidth=2)
    # No line needed for axial slice since we're already in axial view
    
    # Create a custom legend
    handles = []
    labels = []
    
    # Add a gray box for all other regions
    handles.append(plt.Rectangle((0,0), 1, 1, color='gray'))
    labels.append("Regions not in slice")
    
    # Add colored boxes for regions in the slice
    # Limit to max 10 regions in legend to avoid overcrowding
    legend_limit = min(10, len(regions_in_slice))
    for i in range(legend_limit):
        handles.append(plt.Rectangle((0,0), 1, 1, color=colormap(i)[:3]))
        labels.append(f"Region {regions_in_slice[i]}")
    
    if len(regions_in_slice) > legend_limit:
        handles.append(plt.Rectangle((0,0), 1, 1, color='white'))
        labels.append(f"+ {len(regions_in_slice) - legend_limit} more regions")
    
    plt.legend(handles, labels, loc='upper right', title="Region Legend", fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Found {len(regions_in_slice)} regions passing through {slice_name} slice {slice_idx}")
    
    return regions_in_slice

def get_sorted_regions_by_posterior_extent(data_3d, limit=None):
    """
    Get regions sorted by their posterior-most slice (from back to front)
    
    Parameters:
    data_3d: 3D numpy array of region IDs
    limit: Optional limit to return only the N most posterior regions
    
    Returns:
    List of (region_id, (posterior_slice, anterior_slice)) tuples
    """
    region_extent = calculate_region_extents(data_3d)
    
    # Sort regions by their posterior-most slice (from back to front)
    sorted_regions = sorted(region_extent.items(), key=lambda x: x[1][0])
    
    if limit:
        return sorted_regions[:limit]
    return sorted_regions

def plot_multiple_posterior_regions(data_3d, num_regions=5, figsize=(15, 10)):
    """
    Plot the most posterior regions in the brain
    
    Parameters:
    data_3d: 3D numpy array of region IDs
    num_regions: Number of most posterior regions to highlight
    figsize: Figure size as (width, height) tuple
    
    Returns:
    List of the most posterior region IDs
    """
    # Get the most posterior regions
    sorted_regions = get_sorted_regions_by_posterior_extent(data_3d, limit=num_regions)
    most_posterior_regions = [region for region, _ in sorted_regions]
    
    # Print information about these regions
    print(f"{num_regions} most posterior regions:")
    for i, (region, (posterior, anterior)) in enumerate(sorted_regions):
        print(f"{i+1}. Region {region} - spans from coronal slice {posterior} to {anterior}")
    
    # Create subplots for individual regions
    plt.figure(figsize=figsize)
    plt.suptitle(f"{num_regions} Most Posterior Regions - Axial View", fontsize=16)
    
    for i, region in enumerate(most_posterior_regions):
        # Create mask for this region
        region_mask = (data_3d == region)
        
        # Create a binary version for this region
        region_data = np.zeros_like(data_3d)
        region_data[region_mask] = 1
        
        # Project along z-axis for axial view
        axial_projection = np.max(region_data, axis=2)
        
        # Subplot
        plt.subplot(2, 3, i+1)
        plt.imshow(axial_projection, cmap='hot', origin='lower')
        plt.title(f'Region {region}', fontsize=14)
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    plt.show()
    
    # Create a combined view with all regions in different colors
    plt.figure(figsize=(8, 8))
    plt.title(f"All {num_regions} Most Posterior Regions - Axial View", fontsize=16)
    
    # Generate distinct colors
    colors = []
    for i in range(len(most_posterior_regions)):
        hue = i/len(most_posterior_regions)
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append(rgb)
    
    # Create colormap with black background
    colors.insert(0, (0,0,0))
    custom_cmap = ListedColormap(colors)
    
    # Create combined data
    combined_data = np.zeros_like(data_3d)
    for i, region in enumerate(most_posterior_regions):
        # Assign unique values to each region (1, 2, 3, etc.)
        combined_data[data_3d == region] = i+1
    
    # Project along z-axis
    combined_projection = np.max(combined_data, axis=2)
    
    # Plot
    plt.imshow(combined_projection, cmap=custom_cmap, origin='lower')
    plt.axis('off')
    
    # Add legend
    handles = []
    labels = []
    for i, region in enumerate(most_posterior_regions):
        handles.append(plt.Rectangle((0,0), 1, 1, color=colors[i+1]))
        labels.append(f"Region {region}")
    
    plt.legend(handles, labels, loc='best', title="Region Legend")
    plt.tight_layout()
    plt.show()
    
    return most_posterior_regions


def turn_off_regions(input_fmri, regions, mode='off', amplify_factor=4):
    """
    Modifies brain activity in specific regions.
    
    Parameters:
    -----------
    input_fmri : numpy.ndarray
        Input fMRI data with shape [N, 4609], where N is number of TRs
    regions : list
        List of region IDs to be modified, e.g., [1,4,6,7]
    mode : str, optional
        'off' (default): Sets the regions to 0
        'amplify': Multiplies the regions by amplify_factor
    amplify_factor : float, optional
        Factor to multiply region values by when mode='amplify' (default: 4)
        
    Returns:
    --------
    numpy.ndarray
        Modified fMRI data
    """
    region_ids = np.load('region_ids_4609+.npy')
    mask4609 = np.load('mask_schaefer1000_4609.npy')

    # Flatten the mask
    flat_mask = mask4609.flatten()

    # Check if shapes match
    if flat_mask.shape[0] == region_ids.shape[0]:
        # Apply the mask - just multiply the arrays (ones keep values, zeros eliminate)
        masked_values = region_ids * flat_mask
        
        # Extract only the non-zero values (where mask was 1)
        non_zero_indices = np.nonzero(flat_mask)[0]
        result = masked_values[non_zero_indices]
        print(result)
        region_ids = result

    # Find indices of specified regions
    indices = []
    for region_id in regions:
        indices.extend(np.where(region_ids == region_id)[0].tolist())

    # Create a copy of the input
    fmri2 = input_fmri.copy()
    
    #tag temporary
    # Apply the desired operation based on mode
    if mode.lower() == 'off':

        zeros_before = (fmri2 == 0).sum()
        total_values = fmri2.size
        percent_zeros_before = (zeros_before / total_values) * 100
        print(f"Before turning off - Zeros: {zeros_before}/{total_values} ({percent_zeros_before:.2f}%)")

        fmri2[:, indices] = 0

        zeros_after = (fmri2 == 0).sum()
        percent_zeros_after = (zeros_after / total_values) * 100
        new_zeros = zeros_after - zeros_before
        percent_increase = (new_zeros / total_values) * 100
        print(f"After turning off - Zeros: {zeros_after}/{total_values} ({percent_zeros_after:.2f}%)")
        print(f"Values turned off: {new_zeros}/{total_values} ({percent_increase:.2f}%)")
    
    
#    elif mode.lower() == 'amplify':
#        fmri2[:, indices] *= amplify_factor
    else:
        fmri2[:, indices] *= amplify_factor
#        print(f"Warning: Unknown mode '{mode}'. Using default 'off' mode.")
#        fmri2[:, indices] = 0
        
    return fmri2



# Load data
#regions_3d = load_and_reshape_data('region_ids_4609+.npy')
    
    # Get 5 most posterior regions
#    most_posterior_regions = plot_multiple_posterior_regions(regions_3d, num_regions=5)
    
# Highlight regions passing through coronal slice 30



#regions_3d = load_and_reshape_data('region_ids_4609+.npy')
#highlighted_regions = highlight_regions_in_slice(regions_3d, 30, axis=1)
#test_new_decoder()
#test_new_decoder(real=True, model_name='decoder_4609_350', regions=highlighted_regions, add_name='_mask_slice_30')

#test_new_decoder(real=True, model_name='decoder_4609_1650', test_on_train=False, test_input = testset['fMRIs'], test_label = testset['videos'], add_name='', regions = []):

