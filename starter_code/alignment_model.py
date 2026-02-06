"""Implements the alignment algorithm."""

import torch
import torchvision
from metrics import ncc, mse, ssim
from helpers import custom_shifts


class AlignmentModel:
  def __init__(self, image_name, metric='ssim', padding='circular'):
    # Image name
    self.image_name = image_name
    # Metric to use for alignment
    self.metric = metric
    # Padding mode for custom_shifts
    self.padding = padding

  def save(self, output_name):
    torchvision.utils.save_image(self.rgb, output_name)

  def align(self):
    """Aligns the image using the metric specified in the constructor.
       
       Uses image pyramid (coarse-to-fine) approach for efficient alignment.
       Green channel is used as reference, red and blue are aligned to it.
       
       Finally, outputs the rgb image in self.rgb with proper border cropping
       to remove ghosting artifacts.
    """
    # Load the image
    self.img = self._load_image()
    
    # Step 1: Divide the image into three color channels
    b_channel, g_channel, r_channel = self._crop_and_divide_image()
    
    # Step 2: Align channels using image pyramid approach
    # Green channel is the reference (middle channel, often sharpest)
    # Base search window of 15 pixels for coarsest level
    base_delta = 15
    
    # Align red channel to green channel
    r_shift = self._align_pairs(g_channel, r_channel, base_delta)
    
    # Align blue channel to green channel  
    b_shift = self._align_pairs(g_channel, b_channel, base_delta)
    
    # Step 3: Apply the shifts to align all channels
    r_aligned = custom_shifts(r_channel, r_shift, dims=(0, 1), padding=self.padding)
    b_aligned = custom_shifts(b_channel, b_shift, dims=(0, 1), padding=self.padding)
    g_aligned = g_channel  # Green is the reference (no shift)
    
    # Step 4: Post-process cropping to remove ghosting artifacts
    # Crop the borders where shifted channels don't overlap perfectly
    # This removes the "halo" effect at edges
    r_cropped, g_cropped, b_cropped = self._crop_to_overlap(
        r_aligned, g_aligned, b_aligned, r_shift, b_shift
    )
    
    # Step 5: Stack the aligned channels into an RGB image
    # Shape: (3, H, W) for torchvision.utils.save_image
    self.rgb = torch.stack([r_cropped, g_cropped, b_cropped], dim=0)
    
    # Store displacement information for reporting
    self.r_displacement = r_shift
    self.b_displacement = b_shift
    self.g_displacement = (0, 0)  # Green is reference
    
    # Print displacement vectors for the report
    print(f"Image: {self.image_name}")
    print(f"Metric: {self.metric}")
    print(f"Red channel displacement (y, x): {r_shift}")
    print(f"Green channel displacement (y, x): {self.g_displacement}")
    print(f"Blue channel displacement (y, x): {b_shift}")
    print()
  
  def _crop_to_overlap(self, r_channel, g_channel, b_channel, r_shift, b_shift):
    """
    Crop all channels to their common overlap region after alignment.
    
    This removes ghosting artifacts at borders where shifted channels
    don't perfectly overlap. Critical for clean final output.
    
    Args:
        r_channel, g_channel, b_channel: Aligned channel images
        r_shift: (shift_y, shift_x) for red channel
        b_shift: (shift_y, shift_x) for blue channel
    
    Returns: Tuple of cropped (r, g, b) channels
    """
    height, width = g_channel.shape
    
    # Calculate maximum shifts in each direction
    # We need to crop by the maximum absolute shift to ensure overlap
    max_shift_y = max(abs(r_shift[0]), abs(b_shift[0]))
    max_shift_x = max(abs(r_shift[1]), abs(b_shift[1]))
    
    # Add a bit of extra margin to be safe (5% of max shift, min 2 pixels)
    margin_y = max(2, int(max_shift_y * 0.05))
    margin_x = max(2, int(max_shift_x * 0.05))
    
    # CHANGE 2: Add the stored border crop amounts from initial detection
    crop_y = max_shift_y + margin_y + self.border_crop_y
    crop_x = max_shift_x + margin_x + self.border_crop_x
    
    # Ensure we don't crop too much
    if crop_y * 2 >= height:
        crop_y = height // 4  # Fallback to 25% crop
    if crop_x * 2 >= width:
        crop_x = width // 4
    
    # Crop all channels identically to maintain alignment
    r_cropped = r_channel[crop_y:-crop_y, crop_x:-crop_x]
    g_cropped = g_channel[crop_y:-crop_y, crop_x:-crop_x]
    b_cropped = b_channel[crop_y:-crop_y, crop_x:-crop_x]
    
    return r_cropped, g_cropped, b_cropped


  def save(self, output_name):
    torchvision.utils.save_image(self.rgb, output_name)

  def _load_image(self):
    """Load the image from the image_name path,
       typecast it to float, and normalize it.

       Returns: torch.Tensor of shape (H, W)
    """
    from PIL import Image
    import numpy as np
    
    # Load image using PIL
    img = Image.open(self.image_name)
    
    # Convert to grayscale if not already (images should already be grayscale)
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Convert to torch tensor and typecast to float
    ret = torch.from_numpy(img_array).float()
    
    # Normalize to [0, 1] range
    ret = ret / 255.0
    
    return ret

  def _crop_and_divide_image(self):
    """Crop the image boundary and divide the image into three parts, padded to the same size.

       This implementation removes borders more aggressively to eliminate artifacts
       and jagged edges common in Prokudin-Gorskii glass plate images.

       Returns: B, G, R torch.Tensor of shape (roughly H//3, W)
    """
    height, width = self.img.shape
    
    # First pass: Remove obvious dark borders using edge detection
    # Calculate average intensity per row and column
    row_avg = torch.mean(self.img, dim=1)
    col_avg = torch.mean(self.img, dim=0)
    
    # Use threshold to detect very dark border regions
    # More aggressive than before - use 15th percentile
    row_threshold = torch.quantile(row_avg, 0.15)
    col_threshold = torch.quantile(col_avg, 0.15)
    
    # Find non-border rows and columns
    non_border_rows = torch.where(row_avg > row_threshold)[0]
    non_border_cols = torch.where(col_avg > col_threshold)[0]
    
    if len(non_border_rows) > 0:
        top_crop = non_border_rows[0].item()
        bottom_crop = non_border_rows[-1].item() + 1
    else:
        # Fallback: crop 3% from top and bottom
        top_crop = int(height * 0.03)
        bottom_crop = int(height * 0.97)
    
    if len(non_border_cols) > 0:
        left_crop = non_border_cols[0].item()
        right_crop = non_border_cols[-1].item() + 1
    else:
        # Fallback: crop 3% from left and right
        left_crop = int(width * 0.03)
        right_crop = int(width * 0.97)
    
    # CHANGE 1: Store border crop amounts for use in final output cropping
    # Calculate how much we're cropping from each side
    crop_from_top = top_crop
    crop_from_bottom = height - bottom_crop
    crop_from_left = left_crop
    crop_from_right = width - right_crop
    
    # Store symmetric versions (use max for each pair)
    self.border_crop_y = max(crop_from_top, crop_from_bottom)
    self.border_crop_x = max(crop_from_left, crop_from_right)
    
    # Apply initial crop
    cropped_img = self.img[top_crop:bottom_crop, left_crop:right_crop]
    
    # Divide into three equal channels (BGR from top to bottom as specified)
    cropped_height = cropped_img.shape[0]
    channel_height = cropped_height // 3
    
    # Extract the three channels
    b_channel = cropped_img[0:channel_height, :]
    g_channel = cropped_img[channel_height:2*channel_height, :]
    r_channel = cropped_img[2*channel_height:3*channel_height, :]
    
    # Ensure all channels have exactly the same dimensions
    min_height = min(b_channel.shape[0], g_channel.shape[0], r_channel.shape[0])
    min_width = min(b_channel.shape[1], g_channel.shape[1], r_channel.shape[1])
    
    b_channel = b_channel[:min_height, :min_width]
    g_channel = g_channel[:min_height, :min_width]
    r_channel = r_channel[:min_height, :min_width]


    # Additional crop to remove per-channel borders (the colorful edges)
    # Crop an additional 3-5% from each channel
    additional_crop_h = int(min_height * 0.05)  # 5% from top and bottom
    additional_crop_w = int(min_width * 0.05)   # 5% from left and right

    b_channel = b_channel[additional_crop_h:-additional_crop_h, additional_crop_w:-additional_crop_w]
    g_channel = g_channel[additional_crop_h:-additional_crop_h, additional_crop_w:-additional_crop_w]
    r_channel = r_channel[additional_crop_h:-additional_crop_h, additional_crop_w:-additional_crop_w]

    # Update the stored border crop amounts to include this additional crop
    self.border_crop_y = self.border_crop_y + additional_crop_h
    self.border_crop_x = self.border_crop_x + additional_crop_w

    return b_channel, g_channel, r_channel

  def _align_pairs(self, img1, img2, delta):
    """
    Aligns two images using image pyramid (coarse-to-fine) approach.
    
    This method recursively downsamples images to create a pyramid, performs
    alignment at the coarsest level with a wide search, then refines at each
    higher resolution level with a narrow local search.
    
    Args:
        img1: Reference image (fixed)
        img2: Image to align (will be shifted)
        delta: Search window size at base level (e.g., 15 pixels)
    
    Returns: Tuple of (shift_y, shift_x) that minimizes the metric.
    """
    # Use image pyramid for efficient alignment
    return self._pyramid_align(img1, img2, delta)
  
  def _pyramid_align(self, img1, img2, base_delta, min_width=64):
    """
    Recursive image pyramid alignment using coarse-to-fine search.
    
    Args:
        img1: Reference image
        img2: Image to align
        base_delta: Search window at coarsest level
        min_width: Minimum width to stop downsampling (base of pyramid)
    
    Returns: Optimal (shift_y, shift_x) displacement
    """
    height, width = img1.shape
    
    # Base case: image is small enough, perform exhaustive search
    if width <= min_width:
        return self._exhaustive_search(img1, img2, base_delta)
    
    # Recursive case: downsample and align at coarser level
    # Downsample by factor of 2
    img1_small = self._downsample(img1)
    img2_small = self._downsample(img2)
    
    # Recursively find alignment at coarser level
    coarse_shift_y, coarse_shift_x = self._pyramid_align(
        img1_small, img2_small, base_delta, min_width
    )
    
    # Scale up the displacement (multiply by 2)
    scaled_shift_y = coarse_shift_y * 2
    scaled_shift_x = coarse_shift_x * 2
    
    # Refine with local search around scaled displacement
    # Use smaller window for refinement (±2 pixels)
    local_delta = 2
    
    best_score = float('inf')
    best_shift = (scaled_shift_y, scaled_shift_x)
    
    # Select metric function
    if self.metric == 'ncc':
        metric_fn = ncc
    elif self.metric == 'mse':
        metric_fn = mse
    elif self.metric == 'ssim':
        metric_fn = ssim
    else:
        raise ValueError(f"Unknown metric: {self.metric}")
    
    # Local search around scaled displacement
    for dy in range(-local_delta, local_delta + 1):
        for dx in range(-local_delta, local_delta + 1):
            shift_y = scaled_shift_y + dy
            shift_x = scaled_shift_x + dx
            
            # Apply shift
            shifted_img2 = custom_shifts(img2, (shift_y, shift_x), 
                                        dims=(0, 1), padding=self.padding)
            
            # Compute score (using cropped regions to avoid border artifacts)
            score = self._compute_score_with_crop(img1, shifted_img2, metric_fn)
            
            if score < best_score:
                best_score = score
                best_shift = (shift_y, shift_x)
    
    return best_shift
  
  def _exhaustive_search(self, img1, img2, delta):
    """
    Perform exhaustive search over displacement window at base pyramid level.
    
    Args:
        img1: Reference image
        img2: Image to align
        delta: Search window (±delta pixels)
    
    Returns: Optimal (shift_y, shift_x)
    """
    # Select metric function
    if self.metric == 'ncc':
        metric_fn = ncc
    elif self.metric == 'mse':
        metric_fn = mse
    elif self.metric == 'ssim':
        metric_fn = ssim
    else:
        raise ValueError(f"Unknown metric: {self.metric}")
    
    best_score = float('inf')
    best_shift = (0, 0)
    
    # Exhaustive search
    for shift_y in range(-delta, delta + 1):
        for shift_x in range(-delta, delta + 1):
            # Shift image
            shifted_img2 = custom_shifts(img2, (shift_y, shift_x), 
                                        dims=(0, 1), padding=self.padding)
            
            # Compute score with border cropping
            score = self._compute_score_with_crop(img1, shifted_img2, metric_fn)
            
            if score < best_score:
                best_score = score
                best_shift = (shift_y, shift_x)
    
    return best_shift
  
  def _compute_score_with_crop(self, img1, img2, metric_fn, crop_percent=0.1):
    """
    Compute alignment score after cropping borders to avoid edge artifacts.
    
    This is crucial: we crop 10-15% of borders before calculating metrics
    to remove jagged edges and non-overlapping regions that would skew results.
    
    Args:
        img1: First image
        img2: Second image  
        metric_fn: Metric function to use
        crop_percent: Percentage of border to crop (default 10%)
    
    Returns: Alignment score
    """
    height, width = img1.shape
    
    # Calculate crop amounts (10% from each side)
    crop_h = int(height * crop_percent)
    crop_w = int(width * crop_percent)
    
    # Ensure we don't crop too much
    if crop_h * 2 >= height or crop_w * 2 >= width:
        # Fall back to no cropping if image is too small
        return metric_fn(img1, img2)
    
    # Crop borders from both images
    img1_cropped = img1[crop_h:-crop_h, crop_w:-crop_w]
    img2_cropped = img2[crop_h:-crop_h, crop_w:-crop_w]
    
    # Compute metric on cropped regions
    return metric_fn(img1_cropped, img2_cropped)
  
  def _downsample(self, img):
    """
    Downsample image by factor of 2 using average pooling.
    
    Args:
        img: Input image tensor (H, W)
    
    Returns: Downsampled image (H/2, W/2)
    """
    height, width = img.shape
    
    # Ensure even dimensions for clean downsampling
    if height % 2 == 1:
        img = img[:-1, :]
    if width % 2 == 1:
        img = img[:, :-1]
    
    height, width = img.shape
    
    # Reshape and average 2x2 blocks
    # Method: reshape to (H/2, 2, W/2, 2) then average over the "2" dimensions
    img_reshaped = img.reshape(height // 2, 2, width // 2, 2)
    downsampled = torch.mean(img_reshaped, dim=(1, 3))
    
    return downsampled