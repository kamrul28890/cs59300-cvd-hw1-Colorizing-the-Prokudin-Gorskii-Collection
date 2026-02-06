"""Implements different image metrics."""

import torch
from skimage.metrics import structural_similarity

def ncc(img1, img2):
  """Takes two images and computes the negative normalized cross correlation.
     Lower the value, better the alignment.
     
     NCC is robust to brightness differences between channels, making it
     ideal for aligning different color channel exposures.
     
     Formula: -1 * sum((img1 - mean1) * (img2 - mean2)) / (norm(img1 - mean1) * norm(img2 - mean2))
  """
  # Flatten images for computation
  img1_flat = img1.flatten()
  img2_flat = img2.flatten()
  
  # Normalize to zero mean
  img1_normalized = img1_flat - torch.mean(img1_flat)
  img2_normalized = img2_flat - torch.mean(img2_flat)
  
  # Compute normalized cross-correlation
  numerator = torch.sum(img1_normalized * img2_normalized)
  denominator = torch.norm(img1_normalized) * torch.norm(img2_normalized)
  
  # Avoid division by zero
  if denominator == 0:
      ret = float('inf')
  else:
      # Return negative NCC (we want to minimize, so lower is better)
      ret = -numerator / denominator
  
  return ret 

def mse(img1, img2): 
  """Takes two images and computes the mean squared error (SSD).
     Lower the value, better the alignment.
     
     Also known as Sum of Squared Differences (SSD) in alignment contexts.
     Less robust than NCC for channels with different brightness levels.
     
     Formula: mean((img1 - img2)^2)
  """
  # Compute squared differences
  squared_diff = (img1 - img2) ** 2
  
  # Compute mean (equivalent to L2 norm for comparison purposes)
  ret = torch.mean(squared_diff)
  
  return ret 

def ssim(img1, img2):
  """Takes two image and compute the negative structural similarity.

  This function is given to you, nothing to do here.

  Please refer to the classic paper by Wang et al. of Image quality 
  assessment: from error visibility to structural similarity.
  """
  img1 = img1.numpy()
  img2 = img2.numpy()
  return -structural_similarity(img1, img2, data_range=img1.max() - img2.min())