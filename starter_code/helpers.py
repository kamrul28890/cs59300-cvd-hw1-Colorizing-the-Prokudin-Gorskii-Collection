"""Implements helper functions."""

import torch


def custom_shifts(input, shifts, dims=None, padding='circular'):
  """Shifts the input tensor by the specified shifts along
     the specified dimensions. Supports circular and zero padding.

     Args:
         input: Input tensor to shift (2D image tensor)
         shifts: Tuple of (shift_y, shift_x) - pixels to shift
         dims: Tuple of dimensions to shift along (default: (0, 1))
         padding: Padding mode - 'circular' or 'zero'
     
     Returns: Shifted tensor along the specified dimensions
       padded following the padding scheme.
  """
  if dims is None:
      dims = (0, 1)
  
  shift_y, shift_x = shifts
  
  # Use torch.roll for efficient circular padding (best for most cases)
  if padding == 'circular':
      ret = torch.roll(input, shifts=(shift_y, shift_x), dims=dims)
  else:  # zero padding
      # Manual shifting with zero padding
      ret = input.clone()
      
      # Shift along y-axis (vertical)
      if shift_y > 0:
          ret = torch.cat([torch.zeros(shift_y, input.shape[1]), ret[:-shift_y, :]], dim=0)
      elif shift_y < 0:
          ret = torch.cat([ret[-shift_y:, :], torch.zeros(-shift_y, input.shape[1])], dim=0)
      
      # Shift along x-axis (horizontal)
      if shift_x > 0:
          ret = torch.cat([torch.zeros(ret.shape[0], shift_x), ret[:, :-shift_x]], dim=1)
      elif shift_x < 0:
          ret = torch.cat([ret[:, -shift_x:], torch.zeros(ret.shape[0], -shift_x)], dim=1)
  
  return ret
