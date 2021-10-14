import matplotlib.pyplot as plt
import numpy as np


def debug_visualize(tgt_tensor):
    """
    Visualize target tensor. If batch dimension exists, visualizes the first instance. Multi-channel inputs are shown as 'slices'.
    If number of channels is 3, displayed in RGB. Otherwise results are shown as single channel images.
    For inputs that are float, we assume that the tgt_tensor values are normalized within [0, 1].
    For inputs that are int, we assume that the tgt_tensor values are normalized within [0, 255].

    Args:
        tgt_tensor: torch.tensor with one of the following shapes: (H, W), (H, W, C), (B, H, W, C)
    
    Returns:
        None
    """
    if "torch" in str(type(tgt_tensor)):
        vis_tgt = tgt_tensor.cpu().float().numpy()
    elif "numpy" in str(type(tgt_tensor)):
        vis_tgt = tgt_tensor.astype(np.float)
    else:
        raise ValueError("Invalid input!")
    
    if vis_tgt.max() > 2.0:  # If tgt_tensor is in range greater than 2.0, we assume it is an RGB image
        vis_tgt /= 255.

    if len(vis_tgt.shape) == 2:
        H, W = vis_tgt.shape
        plt.imshow(vis_tgt, cmap='gray', vmin=vis_tgt.min(), vmax=vis_tgt.max())
        plt.show()
    
    elif len(vis_tgt.shape) == 3:
        H, W, C = vis_tgt.shape

        if C > 3 or C == 2:
            fig = plt.figure(figsize=(50, 50))
            for i in range(C):
                fig.add_subplot(C // 2, 2, i + 1)
                plt.imshow(vis_tgt[..., i], cmap='gray', vmin=vis_tgt[..., i].min(), vmax=vis_tgt[..., i].max())
        elif C == 3:  # Display as RGB
            plt.imshow(vis_tgt)
        elif C == 1:
            plt.imshow(vis_tgt, cmap='gray', vmin=vis_tgt.min(), vmax=vis_tgt.max())
        
        plt.show()
    
    elif len(vis_tgt.shape) == 4:
        B, H, W, C = vis_tgt.shape
        vis_tgt = vis_tgt[0]
        
        if C > 3 or C == 2:
            fig = plt.figure(figsize=(50, 50))
            for i in range(C):
                fig.add_subplot(C // 2, 2, i + 1)
                plt.imshow(vis_tgt[..., i], cmap='gray', vmin=vis_tgt[..., i].min(), vmax=vis_tgt[..., i].max())
        elif C == 3:  # Display as RGB
            plt.imshow(vis_tgt)
        elif C == 1:
            plt.imshow(vis_tgt, cmap='gray', vmin=vis_tgt.min(), vmax=vis_tgt.max())
        
        plt.show()
