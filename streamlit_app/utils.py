import pydot
import numpy as np
from matplotlib import pyplot as plt

def visualize_unet(topology_levels=4, base_filters=32, save_path=None):
    """
    Create a U-Net schematic with skip connections.

    Args:
        topology_levels (int): Number of levels in the contracting path.
        base_filters (int): Number of filters in the first level.
        save_path (str): Path to save PNG image (optional).
    
    Returns:
        pydot.Dot: The generated graph object.
    """
    graph = pydot.Dot(graph_type='digraph', rankdir='LR')

    encoder_nodes = []

    # Contracting path (encoder)
    for level in range(topology_levels):
        filters = base_filters * (2 ** level)
        node_name = f"Enc{level+1}\n{filters}F"
        node = pydot.Node(node_name, shape="box", style="filled", fillcolor="lightblue")
        graph.add_node(node)
        encoder_nodes.append(node)
        if level > 0:
            graph.add_edge(pydot.Edge(encoder_nodes[level-1], encoder_nodes[level]))

    # Bottleneck
    bottleneck_filters = base_filters * (2 ** topology_levels)
    bottleneck_node = pydot.Node(f"Bottleneck\n{bottleneck_filters}F", shape="box", style="filled", fillcolor="orange")
    graph.add_node(bottleneck_node)
    graph.add_edge(pydot.Edge(encoder_nodes[-1], bottleneck_node))

    # Expanding path (decoder)
    decoder_nodes = []
    for level in reversed(range(topology_levels)):
        filters = base_filters * (2 ** level)
        node_name = f"Dec{level+1}\n{filters}F"
        node = pydot.Node(node_name, shape="box", style="filled", fillcolor="lightgreen")
        graph.add_node(node)
        graph.add_edge(pydot.Edge(bottleneck_node, node))
        decoder_nodes.append(node)

        # Skip connection
        graph.add_edge(pydot.Edge(encoder_nodes[level], node, style="dashed", color="red"))

        bottleneck_node = node

    if save_path:
        graph.write_png(save_path)

    return graph

def patches_to_image(patches: list, original_shape: tuple, patch_size: int, padded_shape: tuple = None):
    """
    Reconstructs an image from square patches.

    Args:
        patches (list of np.ndarray): List of patches (H x W x C or H x W).
        original_shape (tuple): Original image shape (H, W, C) or (H, W).
        patch_size (int): Size of square patches.
        padded_shape (tuple, optional): Shape of the padded image if padding was used.

    Returns:
        np.ndarray: Reconstructed image with original_shape.
    """
    if padded_shape is None:
        padded_shape = original_shape

    H_pad, W_pad = padded_shape[:2]
    channels = 1 if len(padded_shape) == 2 else padded_shape[2]

    if channels == 1:
        img = np.zeros((H_pad, W_pad), dtype=patches[0].dtype)
    else:
        img = np.zeros((H_pad, W_pad, channels), dtype=patches[0].dtype)

    idx = 0
    for y in range(0, H_pad, patch_size):
        for x in range(0, W_pad, patch_size):
            img[y:y+patch_size, x:x+patch_size] = patches[idx]
            idx += 1

    # Crop back to original shape (remove padding if any)
    H_orig, W_orig = original_shape[:2]
    img = img[:H_orig, :W_orig]

    return img

def image_to_patches(img: np.ndarray, patch_size: int, pad: bool = True):
    """
    Splits an image into non-overlapping square patches, with optional padding.

    Args:
        img (np.ndarray): Input image array (H×W×C or H×W).
        patch_size (int): Size of the square patch.
        pad (bool): If True, pad image so H and W are divisible by patch_size.

    Returns:
        patches (list of np.ndarray), padded_img (np.ndarray)
    """
    H, W = img.shape[:2]

    if pad:
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        img = np.pad(img, 
                     ((0, pad_h), (0, pad_w)) + ((0,0),) * (img.ndim - 2),
                     mode='constant')
        H, W = img.shape[:2]

    patches = [
        img[y:y+patch_size, x:x+patch_size]
        for y in range(0, H, patch_size)
        for x in range(0, W, patch_size)
    ]

    return patches, img

def f1_score_from_images(pred, gt, threshold=127):
    """
    Computes the F1 score between a predicted image and a ground truth image.

    Args:
        pred (np.ndarray): Predicted image (H x W), values can be probabilities, labels, or grayscale.
        gt (np.ndarray): Ground truth image (H x W), values should be binary (0 or 1) or grayscale.
        threshold (int or float): Threshold to binarize images if not already {0,1}.

    Returns:
        float: F1 score
    """
    # Binarize
    pred_bin = (pred > threshold).astype(np.uint8)
    gt_bin = (gt > threshold).astype(np.uint8)

    # Flatten to 1D for comparison
    pred_flat = pred_bin.flatten()
    gt_flat = gt_bin.flatten()

    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))

    if tp + fp + fn == 0:
        return 1.0  # both pred and gt empty → perfect match
    else:
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1

def iou_from_images(pred, gt):
    """
    Computes pixel-wise Intersection over Union (IoU) for each class in gt.
    
    Args:
        pred: Predicted segmentation mask
        gt: Ground truth segmentation mask
    
    Returns:
        returns the average IoU across all classes
    """
    classes = np.unique(gt, return_counts=False)
    metrics = []
    
    for c in classes:
        # Filter classes (create binary masks)
        gt_c = (gt == c).astype(int)
        pred_c = (pred == c).astype(int)
        
        # Calculate IoU
        intersection = (gt_c & pred_c).sum()
        union = (gt_c | pred_c).sum()
        iou = (intersection / union) if union != 0 else 0
        
        metrics.append(iou)

    return sum(metrics) / len(metrics) if metrics else 0

def compute_unet_params(num_levels, base_filters, input_channels=3):
    """
    Computes approximate number of parameters for a U-Net.
    
    Args:
        num_levels (int): number of encoder/decoder levels
        base_filters (int): number of filters in the first layer
        input_channels (int): number of channels in input image

    Returns:
        total_params (int)
    """
    total_params = 0
    in_ch = input_channels

    # Encoder
    for level in range(num_levels):
        out_ch = base_filters * (2 ** level)
        # Two conv layers per level
        total_params += (in_ch * 3 * 3 * out_ch) + out_ch      # Conv1
        total_params += (out_ch * 3 * 3 * out_ch) + out_ch      # Conv2
        in_ch = out_ch

    # Bottleneck (2 conv layers)
    out_ch = base_filters * (2 ** num_levels)
    total_params += (in_ch * 3 * 3 * out_ch) + out_ch
    total_params += (out_ch * 3 * 3 * out_ch) + out_ch

    # Decoder
    for level in reversed(range(num_levels)):
        out_ch = base_filters * (2 ** level)
        # Two conv layers per level
        total_params += (in_ch * 3 * 3 * out_ch) + out_ch
        total_params += (out_ch * 3 * 3 * out_ch) + out_ch
        in_ch = out_ch

    return total_params

def compute_unet_flops(num_levels, base_filters, input_size, input_channels=3):
    """
    Computes approximate FLOPs for a U-Net.
    
    Args:
        num_levels (int)
        base_filters (int)
        input_size (int or tuple): input image size (H=W assumed square)
        input_channels (int)
        
    Returns:
        total_flops (int)
    """
    if isinstance(input_size, int):
        H = W = input_size
    else:
        H, W = input_size

    total_flops = 0
    in_ch = input_channels
    h, w = H, W

    # Encoder
    for level in range(num_levels):
        out_ch = base_filters * (2 ** level)
        # Conv1
        total_flops += 2 * in_ch * 3 * 3 * h * w * out_ch
        # Conv2
        total_flops += 2 * out_ch * 3 * 3 * h * w * out_ch
        in_ch = out_ch
        h //= 2  # After pooling
        w //= 2

    # Bottleneck
    out_ch = base_filters * (2 ** num_levels)
    total_flops += 2 * in_ch * 3 * 3 * h * w * out_ch
    total_flops += 2 * out_ch * 3 * 3 * h * w * out_ch
    in_ch = out_ch

    # Decoder
    for level in reversed(range(num_levels)):
        out_ch = base_filters * (2 ** level)
        h *= 2  # After upsample
        w *= 2
        total_flops += 2 * in_ch * 3 * 3 * h * w * out_ch
        total_flops += 2 * out_ch * 3 * 3 * h * w * out_ch
        in_ch = out_ch

    return total_flops

def reshape_board_input(arr:np.ndarray):
    """ this function takes a 4D numpy array and reshapes it into a 1D depth wise raster mode array to be used for inference on the board
    """
    np_arr1d = np.zeros((np.prod(arr.shape),), dtype=arr.dtype)
    line_counter = 0
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    for i in range(arr.shape[0]):
        for r in range(arr.shape[2]):
            for c in range(arr.shape[3]):
                for ch in range(arr.shape[1]):
                    np_arr1d[line_counter] = arr[i, ch, r, c]
                    line_counter += 1
    return np_arr1d

def reshape_board_output(arr:np.ndarray, output_shape):
    """ This function takes a numpy array that is in depth wise raster order 1D and converts it to a 4D array
    """
    assert arr.ndim == 1, "Input array must be 1D"
    assert isinstance(output_shape, (list, tuple)) and len(output_shape) == 4, "output_shape must be a list or tuple of length 4"
    assert np.prod(output_shape) == arr.size, "output_shape does not match the size of the input array"
    line_counter = 0
    np_arr4d = np.zeros(output_shape, dtype=arr.dtype)
    for i in range(np_arr4d.shape[0]):
        for r in range(np_arr4d.shape[2]):
            for c in range(np_arr4d.shape[3]):
                for ch in range(np_arr4d.shape[1]):
                    np_arr4d[i, ch, r, c] = arr[line_counter]
                    line_counter += 1
    assert line_counter == arr.size
    return np_arr4d


def visualize_oil_output_( sample: np.ndarray, output_file_name: str = 'run_test', file_type: str = 'svg', fig_size: int = 25, font_size: int = 25, save_fig: bool = False, show_fig: bool = True, cmap: str = 'jet'):
    """
    Visualizes the output of the oil segmentation model.
    """
    def softmax(x, axis=1):
        # Subtract max for numerical stability
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    sample = softmax(sample, axis=1)
    sample = np.argmax(sample, axis=1)
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': font_size}
    if sample.ndim == 3:
        sample = sample[0]
    elif sample.ndim != 2:
        raise ValueError('Environment should be 2D array')
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(sample, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    cbar.ax.set_aspect('auto')
    ax.patch.set_linewidth(3)
    if save_fig:
        plt.savefig(f'{output_file_name}.{file_type}', format=file_type)
    if show_fig:
        plt.show()
    else:
        plt.close(fig)
        

def visualize_oil_output(
    sample: np.ndarray,
    output_file_name: str = 'run_test',
    file_type: str = 'svg',
    fig_size: int = 25,
    font_size: int = 25,
    save_fig: bool = False,
    show_fig: bool = True,
    cmap: str = 'jet'):
    """
    Visualizes the output of the oil segmentation model and returns a matplotlib figure
    so it can be displayed in Streamlit with st.pyplot().
    """
    def softmax(x, axis=1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    # Apply softmax + argmax
    sample = softmax(sample, axis=1)
    sample = np.argmax(sample, axis=1)

    # Ensure correct shape
    if sample.ndim == 3:
        sample = sample[0]
    elif sample.ndim != 2:
        raise ValueError('Environment should be 2D array')

    # Font settings
    font = {'family': 'sans-serif', 'weight': 'bold', 'size': font_size}
    plt.rc('font', **font)

    # Create figure
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(sample, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    cbar.ax.set_aspect('auto')
    ax.patch.set_linewidth(3)

    # Optionally save
    if save_fig:
        plt.savefig(f'{output_file_name}.{file_type}', format=file_type)

    # Return fig for Streamlit
    if show_fig:
        return fig
    else:
        plt.close(fig)
        return None