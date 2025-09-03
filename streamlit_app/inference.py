
import os
import numpy as np
import socket
import struct
from utils import image_to_patches, patches_to_image, reshape_board_input, reshape_board_output
import json


BATCH_SIZE=1
def fpga_inference(input_patches:list, bitstream:str):
    config_path = os.path.join(os.path.dirname(__file__), 'board_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        board_config = json.load(f)

    BOARD_PATH = board_config['BOARD_PATH']
    BOARD_USER = board_config['BOARD_USER']
    BOARD_ADDRESS = board_config['BOARD_ADDRESS']
    BOARD_PORT = board_config['BOARD_PORT']
    if 'FOREST' in bitstream:
        input_shape = [BATCH_SIZE, 18, 128, 128]
        output_shape = [BATCH_SIZE, 2, 128, 128]
    elif 'OIL' in bitstream:
        input_shape = [BATCH_SIZE, 9, 128, 128]
        output_shape = [BATCH_SIZE, 11, 128, 128]
    else:
        raise Exception("Bitstream name must contain either 'FOREST' or 'OIL' to determine input/output shapes.")
    block_design_name = os.path.basename(bitstream).split('.')[0]
    main_dir = os.path.dirname(bitstream)

    bit_file = os.path.join(main_dir, f"{block_design_name}.bit")
    tcl_file = os.path.join(main_dir, f"{block_design_name}.tcl")
    hwh_file = os.path.join(main_dir, f"{block_design_name}.hwh")

    if not(os.path.exists(bit_file) and os.path.exists(tcl_file) and os.path.exists(hwh_file)):
        raise Exception(f"Bit file, tcl file or hwh file not found in {main_dir}")

    input_array = np.array(input_patches)
    np_path = f'/tmp/{block_design_name}_input.npy'
    if input_array.ndim != 1:
        input_array = reshape_board_input(input_array)
    np.save(np_path, input_array)
    os.system(f"scp {bit_file} {tcl_file} {hwh_file} {np_path} {BOARD_USER}@{BOARD_ADDRESS}:{BOARD_PATH}")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(None)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_socket.connect((BOARD_ADDRESS, BOARD_PORT))
    text_bytes = block_design_name.encode('utf-8')
    text_length = len(text_bytes)

    try:
        input_size = np.prod(input_shape)
        output_size = np.prod(output_shape)
        header = struct.pack(f'!III{text_length}s', 0, output_size, text_length, text_bytes)
        client_socket.sendall(header)
        # Receive result metadata
        result_header = client_socket.recv(24) # time_ns, ideal_power_mean, run_power_mean
        latency, ideal_power_mean, run_power_mean = struct.unpack('!ddd', result_header)
        os.system(f"scp {BOARD_USER}@{BOARD_ADDRESS}:{BOARD_PATH}{block_design_name}_output.npy /tmp/{block_design_name}_output.npy")
        output_array = np.load(f'/tmp/{block_design_name}_output.npy')
        os.system(f'rm /tmp/{block_design_name}_output.npy')
        fps = 1 / latency
        return ideal_power_mean, run_power_mean, fps, reshape_board_output(output_array, output_shape)
    finally:
        client_socket.close()


def fpga_inference_dummy(input_patches: list, bitstream):
    power_idle = 0.0
    power_run = 0.1
    fps = 0.0
    predicted_patches = input_patches
    
    return power_idle, power_run, fps, predicted_patches


def fpga_inference_wrapper(input_img: np.ndarray, bitstream, patch_size: int, topology_quant_params):
    # toplogy_quant_params is a list of (input_scale, input_zero_point, output_scale, output_zero_point, qmin, qmax)
    # Function that cuts input_img into patches
    # input_img -> patches

    input_scale, input_zero_point, output_scale, output_zero_point, qmin, qmax = topology_quant_params
    input_img = np.round((input_img/input_scale) + input_zero_point)
    input_img = np.clip(input_img, qmin, qmax)
    input_img = input_img.astype(np.int8) if qmin == -128 else input_img.astype(np.int16)
    input_patches, padded_img = image_to_patches(input_img, patch_size, True)
    # power_idle, power_run, fps, predicted_patches = fpga_inference_dummy(input_patches, bitstream)
    predicted_patches = []
    power_idle, power_run, fps = 0.0, 0.0, 0.0
    for patch in input_patches:
        patch = np.transpose(patch, (2, 0, 1))
        p_power_idle, p_power_run, p_fps, predicted_patch = fpga_inference(patch, bitstream)
        predicted_patches.append(predicted_patch)
        power_idle += p_power_idle
        power_run += p_power_run
        fps += p_fps
    power_idle /= len(input_patches)
    power_run /= len(input_patches)
    fps /= len(input_patches)

    # Function that runs inference
    predicted_img = patches_to_image(predicted_patches, input_img.shape, patch_size, padded_img.shape)
    predicted_img = predicted_img.astype(np.float32)
    predicted_img = (predicted_img - output_zero_point) * output_scale
    predicted_img = np.mean(predicted_img, axis=-1).astype(np.uint8)
  
    return power_idle, power_run, fps, predicted_img