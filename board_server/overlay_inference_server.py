"""
overlay_timeit_server.py

A TCP server for running and benchmarking PYNQ FPGA overlays. The server listens for incoming connections,
loads FPGA overlays, allocates input/output buffers, and measures execution time and power consumption. Results are sent back to the client.

Main Features:
- Receives overlay bitstream and configuration files from the client.
- Loads overlays onto the PYNQ board and sets up register maps.
- Allocates and initializes input, output, and optional weight buffers.
- Measures idle and run-time power using PYNQ's power rails.
- Benchmarks overlay execution time using timeit.
- Sends latency and power metrics back to the client.
- Cleans up resources and temporary files after each run.

Functions:
- run_pl(): Starts the hardware accelerator and waits for completion.
- start_server(host, port, num_reps): Main server loop handling client connections, overlay loading,
    buffer allocation, benchmarking, and result transmission.

Usage:
        python overlay_timeit_server.py --port <PORT> --num_reps <REPETITIONS>

Arguments:
        --port: Port number to listen on (default: 12345)
        --num_reps: Number of repetitions for timing and power measurement (default: 15)

Requirements:
- PYNQ board with overlays and associated files (.bit, .tcl, .hwh, .npy) in /tmp directory.
- pynq, numpy, and other dependencies installed.

Note:
- This script is intended for use on a PYNQ board and requires root privileges for hardware access.


"""
# server.py
import socket
import pynq
import numpy as np
from pynq import Overlay, allocate
from pynq import PL
import time
import timeit
import struct
import os
import gc
import argparse


register_map = None

def run_pl():
    """
    Starts the programmable logic (PL) operation by setting the AP_START bit in the register map's control register.
    Waits for the operation to complete by polling the AP_IDLE bit.
    Raises:
        AttributeError: If the register map is not initialized.
    """

    if register_map is None:
        raise AttributeError("Register map is not initialized")
    register_map.CTRL.AP_START = 1
    while(register_map.CTRL.AP_IDLE != 1):
        pass

def start_server(host, port, num_reps, time_limit):
    """
    Starts a TCP server that listens for client connections to perform hardware overlay loading, 
    data transfer, and performance measurement on a Pynq board.
    The server expects clients to send a header specifying input/output array shapes and the bitstream 
    file name, followed by the bitstream and optional weights. It loads the specified overlay, allocates 
    input/output buffers, optionally loads weights, and measures execution time and power consumption 
    for a hardware-accelerated operation. Results are sent back to the client.
    Args:
        host (str): The IP address or hostname to bind the server socket.
        port (int): The port number to listen for incoming connections.
        num_reps (int): Number of repetitions to run the hardware operation for timing and power measurement.
    Protocol:
        - Client sends a 12-byte header: 3 unsigned integers (input_shape, output_shape, bit_file_name_length).
        - Client sends the bitstream file name (UTF-8, length specified above).
        - Server expects corresponding .bit, .tcl, .hwh, and optionally .npy files in /tmp.
        - Server loads overlay, allocates buffers, optionally loads weights, and runs the hardware operation.
        - Server measures and records ideal and run-time power, as well as execution latency.
        - Server sends back a struct-packed triple of floats: (latency_ns, ideal_power_mean, run_power_mean).
    Error Handling:
        - If an error occurs, the server sends -2.0 for all result values.
        - Cleans up resources and resets the programmable logic after each request.
    """

    global register_map
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server_socket.bind((host, port))
        server_socket.listen(50)
        print(f"Server listening on {host}:{port}")
        rails = pynq.get_rails()

        while True:
            try:
                client_socket, address = server_socket.accept()
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"Connection from {address}")
                header = client_socket.recv(12)  # III I=4  input_shape, output_shape, text_length
                if not header:  # Client disconnected
                    time.sleep(5)
                    continue

                # Receive array data
                input_shape, output_shape, bit_file_name_length = struct.unpack('!III', header)
                bit_file_name = client_socket.recv(bit_file_name_length).decode('utf-8')
                print('#'*20, bit_file_name, '#'*20)
                # Check for required overlay files, send -2 for all values if missing
                required_files = [f'/tmp/{bit_file_name}.bit', f'/tmp/{bit_file_name}.tcl', f'/tmp/{bit_file_name}.hwh']
                missing = [f for f in required_files if not os.path.exists(f)]
                if missing:
                    print(f"Missing required files: {', '.join(missing)}")
                    raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")
                overlay = Overlay(f'/tmp/{bit_file_name}.bit')
                print(f'\tOverlay Written {bit_file_name}')
                register_map = overlay.top_level_0_0.register_map
                if input_shape == 0:
                    print(f'\tLoading input array from {bit_file_name}_input.npy')
                    array = np.load(f'/tmp/{bit_file_name}_input.npy')
                else:
                    array = np.random.randint(-128, 127, input_shape)

                # Process the array
                input_array = allocate(shape=array.shape, dtype=array.dtype)
                output_array = allocate(shape=(output_shape,), dtype=array.dtype)

                input_array[:] = array
                output_array[:] = -1
                input_array.flush()
                output_array.flush()
                input_mem_attr=None
                output_mem_attr=None
                weight_mem_attr=None
                for attr in dir(register_map):
                    if '_input_mem_offset_1' in attr:
                        input_mem_attr=attr
                    elif '_output_mem_offset_1' in attr:
                        output_mem_attr=attr
                    elif 'off_chip_weights_offset_1' in attr:
                        weight_mem_attr=attr
                if weight_mem_attr is not None:
                    print('\tFound weight interface')
                    weights_array_path = f'/tmp/{bit_file_name}_weight.npy'
                    if not os.path.exists(weights_array_path):
                        raise FileNotFoundError(f'File {weights_array_path} not found')
                    weight_npy = np.load(weights_array_path)
                    weight_array = allocate(shape=(weight_npy.size,), dtype=np.int8)
                    weight_array[:] = weight_npy
                    weight_array.flush()
                    del weight_npy
                    setattr(register_map, weight_mem_attr, weight_array.device_address)
                else:
                    print('\tNo weight interface found')

                if input_mem_attr is None:
                    raise AttributeError('Input memory attribute not found in register map')
                if output_mem_attr is None:
                    raise AttributeError('Output memory attribute not found in register map')
                setattr(register_map, input_mem_attr, input_array.device_address)
                setattr(register_map, output_mem_attr, output_array.device_address)
                ideal_power_mean = -99
                time_ns= -99
                run_power_mean = -99
                if len(rails) > 0:
                    print(f'\tMeasuring ideal Power')
                    recorder = pynq.DataRecorder(rails['INT'].power)
                    recorder.reset()
                    with recorder.record(0):
                        time.sleep(10)
                    ideal_power_mean = recorder.frame.mean()['INT_power']

                print('\tMeasuring Time')
                register_map.CTRL.AP_START = 1
                for ts in range(time_limit):
                    time.sleep(1)
                    if register_map.CTRL.AP_IDLE:
                        print(f'\tDesign finishes in less than {time_limit} seconds :)')
                        break
                if not register_map.CTRL.AP_IDLE:
                    print(f'\tDesign did not finish in {time_limit} seconds :(')
                else:
                    if input_shape == 0:
                        output_array.invalidate()
                        np.save(f'/tmp/{bit_file_name}_output.npy', output_array)
                    time_ns=timeit.timeit("run_pl()", setup="from __main__ import run_pl", number=num_reps, timer=time.perf_counter_ns)
                    time_ns=time_ns/(num_reps)
                    print(f'\trun_time   : {time_ns*1e-9:.2f}s  or {time_ns:.2f} ns')
                    # Record static power
                    if len(rails) > 0:
                        print('\tMeasuring Power')
                        recorder.reset()
                        with recorder.record(0):
                            for _ in range(num_reps):
                                run_pl()
                        run_power_mean = recorder.frame.mean()['INT_power']
                    print(f'\tMeasured Power Ideal: {ideal_power_mean}, \n\tRun Time Power: {run_power_mean}')
                    input_array.invalidate()
                    output_array.invalidate()
                # Pickle and send the result
                latency = struct.pack('!ddd', time_ns, ideal_power_mean, run_power_mean)
                client_socket.sendall(latency)
                print('\tSent output')
            except KeyboardInterrupt:
                print("Server interrupted by user.")
                break
            except (OSError, FileNotFoundError, AttributeError) as e:
                print(f"OSError occurred: {e}")
                # Send -2 for all values to indicate error
                latency = struct.pack('!ddd', -2.0, -2.0, -2.0)
                client_socket.sendall(latency)
            finally:
                PL.reset()
                if 'client_socket' in locals():
                    client_socket.close()
                if 'overlay' in locals():
                    del overlay
                    os.system(f'rm -rf /tmp/{bit_file_name}.*')
                    os.remove(f'/usr/lib/firmware/{bit_file_name}.bin')  # remove the bitstream from the firmware folder to avoid running out of space
                gc.collect()
                time.sleep(1)

if __name__ == "__main__":
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    host = s.getsockname()[0]
    s.close()
    parser = argparse.ArgumentParser(description="Pynq Overlay Timeit Server")
    parser.add_argument('--port', type=int, default=12345, help='Port to listen on')
    parser.add_argument('--num_reps', type=int, default=15, help='Number of repetitions for measuring time and power')
    parser.add_argument('--time_limit', type=int, default=60, help='Time limit for the operation in seconds')
    args = parser.parse_args()
    start_server(host, args.port, args.num_reps, args.time_limit)

