   
import streamlit as st
from PIL import Image
import time
import pandas as pd
import os
import numpy as np
import io

from sklearn.decomposition import PCA
import cairosvg
        
from utils import visualize_unet, f1_score_from_images, iou_from_images, visualize_oil_output, compute_unet_params, compute_unet_flops
from inference import fpga_inference_wrapper
    
def main():
    
    batch_size = 1
    patch_size = 128
    
    fpga_images = {
        "ZCU104": "app_images/ZCU104.webp",
        "ZCU102": "app_images/ZCU102.jpg",
        "ULTRA96-V2": "app_images/ULTRA96-V2.jpg"
    }
    
    QuantParams = {
        'OIL':
            {
                'D1F16':(0.009468217380344868, -45, 0.2623797059059143, 31, -128, 127),
                'D2F32':(0.009468217380344868, -45, 0.2372426688671112, 24, -128, 127),
                'D2F16':(0.009468217380344868, -45, 0.23054447770118713, 9, -128, 127)
            },
        'FOREST':
            {
                'D1F32':(0.01057740394026041, 2873, 0.0002576748374849558, -7402, -32768, 32767),
                'D1F16':(0.0064476728439331055, -16173, 0.00013310363283380866, -5303, -32768, 32767),
                'D2F16':(0.0064476728439331055, -16173, 0.0001152641634689644, -13881, -32768, 32767)
            }
    }

    st.set_page_config(page_title="Streamlit Sidebar App", layout="wide")
       
    with st.sidebar: 
        st.image("app_images/logo.PNG", use_container_width=True)
        
        st.title("FPGA4Earth Configuration App")

        if "show_predictions" not in st.session_state:
            st.session_state.show_predictions = False
        else:
            st.session_state.show_predictions = False  
        
        # Dropdown for Task selection
        task_options = ['Satellite Image Forest Segmentation', 'Radar Image Oil Spill Segmentation']
        task = st.selectbox('Select Task', task_options)     
        # --- Set folders based on task
        if task == 'Satellite Image Forest Segmentation':
            input_folder = "forest_data/input"
            gt_folder = "forest_data/gt"
            gt_npy_folder = None
            application = "FOREST"
            input_channels = 18
            model_precision = "INT16"
            data_precision = "INT16"
            topology_options = ['D1F16', 'D1F32', 'D2F16']
        elif task == 'Radar Image Oil Spill Segmentation':
            input_folder = "oil_data/input"
            gt_folder = "oil_data/gt"  
            gt_npy_folder = "oil_data/gt_npy" 
            application = "OIL" 
            input_channels = 9
            model_precision = "INT8"
            data_precision = "INT8"
            topology_options = ['D1F16', 'D2F16', 'D2F32']        
        # --- Load images
        input_images = sorted(os.listdir(input_folder))
        gt_images = sorted(os.listdir(gt_folder))        
        
        # Slider to select number of images (from 1 to 1024)
        subset_size = st.slider("Select the number of images:", 
            min_value=0, max_value=len(input_images), value=1, step=1)
        
        input_images = input_images[:subset_size]
        gt_images = gt_images[:subset_size]
        if gt_npy_folder is not None:
            gt_npy = sorted(os.listdir(gt_npy_folder))
            gt_npy = gt_npy[:subset_size]
        
        # Dropdown for Hardware platform selection
        fpga_options = ['ZCU104', 'ZCU102 (Coming soon)', 'ULTRA96-V2 (Coming soon)']
        #fpga_options = ['ZCU104']
        fpga = st.selectbox('Select FPGA Board', fpga_options)
        # Only allow actual supported option
        if "Coming soon" in fpga:
            st.warning("This FPGA board is not supported yet. Please select ZCU104.")
        else:
            pass
            #st.success(f"You selected: {fpga}")
        
        # Dropdown for Hardware platform selection
        topology = st.selectbox('Select Topology', topology_options)
        st.write(f"**D:** Number of levels")
        st.write(f"**F:** Number of filters in the first level")
        if topology == 'D1F32':
            topology_levels = 1
            base_filters = 32
        elif topology == 'D1F16':
            topology_levels = 1
            base_filters = 16
        elif topology == 'D2F16':
            topology_levels = 2
            base_filters = 16
        elif topology == 'D2F32':
            topology_levels = 2
            base_filters = 32        

        topology_quant_params = QuantParams[application][topology]

        bitstream = os.path.join("bitstreams", f"{application}_{fpga}_{topology}.bit")
    
    #############################################################################################################################################################
    col1, col2, col3 = st.columns([1, 1, 1]) 
    
    with col1:
        params = compute_unet_params(topology_levels, base_filters, input_channels) / 1e6
        flops = compute_unet_flops(topology_levels, base_filters, patch_size, input_channels) / 1e9
        # Display selected options
        st.write("### Selected Configuration:")
        st.write(f"**Task:** {task}")
        st.write(f"**FPGA Platform:** {fpga}")
        st.write(f"**Topology:** {topology} (Params: {params:.4f}M, OPS: {flops:.2f}G)")        
        #st.write(f"**Num Params:** {params:.4f} M")
        #st.write(f"**FLOPS:** {flops:.4f} M")
        st.write(f"**Batch Size:** {batch_size}")
        st.write(f"**Patch Size:** {patch_size}x{patch_size} px")
        st.write(f"**Model Precision:** {model_precision}")
        st.write(f"**Data Precision:** {data_precision}")
    
    with col2:
        # Display hardware image
        if fpga in fpga_images:
            st.image(fpga_images[fpga], caption=fpga, width=400)    
    
    with col3:
        graph = visualize_unet(topology_levels, base_filters)
        png_bytes = graph.create_png()
        image = Image.open(io.BytesIO(png_bytes))
        st.image(image, caption="U-Net Graph", use_container_width=True)
    
    #############################################################################################################################################################
    # Add a horizontal line (separating the next object from previous ones)
    st.markdown('---')
    
    # Initialize session state to store experiments
    if "experiments" not in st.session_state:
        st.session_state.experiments = []        
    
    if "run_count" not in st.session_state:
        st.session_state.run_count = 0  # Tracks number of runs

    # Button to run inference
    if st.button('Run Inference'):
        with st.status("Running inference...", expanded=False) as status:            
            
            ################################################################################################
            power_idle_list = []
            power_run_list = []
            fps_list = []
            accuracy_list = []
            prediction_img_list = []
            if task == 'Satellite Image Forest Segmentation':
                zipp = zip(input_images, gt_images)
                gt_folder_ = gt_folder
            elif task == 'Radar Image Oil Spill Segmentation':
                zipp = zip(input_images, gt_npy)
                gt_folder_ = gt_npy_folder
            for inp, gt in zipp:
                inp_path = os.path.join(input_folder, inp)
                gt_path = os.path.join(gt_folder_, gt)
                
                if task == 'Satellite Image Forest Segmentation':
                    inp_img = Image.open(inp_path)
                    gt_img = Image.open(gt_path)
                    # Now it's a NumPy ndarray
                    inp_img = np.array(inp_img)
                    gt_img = np.array(gt_img)
                elif task == 'Radar Image Oil Spill Segmentation':
                    inp_img = np.load(inp_path)
                    gt_img = np.load(gt_path)                  
                
                ################################################################################################

                power_idle, power_run, fps, prediction_img = fpga_inference_wrapper(inp_img, bitstream, patch_size, topology_quant_params) 
                
                ################################################################################################ 
                if task == 'Satellite Image Forest Segmentation':
                    accuracy = f1_score_from_images(prediction_img, gt_img, threshold=127) 
                elif task == 'Radar Image Oil Spill Segmentation':
                    accuracy = iou_from_images(prediction_img, gt_img)
                             
                power_idle_list.append(power_idle)
                power_run_list.append(power_run)
                fps_list.append(fps)
                accuracy_list.append(accuracy) 
                prediction_img_list.append(prediction_img)           
                time.sleep(0.2)
            
            # Mean values over all images    
            power_idle = sum(power_idle_list) / len(power_idle_list)
            power_run = sum(power_run_list) / len(power_run_list)
            fps = sum(fps_list) / len(fps_list)
            accuracy = sum(accuracy_list) / len(accuracy_list)
            ################################################################################################
            
            data = {}
            data['Task'] = (f"{task}")
            data['FPGA'] = (f"{fpga}")
            data['Topology/Params/OPS'] = (f"{topology}/{params:.4f}M/{flops:.2f}G")
            data['Batch/Model/Data'] = (f"{batch_size}/{model_precision}/{data_precision}")
            data['Patches/s'] = fps                
            data['P idle, [W]'] = f"{power_idle:.2f}"
            data['P run, [W]'] = f"{power_run:.2f}" 
            if task == 'Satellite Image Forest Segmentation':         
                data['F1-score'] = f"{accuracy:.4f}" 
            elif task == 'Radar Image Oil Spill Segmentation': 
                data['IoU'] = f"{accuracy:.4f}" 
            fps_power = fps / power_run
            data['FPS/Power'] = f"{fps_power:.2f}"
            st.session_state.experiments.append(data)
            st.session_state.run_count += 1
            status.update(label="Inference completed!", state="complete")
            
            st.session_state.show_predictions = True
    
    if len(st.session_state.experiments) != 0: 
        st.write(f"Total Experiments Run: {st.session_state.run_count}")  
        # Create a DataFrame
        df = pd.DataFrame(st.session_state.experiments) 
        # Display the table
        #st.table(df)  # Static table
        #st.dataframe(df)  # Interactive table with sorting and filtering    
        # Use st.data_editor for a smoother update experience
        st.data_editor(df, use_container_width=True, disabled=True)  # Keeps the table stable without flashing  
            
    #############################################################################################################################################################
    # Add a horizontal line (separating the next object from previous ones)
    st.markdown('---')
    
    col1, col2, col3 = st.columns([1, 1, 1])  # Three columns: one for input image, one for GT, and one for prediction
    
    # Column headers
    with col1:
        st.markdown("### Input")
    with col2:
        st.markdown("### Ground Truth")
    with col3:
        st.markdown("### Prediction")
    
    # Display images row by row
    for i, (inp, gt) in enumerate(zip(input_images, gt_images)):
        inp_path = os.path.join(input_folder, inp)
        gt_path = os.path.join(gt_folder, gt)        

        if task == 'Radar Image Oil Spill Segmentation':
            gt_img_png = cairosvg.svg2png(url=gt_path)     
            gt_img = Image.open(io.BytesIO(gt_img_png))        

            # Load the input NumPy array from a .npy file
            input_array = np.load(inp_path)  # shape: (100, 100, 9)
            height, width, channels = input_array.shape
            # Flatten the spatial dimensions to apply PCA
            # New shape: (100*100, 9) => (10000, 9)
            flattened_array = input_array.reshape(-1, channels)
            # Apply PCA to reduce 9 channels to 3 (for RGB visualization)
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(flattened_array)
            # Reshape back to original spatial dimensions with 3 channels
            # Shape: (100, 100, 3)
            pca_image = pca_result.reshape(height, width, 3)
            # Normalize the values to range [0, 1] for proper visualization
            normalized_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())
            # Convert to 8-bit unsigned integer [0, 255] for PIL
            image_uint8 = (normalized_image * 255).astype(np.uint8)
            # Create a PIL Image object
            inp_img = Image.fromarray(image_uint8)
        else:     
            inp_img = Image.open(inp_path)
            gt_img = Image.open(gt_path)

        with col1:
            width, height = inp_img.size
            st.image(inp_img, caption=f"{height}x{width} px", use_container_width=True)
        with col2:
            width, height = gt_img.size
            st.image(gt_img, caption=f"{height}x{width} px", use_container_width=True)
        with col3:                    
            if st.session_state.show_predictions:
                pred = prediction_img_list[i]
                if task == 'Radar Image Oil Spill Segmentation':
                    pred = visualize_oil_output(pred, show_fig=True)
                if len(pred.shape) == 3:
                    height, width, _ = pred.shape   
                else:
                    height, width = pred.shape 
                if task == 'Satellite Image Forest Segmentation':
                    st.image(pred, caption=f"{height}x{width} px", use_container_width=True)
                elif task == 'Radar Image Oil Spill Segmentation':
                    st.pyplot(pred)
            else:
                st.empty()
    
        
if __name__ == "__main__":
    main()
