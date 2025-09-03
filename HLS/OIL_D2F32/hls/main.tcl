delete_project UNET_2F_2S_32FPS
open_project UNET_2F_2S_32FPS

add_files src/dnn_top_level_wrapper.cpp -cflags "-Iinclude_hls -Iinclude_finn"
add_files -tb src/main.cpp -cflags "-Iinclude_hls -Iinclude_finn"

set_top top_level_0

open_solution ZCU104 -flow_target vivado
config_compile -no_signed_zeros -enable_auto_rewind=false -pipeline_style flp
set_part "xczu7ev-ffvc1156-2-e"
create_clock -period 225MHz -name default

#csim_design -O
csynth_design
#cosim_design
export_design -description UNET_2F_2S_32FPS -display_name UNET_2F_2S_32FPS -library UNET_2F_2S_32FPS -vendor EMS_RPTU -version 1.2
