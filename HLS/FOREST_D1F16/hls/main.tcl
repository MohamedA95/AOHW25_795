delete_project project
open_project project

add_files src/dnn_top_level_wrapper.cpp -cflags "-Iinclude_hls -Iinclude_finn"
add_files -tb src/main.cpp -cflags "-Iinclude_hls -Iinclude_finn"

set_top top_level_0

open_solution ZCU104 -flow_target vivado
config_compile -no_signed_zeros
set_part "xczu7ev-ffvc1156-2-e"
create_clock -period 225MHz -name default

#csim_design -clean -O
csynth_design
#cosim_design -O -enable_dataflow_profiling 
export_design -description project -display_name project -library project -vendor EMS_RPTU -version 1.2
