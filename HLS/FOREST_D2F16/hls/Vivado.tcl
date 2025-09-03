#------------------------------------------------------------------------------
# Usage check
#------------------------------------------------------------------------------
set start_time [clock seconds]
if {$argc != 3} {
    puts "Usage: vivado -mode batch -source Vivado.tcl -tclargs <block_design_name> <input_interface_name> <output_interface_name>"
    exit
}

#------------------------------------------------------------------------------
# Input arguments
#------------------------------------------------------------------------------

set bd_name [lindex $argv 0]
set input_interface_name [lindex $argv 1]
set output_interface_name [lindex $argv 2]

#------------------------------------------------------------------------------
# Project-wide settings
#------------------------------------------------------------------------------

set PL_FREQ 225
set solution_name "ZCU104"
set proj_name "Vivado"
set part_number "xczu7ev-ffvc1156-2-e"
set script_path [ file  normalize [ info script ] ]
set script_dir [file dirname $script_path ]
set repo_path $script_dir/$bd_name/$solution_name/impl/ip
set_param general.maxThreads 6

create_project $proj_name $script_dir/$proj_name -part $part_number -force
create_bd_design $bd_name
set_property  ip_repo_paths $repo_path [current_project]
update_ip_catalog
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0
create_bd_cell -type ip -vlnv EMS_RPTU:$bd_name:top_level_0:1.2 top_level_0_0
set_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $PL_FREQ [get_bd_cells zynq_ultra_ps_e_0]
set_property -dict [list \
  CONFIG.PSU__SAXIGP0__DATA_WIDTH {128} \
  CONFIG.PSU__SAXIGP1__DATA_WIDTH {128} \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP0 {1} \
  CONFIG.PSU__USE__S_AXI_GP1 {1} \
] [get_bd_cells zynq_ultra_ps_e_0]

apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq_ultra_ps_e_0]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/top_level_0_0/s_axi_control} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins top_level_0_0/s_axi_control]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master "/top_level_0_0/$input_interface_name" Slave {/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}]  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master "/top_level_0_0/$output_interface_name" Slave {/zynq_ultra_ps_e_0/S_AXI_HPC1_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}]  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC1_FPD]
set bd_path [get_files $bd_name\.bd]
open_bd_design $bd_path
validate_bd_design
save_bd_design
make_wrapper -files $bd_path -top
add_files -norecurse $script_dir/$proj_name/$proj_name\.gen/sources_1/bd/$bd_name/hdl/$bd_name\_wrapper.v

launch_runs impl_1 -to_step write_bitstream -jobs 20

wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    error "Implementation failed"
}

file copy -force $script_dir/$proj_name/$proj_name\.runs/impl_1/$bd_name\_wrapper.bit $script_dir/$proj_name/$bd_name.bit
open_bd_design $bd_path
write_bd_tcl -force $script_dir/$proj_name/$bd_name.tcl

open_run impl_1
report_utilization -file $script_dir/$proj_name/Vivado_Utilization.rpt -hierarchical
report_timing_summary -file $script_dir/$proj_name/Vivado_Timing.rpt

set end_time [clock seconds]

# Calculate and print elapsed time in minutes (float)
set elapsed_minutes [expr {($end_time - $start_time) / 60.0}]
puts "Vivado Script finished in $elapsed_minutes minutes."
exit
