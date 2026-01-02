#code
open_project energy_anomaly_fpga
set_top anomaly_detector
add_files anomaly_detector.cpp
add_files -tb anomaly_detector_tb.cpp
open_solution "solution1"
set_part {xc7z020clg400-1}   ;# PYNQ-Z2
create_clock -period 10
csim_design
csynth_design
export_design -format ip_catalog
exit
