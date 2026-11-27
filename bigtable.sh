#!/bin/bash

# Define the options for the synth_mask parameter
options_synth_mask=("C" "M" "F")
options_encoding=("std")
options_dataset=("AustraliaTourism" "MetroTraffic" "BeijingAirQuality" "RossmanSales" "PanamaEnergy")
# Loop through all synth_mask options and run the Python script with each one
for dataset in "${options_dataset[@]}"
do
  for synth_mask in "${options_synth_mask[@]}"
  do
    #python3.12 synthesis_tsdiff.py -d $dataset -synth_mask $synth_mask -strength 0.0
    #python3.12 synthesis_tsdiff.py -d $dataset -synth_mask $synth_mask -strength 0.5
    #python3.12 synthesis_tsdiff.py -d $dataset -synth_mask $synth_mask -strength 1.0
    #python3.12 synthesis_tsdiff.py -d $dataset -synth_mask $synth_mask -strength 2.0
    #python3.12 synthesis_hyacinth_pipeline.py -d $dataset -synth_mask $synth_mask
    #python3.12 synthesis_timeweaver.py -d $dataset -synth_mask $synth_mask
#    python3.12 synthesis_timegan.py -d $dataset -synth_mask $synth_mask
    python3.12 synthesis_tsdiff.py -d $dataset -synth_mask $synth_mask -strength 0.5
    python3.12 synthesis_timeweaver.py -d $dataset -synth_mask $synth_mask
    python3.12 synthesis_timegan.py -d $dataset -synth_mask $synth_mask
    python3.12 synthesis_wavestitch_pipeline_strided_preconditioning.py -d $dataset -synth_mask $synth_mask -stride 8
    #python3.12 synthesis_hyacinth_autoregressive.py -d $dataset -synth_mask $synth_mask -stride 32
    python3.12 synthesis_timeautodiff.py -d $dataset -synth_mask $synth_mask
    python3.12 synthesis_sssd.py -d $dataset -synth_mask $synth_mask
  done
done