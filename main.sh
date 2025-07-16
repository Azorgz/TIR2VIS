#!/bin/bash
#OAR -n tir2vis
#OAR -l /nodes=1/gpu=1 --project test -p "gpumodel='A100'" "nvidia-smi -L"
#OAR --stdout logs/%jobid%.out
#OAR --stderr logs/%jobid%.err
#OAR --projet pr-remote-sensing-1a
python ~/TIR2VIS/train.py

