#!/bin/bash
#OAR -n tir2vis
#OAR -l /nodes=1/gpu=1 --project pr-remote-sensing-1a -p "gpumodel='A100'" "nvidia-smi -L"
#OAR --stdout logs/%jobid%.out
#OAR --stderr logs/%jobid%.err
python ~/TIR2VIS/train.py
