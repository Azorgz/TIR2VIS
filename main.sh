#!/bin/bash
#OAR -n tir2vis
#OAR -l /nodes=1/gpu=1,walltime=02:00:00
#OAR -p gpumodel='V100'
#OAR --stdout logs/%jobid%.out
#OAR --stderr logs/%jobid%.err
#OAR --project pr-remote-sensing-1a
python ~/TIR2VIS/train.py
