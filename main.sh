#!/bin/bash
#OAR -n tir2vis
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR -p gpumodel='A100'
#OAR --stdout logs/print.out
#OAR --stderr logs/error.err
#OAR --project pr-remote-sensing-1a

python train.py