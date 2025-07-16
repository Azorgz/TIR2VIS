#!/bin/bash
#OAR -n tir2vis
#OAR  -l nodes=2/gpu=12,walltime=05:00:00
#OAR --stdout logs/%jobid%.out
#OAR --stderr logs/%jobid%.err
#OAR --projet pr-remote-sensing-1a

conda activate Aurelien_torch
python ~/TIR2VIS/train.py

