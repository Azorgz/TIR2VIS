#!/bin/bash
oarsub -n tir2vis
oarsub -l nodes=2/gpu=6,walltime=01:00:00
oarsub --stdout logs/%jobid%.out
oarsub --stderr logs/%jobid%.err
oarsub --projet pr-remote-sensing-1a

python ~/TIR2VIS/train.py

