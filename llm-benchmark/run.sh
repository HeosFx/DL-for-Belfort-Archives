#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python run.py > my.log 2>&1 &
echo $! > save_pid.txt
pstree -p $! | grep -oP '\(\d+\)' | grep -oP '\d+' >> save_pid.txt