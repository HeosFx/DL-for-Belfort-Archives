#!/bin/bash

nohup python tune_yolo.py > my.log 2>&1 &
echo $! > save_pid.txt
pstree -p $! | grep -oP '\(\d+\)' | grep -oP '\d+' >> save_pid.txt