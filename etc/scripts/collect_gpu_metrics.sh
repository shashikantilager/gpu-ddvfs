#!/usr/bin/env bash

#echo arguments to scripts are $1 $2 $3 
deviceId=$1
interval=$2
dir=$3
filename=$4


echo started collecting
echo output directory $dir/$filename
# nvidia dmon collector
#nvidia-smi dmon   -d  $interval  -f $dir/$filename
#nvidia-smi dmon   -d  $interval >> $dir/$filename
nvidia-smi dmon -i $deviceId -s pucvmet   -d  $interval >> $dir/$filename

echo finsihed collecting
