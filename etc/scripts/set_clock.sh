#!/bin/sh


echo setting application clocks 
deviceId=$1
memory_clock=$2
graphics_clock=$3
echo command line argumets to set clock scripts are:  $deviceId $memory_clock and $graphics_clock
sudo nvidia-smi -i $deviceId -pm 1
sudo nvidia-smi -i $deviceId -ac  $memory_clock,$graphics_clock

echo clock have been set
