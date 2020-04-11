#!/bin/sh
##TODO all monitoring  proceeses should be included
## Kill the nvidia dmon process
echo killing dmon procesees
process_ids=$(ps ax | grep "nvidia-smi dmon" | grep -v grep | awk  '{print $1}')
#convert into array
process_ids=$(echo $process_ids | tr " " "\n")
for process in $process_ids
do
    kill -9 $process
    echo "$process process has been killed"
done

echo finished
