#!/bin/bash

filename=$1
## store the first row to use it as header 

header_data=$(head -1 $filename)
echo  $header_data

#remove all the strings in the files that are inserted after each screen length by dmon
sed -i '/Idx/d' $filename
sed -i '/gpu/d' $filename

#insert header back to first line
awk -v x="$header_data" 'NR==1{print x} 1' $filename >tmp && mv tmp $filename

## convert the file to CSV format with text transform- It replaces tab and other spaces to ","
tr -s ' ' < $filename | tr ' ' ',' > tmp && mv tmp $filename
#the first column is redudant after the transformation, we cut it
cut --complement -d , -f 1 $filename > tmp && mv tmp $filename

echo processing $filename finished
