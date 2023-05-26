#!/usr/local/bin/bash

# copy files from a directorty to the current directory hear but replatce "test" with "test2" in filename and remove directory using basename

PATTERN=$1
ORIG=$2
NEW=$3


for file in $PATTERN; do
    cp $file $(basename $file | sed 's/$orig/$new/')
done
for 