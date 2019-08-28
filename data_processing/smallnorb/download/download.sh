#!/bin/bash

# STORAGE=${1:-"./"}
# DATADIR="$STORAGE/data/smallNORB/"
DATADIR="./data/smallNORB/mat"
echo $DATADIR
mkdir -p $DATADIR

while read url ; do
    echo "fetching $url"
    wget -P "$DATADIR/" "$url"
done < "./url.txt"

echo "Done fetching archive files. Extracting..."


for gzfile in $DATADIR/*.gz ; do
   gzip -d "$gzfile"
done

echo "Done!"