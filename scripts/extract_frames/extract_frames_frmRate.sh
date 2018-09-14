#!/bin/bash
#1:File address,2:frm_rate,3:save_path

EXPECTED_ARGS=3
E_BADARGS=65

if [ $# -lt $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` video frames/sec [size=256] save path"
  exit $E_BADARGS
fi

NAME=${1%.*}
FRAMES=$2
BNAME=$3
#echo $BNAME
#mkdir -m 755 $BNAME
#ffmpeg -y -i $1 -r $FRAMES $3_%4d.jpg

ffmpeg -i $1 -qscale:v 2 -r $FRAMES  $3_%4d.jpg
