#!/bin/bash
rm -rf frameOut/*.*
rm -rf test/*.*
echo "Video is under processing"
python label_image.py --NoDisplay --pathIn $1
python displayImage.py
echo "Video Successfully being processed"
today=`date '+%Y_%m_%d__%H_%M_%S'`;

ffmpeg -r 8 -i frameOut/f%d.jpg  frameOut/$today.mp4
echo "Success !!! "
cp -rf frameOut/$today.mp4 ProjectVideoOutputs/
