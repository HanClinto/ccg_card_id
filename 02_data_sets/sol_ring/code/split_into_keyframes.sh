#!/bin/bash
# Splits a video file into keyframes with ffmpeg.
# Usage: split_into_keyframes.sh <video_path>
# Output: JPEGs placed in the same directory as the input video.
ffmpeg -skip_frame nokey -i "$1" -vsync vfr -frame_pts true "$1-%04d.jpg"
