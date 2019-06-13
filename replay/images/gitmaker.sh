#!/bin/sh

palette="/tmp/palette.png"

filters="fps=30,scale=350:-1:lanczos"

ffmpeg -v warning -i $1 -vf "$filterspalettegen=stats_mode=diff" -y $palette

ffmpeg -i $1 -i $palette -lavfi "$filters,paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -y $2

