ffmpeg -i IMG_1938.MOV -vf "scale=640:360" -r 30 -t 3 -c:v libx264 -crf 16 -c:a aac -b:a 64k -preset veryslow inputs/ebu7240_hand_2.mp4

ffmpeg -i IMG_1938.MOV -vf "scale=640:360,transpose=1" -r 30 -t 3 -c:v libx264 -crf 16 -c:a aac -b:a 64k -preset veryslow inputs/ebu7240_hand.mp4