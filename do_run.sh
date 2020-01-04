#!/bin/bash

#BASE=run10-r=4-f=32-b=6-g=500-v=300-no-distl
#BASE=run12-r=11-f=32-b=12-g=500-v=300-distl
#BASE=run14-r=7-f=32-b=8-g=500-v=800-no-distl
BASE=run15-r=11-f=32-b=12-g=500-v=600-distl

mkdir -p $BASE/{games,models}

make -C cpp clean
time make -C cpp -j

time python ./generate_games.py --random-play --output-games $BASE/games/random-play.json --game-count 2000

time python ./train.py --games $BASE/games/random-play.json --steps 2000 --new-path $BASE/models/model-001.npy

#time python looper.py --prefix $BASE/ --game-count 500 --visits 300 --training-steps-const 1600 --training-steps-linear 400

# For run14:
#time python looper.py --prefix $BASE/ --game-count 500 --visits 800 --training-steps-const 800 --training-steps-linear 200

#For run15:
time python looper.py --prefix $BASE/ --game-count 500 --visits 600 --training-steps-const 1600 --training-steps-linear 400


