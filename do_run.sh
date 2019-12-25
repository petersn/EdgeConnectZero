#!/bin/bash

BASE=run9-r=4-f=32-b=6-g=500-v=300-distl

mkdir -p $BASE/{games,models}

make -C cpp clean
time make -C cpp -j

time python ./generate_games.py --random-play --output-games $BASE/games/random-play.json --game-count 2000

time python ./train.py --games $BASE/games/random-play.json --steps 2000 --new-path $BASE/models/model-001.npy

time python looper.py --prefix $BASE/ --game-count 500 --visits 300

