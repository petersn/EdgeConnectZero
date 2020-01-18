#!/bin/bash

set -e

#BASE=run10-r=4-f=32-b=6-g=500-v=300-no-distl
#BASE=run12-r=11-f=32-b=12-g=500-v=300-distl
#BASE=run14-r=7-f=32-b=8-g=500-v=800-no-distl
#BASE=run15-r=11-f=32-b=12-g=500-v=600-distl

#BASE=run-cm1-r=11-f=32-b=12-g=500-v=800-no-distl
#BASE=run-cm2-r=11-f=64-b=12-g=500-v=800-no-distl
#BASE=run-cm3-r=11-f=64-b=12-fc=32-g=500-v=800-no-distl-t0
#BASE=run-cm4-r=11-f=64-b=12-fc=32-g=500-v=800-distl-t0
BASE=run-cm5-r=11-f=64-b=12-fc=32-g=500-v=800-distl-t0

mkdir -p $BASE/{games,models}

#make -C cpp clean
#time make -C cpp -j

#time parallel --no-notice -j20 --ungroup python ./generate_games.py --random-play --game-count 250 --output-games ::: /tmp/random-play-{1..20}.json
#cat /tmp/random-play-*.json > $BASE/games/random-play.json

#time python ./train.py --games $BASE/games/random-play.json --steps 5000 --new-path $BASE/models/model-001.npy

#time python looper.py --prefix $BASE/ --game-count 500 --visits 300 --training-steps-const 1600 --training-steps-linear 400

# For run14:
#time python looper.py --prefix $BASE/ --game-count 500 --visits 800 --training-steps-const 800 --training-steps-linear 200

# For run15:
#time python looper.py --prefix $BASE/ --game-count 500 --visits 600 --training-steps-const 1600 --training-steps-linear 400

# For run-cm1
#time python looper.py --prefix $BASE/ --game-count 500 --visits 800 --training-steps-const 800 --training-steps-linear 200

# For run-cm2
#time python looper.py --prefix $BASE/ --game-count 500 --visits 800 --training-steps-const 400 --training-steps-linear 100

# For run-cm3
#time python looper.py --prefix $BASE/ --game-count 500 --visits 800 --training-steps-const 400 --training-steps-linear 100

# For run-cm4
#time python looper.py --prefix $BASE/ --game-count 500 --visits 800 --training-steps-const 400 --training-steps-linear 100

# For run-cm5.
# Run-cm5 is a copy of run-cm4, but with the training per game increased after the first 13 models.
time python looper.py --prefix $BASE/ --game-count 500 --visits 800 --training-steps-const 800 --training-steps-linear 200

