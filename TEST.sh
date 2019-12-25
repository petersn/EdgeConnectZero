#!/bin/bash

time python3 uai_ringmaster.py \
	--tc 100 \
	--pgn-out /tmp/is-distillation-good-v=400.pgn \
	--show-games \
	--engine "python3 ./uai_interface.py --visits 400 --network-path run9-r\=4-f\=32-b\=6-g\=500-v\=300-distl/models/model-020.npy" \
	--engine "python3 ./uai_interface.py --visits 400 --network-path run10-r\=4-f\=32-b\=6-g\=500-v\=300-no-distl/models/model-020.npy"


#time python3 uai_ringmaster.py \
#	--pgn-out evals/turbo_tc5_model-173.pgn \
#	--tc 5 \
#	--max-plies 400 \
#	--show-games \
#	--engine "tiktaxx" \
#	--engine "python3 ./uai_interface.py --network-path /tmp/model-173.npy"

