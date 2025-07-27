#!/bin/bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
HDF5=$4
CHUNK_SIZE=$5
OUTFILE=$6

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    $(dirname "$0")/test_chunks.py \
    $CONFIG \
    $CHECKPOINT \
    --hdf5 $HDF5 \
    --chunk_size $CHUNK_SIZE \
    --out $OUTFILE
