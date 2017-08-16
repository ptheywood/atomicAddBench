#!/bin/bash 
#$ -l gpu=1 -l rmem=16G -j y -o ~/atomicAddBench/log.txt -P rse -q rse.q 

module load libs/CUDA/8.0.44/binary

nvidia-smi

echo $CUDA_VISIBLE_DEVICES

pushd  /home/$USER/atomicAddBench/

mkdir -p /home/$USER/atomicAddBench/
mkdir -p  /data/$USER/atomicAddBench/
mkdir -p  /fastdata/$USER/atomicAddBench/

./atomicAddBench 4 16 65536 0 0
nvprof -f -o /home/$USER/atomicAddBench/atomicAddBench-home.nvvp ./x64/Release/atomicAddBench 4 16 65536 0 0
nvprof -f -o /data/$USER/atomicAddBench/atomicAddBench-data.nvvp ./x64/Release/atomicAddBench 4 16 65536 0 0
nvprof -f -o /fastdata/$USER/atomicAddBench/atomicAddBench-fastdata.nvvp ./x64/Release/atomicAddBench 4 16 65536 0 0
