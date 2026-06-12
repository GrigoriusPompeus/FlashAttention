#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH

echo "Compiling..."
nvcc -O2 -o attention_gpu attention_gpu.cu
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

echo "Running GPU experiment..."
./attention_gpu
