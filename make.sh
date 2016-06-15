nvcc -gencode arch=compute_52,code=compute_52 -gencode arch=compute_61,code=compute_61 -std=c++11 -O2 -m64 -I include -lcurand -o x64/Release/atomicAddBench atomicAddBench/main.cu 
