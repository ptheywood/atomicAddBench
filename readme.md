# atomicAddBench

Micro-benchmark to compare CUDA `atomicAdd()` performance for a range of data types for different hardware architectures.


## Benchmarks


### GTX 1080

    $ ./Release/atomicAddBench.exe 16 65536 0 0
    iterations: 16
    threads:    65536
    seed:       0
    Device: GeForce GTX 1080
      pci 0
      bus 1
      tcc 0

    float   intrinsic    2.061312ms Accumulator: 522448.187500
    double  intrinsic    2.063360ms Accumulator: 522496.976176
    double  atomicCAS 4603.895020ms Accumulator: 522496.976176



