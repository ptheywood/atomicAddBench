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

+ Windows 8.1 x64
+ Geforce Driver 368.39 WDDM
+ CUDA 8.0RC, SM_61

### GTX 1070

    $ ./Release/atomicAddBench.exe 16 65536 0 0
    iterations: 16
    threads:    65536
    seed:       0
    Device: GeForce GTX 1070
      pci 0
      bus 1
      tcc 0

    float   intrinsic    2.230272ms Accumulator: 522444.343750
    double  intrinsic    2.223104ms Accumulator: 522496.976176
    double  atomicCAS 3194.879883ms Accumulator: 522496.976176

+ Windows 10 x64
+ Geforce Driver 368.39 WDDM
+ CUDA 8.0RC, SM_61
