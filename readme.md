# atomicAddBench

Micro-benchmark to compare CUDA `atomicAdd()` performance for a range of data types for different hardware architectures.


## Benchmarks

### Windows

#### GTX 1080

    $ ./x64/Release/atomicAddBench.exe 4 16 65536 0 0
    repeats:    4
    iterations: 16
    threads:    65536
    seed:       0
    Device: GeForce GTX 1080
      pci 0 bus 1
      tcc 0
      SM 61

    float intrinsic
      Value: 522451.218750
      Total  : 13.884416ms
      Average: 3.471104ms

    double intrinsic
      Value: 522496.976176
      Total  : 13.852673ms
      Average: 3.463168ms

    double atomicCAS
      Value: 522496.976176
      Total  : 38954.253906ms
      Average: 9738.563477ms


+ Windows 8.1 x64
+ Driver 368.39 WDDM
+ CUDA 8.0RC, SM_61
+ *NOTE* - Performance on Windows 8.1 seems low for 1080 with CUDA 8.0RC, being investigated

#### GTX Titan X

    $ ./x64/Release/atomicAddBench.exe 4 16 65536 0 1
    repeats:    4
    iterations: 16
    threads:    65536
    seed:       0
    Device: GeForce GTX TITAN X
      pci 0 bus 2
      tcc 0
      SM 52

    float intrinsic
      Value: 522456.187500
      Total  : 14.463680ms
      Average: 3.615920ms

    double intrinsic not available SM 5.2

    double atomicCAS
      Value: 522496.976176
      Total  : 35176.460938ms
      Average: 8794.115234ms


+ Windows 8.1 x64
+ Driver 368.39 WDDM
+ CUDA 8.0RC, SM_52

#### GTX 1070 (@todo - Update)

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
+ Driver 368.39 WDDM
+ CUDA 8.0RC, SM_61

### Ubuntu 14.04

#### GTX 1080

    ./x64/Release/atomicAddBench 4 16 65536 0 0
    repeats:    4
    iterations: 16
    threads:    65536
    seed:       0
    Device: GeForce GTX 1080
      pci 0 bus 1
      tcc 0
      SM 61

    f intrinsic 
      Value: 522449.375000
      Total  : 8.057217ms
      Average: 2.014304ms

    d intrinsic 
      Value: 522496.976176
      Total  : 8.052192ms
      Average: 2.013048ms

    d atomicCAS 
      Value: 522496.976176
      Total  : 19113.062500ms
      Average: 4778.265625ms


+ Ubuntu 14.04
+ Driver 367.27
+ CUDA 8.0RC, SM_61

#### GTX Titan X

    ./x64/Release/atomicAddBench 4 16 65536 0 1
    repeats:    4
    iterations: 16
    threads:    65536
    seed:       0
    Device: GeForce GTX TITAN X
      pci 0 bus 2
      tcc 0
      SM 52

    f intrinsic 
      Value: 522453.718750
      Total  : 14.321280ms
      Average: 3.580320ms

    double intrinsic not available SM 5.2

    d atomicCAS 
      Value: 522496.976176
      Total  : 34452.035156ms
      Average: 8613.008789ms



+ Ubuntu 14.04
+ Driver 367.27
+ CUDA 8.0RC, SM_52
