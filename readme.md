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

#### GTX Titan X (Maxwell)

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

#### GTX 1070 

    $ ./x64/Release/atomicAddBench.exe 4 16 65536 0 0
    repeats:    4
    iterations: 16
    threads:    65536
    seed:       0
    Device: GeForce GTX 1070
      pci 0 bus 1
      tcc 0
      SM 61

    float intrinsic
      Value: 522445.593750
      Total  : 8.932384ms
      Average: 2.233096ms

    double intrinsic
      Value: 522496.976176
      Total  : 8.949760ms
      Average: 2.237440ms

    double atomicCAS
      Value: 522496.976176
      Total  : 14708.046875ms
      Average: 3677.011719ms


+ Windows 10 x64
+ Driver 368.39 WDDM
+ CUDA 8.0RC, SM_61


### Ubuntu 16.04.01

#### GTX 1070

    ./x64/Release/atomicAddBench 4 16 65536 0 0 
    repeats:    4
    iterations: 16
    threads:    65536
    seed:       0
    Device: GeForce GTX 1070
      pci 0 bus 1
      tcc 0
      SM 61

    f intrinsic 
      Value: 522445.718750
      Total  : 8.854655ms
      Average: 2.213664ms

    d intrinsic 
      Value: 522496.976176
      Total  : 9.449568ms
      Average: 2.362392ms

    d atomicCAS 
      Value: 522496.976176
      Total  : 16590.242188ms
      Average: 4147.560547ms

+ Ubuntu 16.04.1
+ Driver 367.27
+ CUDA 8.0RC, SM_61
+ GCC 4.9

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
      Value: 522448.375000
      Total  : 8.056353ms
      Average: 2.014088ms

    d intrinsic 
      Value: 522496.976176
      Total  : 8.643200ms
      Average: 2.160800ms

    d atomicCAS 
      Value: 522496.976176
      Total  : 18750.568359ms
      Average: 4687.642090ms

+ Ubuntu 16.04.1
+ Driver 367.27
+ CUDA 8.0RC, SM_61
+ GCC 4.9

#### GTX Titan X (Maxwell)
    
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
      Value: 522451.000000
      Total  : 14.333632ms
      Average: 3.583408ms

    double intrinsic not available SM 5.2

    d atomicCAS 
      Value: 522496.976176
      Total  : 34460.679688ms
      Average: 8615.169922ms

+ Ubuntu 16.04.1
+ Driver 367.27
+ CUDA 8.0RC, SM_52
+ GCC 4.9


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

#### GTX Titan X (Maxwell)

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
