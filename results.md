# Results

Below are the raw output of the results for the machines in use, including details of driver version etc.

## Windows 10

### Titan X (Pascal) WDDM

    $ ./x64/Release/atomicAddBench.exe 4 16 65536 0 0
    repeats:    4
    iterations: 16
    threads:    65536
    seed:       0
    Device: TITAN X (Pascal)
      pci 0 bus 2
      tcc 0
      SM 61

    float intrinsic
      Value: 522494.343750
      Total  : 0.412672ms
      Average: 0.103168ms

    double intrinsic
      Value: 522496.976176
      Total  : 9.146336ms
      Average: 2.286584ms

    double atomicCAS
      Value: 522496.976176
      Total  : 18571.289063ms
      Average: 4642.822266ms



+ Windows 10 x64
+ Driver 372.54 WDDM
+ CUDA 8.0RC, SM_61
+ `make.sh`

### GTX 1080

  $ ./x64/Release/atomicAddBench.exe 4 16 65536 0 1
  repeats:    4
  iterations: 16
  threads:    65536
  seed:       0
  Device: GeForce GTX 1080
    pci 0 bus 1
    tcc 0
    SM 61

  float intrinsic
    Value: 522494.093750
    Total  : 0.479232ms
    Average: 0.119808ms

  double intrinsic
    Value: 522496.976176
    Total  : 8.552448ms
    Average: 2.138112ms

  double atomicCAS
    Value: 522496.976176
    Total  : 11832.431641ms
    Average: 2958.107910ms

+ Windows 10 x64
+ Driver 372.54 WDDM
+ CUDA 8.0RC, SM_61
+ `make.sh`


### GTX 1070

    $ ./x64/release/atomicAddBench.exe 4 16 65536 0 0
    repeats:    4
    iterations: 16
    threads:    65536
    seed:       0
    Device: GeForce GTX 1070
      pci 0 bus 1
      tcc 0
      SM 61

    float intrinsic
      Value: 522492.281250
      Total  : 0.377824ms
      Average: 0.094456ms

    double intrinsic
      Value: 522496.976176
      Total  : 9.191424ms
      Average: 2.297856ms

    double atomicCAS
      Value: 522496.976176
      Total  : 14477.271484ms
      Average: 3619.317871ms



+ Windows 10 x64
+ Driver 372.70 WDDM
+ CUDA 8.0RC, SM_61

## Windows 8

### Titan X (Pascal)

    $ ./x64/Release/atomicAddBench.exe 4 16 65536 0 0
    repeats:    4
    iterations: 16
    threads:    65536
    seed:       0
    Device: TITAN X (Pascal)
      pci 0 bus 2
      tcc 0
      SM 61

    float intrinsic
      Value: 522464.406250
      Total  : 9.128961ms
      Average: 2.282240ms

    double intrinsic
      Value: 522496.976176
      Total  : 9.133088ms
      Average: 2.283272ms

    double atomicCAS
      Value: 522496.976176
      Total  : 18789.156250ms
      Average: 4697.289063ms


+ Windows 8.1 x64
+ Driver 372.54 WDDM
+ CUDA 8.0RC, SM_61
+ MSVC

### GTX 1080

    $ ./x64/Release/atomicAddBench.exe 4 16 65536 0 1
    repeats:    4
    iterations: 16
    threads:    65536
    seed:       0
    Device: GeForce GTX 1080
      pci 0 bus 1
      tcc 0
      SM 61

    float intrinsic
      Value: 522453.343750
      Total  : 8.431616ms
      Average: 2.107904ms

    double intrinsic
      Value: 522496.976176
      Total  : 8.443904ms
      Average: 2.110976ms

    double atomicCAS
      Value: 522496.976176
      Total  : 18491.679688ms
      Average: 4622.919922ms


+ Windows 8.1 x64
+ Driver 372.54 WDDM
+ CUDA 8.0RC, SM_61
+ MSVC

### GTX Titan X (Maxwell)

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
+ MSVC

## Ubuntu 16.04.01

### GTX 1070

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
+ make.sh

### GTX 1080

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
+ make.sh

### Titan X (Pascal)
    
    ../x64/Release/atomicAddBench 4 16 65536 0 0
    repeats:    4
    iterations: 16
    threads:    65536
    seed:       0
    Device: TITAN X
      pci 0 bus 2
      tcc 0
      SM 61

    f intrinsic 
      Value: 522462.718750
      Total  : 9.172672ms
      Average: 2.293168ms

    d intrinsic 
      Value: 522496.976176
      Total  : 9.173632ms
      Average: 2.293408ms

    d atomicCAS 
      Value: 522496.976176
      Total  : 18814.171875ms
      Average: 4703.542969ms

+ Ubuntu 16.04.1
+ Driver 367.27
+ CUDA 8.0RC, SM_52
+ GCC 4.9
+ make.sh

### GTX Titan X (Maxwell)
    
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
+ make.sh


## Ubuntu 14.04

### GTX 1080

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
+ make.sh

### GTX Titan X (Maxwell)

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
+ make.sh
