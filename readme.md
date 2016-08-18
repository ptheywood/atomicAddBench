# atomicAddBench

Micro-benchmark to compare CUDA `atomicAdd()` performance for a range of data types for different hardware architectures.


## Benchmarks

```
./x64/Release/atomitAddBench.exe <repeats> <iterations> <threads> <device_id>
```
with values

| Argument   | Value |
| ---------- | ----- |
| Repeats    |     4 |
| Iterations |    16 |
| threads    | 65536 |
| device_id  |   0/1 |

i.e. `./x64/Release/atomicAddBench.exe 4 16 65536 0 0`


### Windows 10

| Device            | Driver  | Build System | FP32 atomicAdd | FP64 atomicAdd | FP64 atomicCAS |
| ----------------  | ------- | ------------ | -------------- | -------------- | -------------- |
| TITAN X (Pascal)  |  372.54 | make.sh      |    0.103168 ms |    2.286584 ms | 4642.822266 ms |
| GTX 1080          |  372.54 | make.sh      |    0.119808 ms |    2.138112 ms | 2958.107910 ms |
| GTX 1070          |  372.54 | MSVC         |    2.233096 ms |    2.237440 ms | 3677.011719 ms |

### Windows 8
| Device            | Driver  | Build System | FP32 atomicAdd | FP64 atomicAdd | FP64 atomicCAS |
| ----------------  | ------- | ------------ | -------------- | -------------- | -------------- |
| TITAN X (Pascal)  |  372.54 | MSVC         |    2.282240 ms |    2.283272 ms | 4697.289063 ms |
| GTX 1080          |  372.54 | MSVC         |    2.107904 ms |    2.110976 ms | 4622.919922 ms |
| Titan X (Maxwell) |  368.39 | MSVC         |    3.615920 ms |            N/A | 8794.115234 ms |

### Ubuntu 16.04.01
| Device            | Driver  | Build System | FP32 atomicAdd | FP64 atomicAdd | FP64 atomicCAS |
| ----------------  | ------- | ------------ | -------------- | -------------- | -------------- |
| TITAN X (Pascal)  |  367.27 | make.sh      |    2.293168 ms |    2.293408 ms | 4703.542969 ms |
| GTX 1080          |  367.27 | make.sh      |    2.014088 ms |    2.160800 ms | 4687.642090 ms |
| GTX 1070          |  367.27 | make.sh      |    2.213664 ms |    2.362392 ms | 4147.560547 ms |
| Titan X (Maxwell) |  367.27 | make.sh      |    3.583408 ms |            N/A | 8615.169922 ms |


Further details can be found in [results.md](results.md)
