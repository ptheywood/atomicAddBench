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
| GTX 1070          |  372.54 | make.sh      |    0.094456 ms |    2.297856 ms | 3619.317871 ms |
| GTX 1070          |  372.54 | MSVS         |    2.233096 ms |    2.237440 ms | 3677.011719 ms |

### Windows 8
| Device            | Driver  | Build System | FP32 atomicAdd | FP64 atomicAdd | FP64 atomicCAS |
| ----------------  | ------- | ------------ | -------------- | -------------- | -------------- |
| TITAN X (Pascal)  |  372.54 | MSVS         |    2.282240 ms |    2.283272 ms | 4697.289063 ms |
| GTX 1080          |  372.54 | MSVS         |    2.107904 ms |    2.110976 ms | 4622.919922 ms |
| Titan X (Maxwell) |  368.39 | MSVS         |    3.615920 ms |            N/A | 8794.115234 ms |

### Linux
| Device            | Driver  | Build System | FP32 atomicAdd | FP64 atomicAdd | FP64 atomicCAS  |
| ----------------  | ------- | ------------ | -------------- | -------------- | --------------- |
| TITAN X (Pascal)  | 367.27  | make.sh      |    2.293168 ms |    2.293408 ms |  4703.542969 ms |
| GTX 1080          | 367.27  | make.sh      |    2.014088 ms |    2.160800 ms |  4687.642090 ms |
| GTX 1070          | 367.27  | make.sh      |    2.213664 ms |    2.362392 ms |  4147.560547 ms |
| Titan X (Maxwell) | 367.27  | make.sh      |    3.583408 ms |    2.777600 ms |  7704.697754 ms |
| TITAN X (Pascal)  | 418.40  | make.sh      |    2.107424 ms |    2.315024 ms |  5077.749512 ms |
| Titan V           | 418.40  | make.sh      |    2.775552 ms |    2.777600 ms |  7704.697754 ms |
| A100-SXM4-80GB    | 510.47 (11.4) | make.sh|    2.600448 ms |    2.600192 ms |  7502.596191 ms |
| Tesla V100 PCIE 32| 418.67  | make.sh      |    2.731520 ms |    2.725120 ms |  9538.770508 ms |
| Tesla P100        | 418.40  | make.sh      |    2.203728 ms |    2.201264 ms |  7350.522949 ms |
| Tesla K80         | 410.104 | make.sh      |    3.764568 ms |            N/A | 17993.359375 ms |



Further details can be found in [results.md](results.md)
