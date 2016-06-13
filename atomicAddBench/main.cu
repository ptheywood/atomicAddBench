
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"

#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include <typeinfo>

#define VERBOSE 0
#define INTEGER_SCALE_FACTOR 100

static void HandleError(const char *file, int line, cudaError_t status = cudaGetLastError()) {
	if (status != cudaSuccess || (status = cudaGetLastError()) != cudaSuccess)
	{
		if (status == cudaErrorUnknown)
		{
			printf("%s(%i) An Unknown CUDA Error Occurred :(\n", file, line);
			exit(1);
		}
		printf("%s(%i) CUDA Error Occurred;\n%s\n", file, line, cudaGetErrorString(status));
		exit(1);
	}
}

#define CUDA_CALL( err ) (HandleError(__FILE__, __LINE__ , err))
#define CUDA_CHECK() (HandleError(__FILE__, __LINE__))


__device__ double atomicAddFP64(double* address, double val)
{
	// https://devtalk.nvidia.com/default/topic/529341/speed-of-double-precision-cuda-atomic-operations-on-kepler-k20/
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}


__global__ void atomicAdd_test(unsigned int numInputs, float * d_inputData, int * d_accumulator, unsigned int * d_start, unsigned int * d_stop){
	unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
	unsigned int start_time = 0;
	unsigned int stop_time = 0;

	if(tid < numInputs){
		start_time = clock();
		atomicAdd(d_accumulator, d_inputData[tid] * INTEGER_SCALE_FACTOR);
		stop_time = clock();

		d_start[tid] = start_time;
		d_stop[tid] = stop_time;
	}
}

__global__ void atomicAdd_test(unsigned int numInputs, float * d_inputData, unsigned int * d_accumulator, unsigned int * d_start, unsigned int * d_stop){
	unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
	unsigned int start_time = 0;
	unsigned int stop_time = 0;

	if(tid < numInputs){
		start_time = clock();
		atomicAdd(d_accumulator, d_inputData[tid] * INTEGER_SCALE_FACTOR);
		stop_time = clock();

		d_start[tid] = start_time;
		d_stop[tid] = stop_time;
	}
}

__global__ void atomicAdd_test(unsigned int numInputs, float * d_inputData, float * d_accumulator, unsigned int * d_start, unsigned int * d_stop){
	unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
	unsigned int start_time = 0;
	unsigned int stop_time = 0;

	if(tid < numInputs){
		start_time = clock();
		atomicAdd(d_accumulator, d_inputData[tid]);
		stop_time = clock();

		d_start[tid] = start_time;
		d_stop[tid] = stop_time;
	}
}

__global__ void atomicAdd_test(unsigned int numInputs, double * d_inputData, double * d_accumulator, unsigned int * d_start, unsigned int * d_stop){
	unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
	unsigned int start_time = 0;
	unsigned int stop_time = 0;

	if(tid < numInputs){
		start_time = clock();
#if CUDA_VERSION >= 8000 && __CUDA_ARCH__ >= 600
		atomicAdd(d_accumulator, d_inputData[tid]);
#else 
		atomicAddFP64(d_accumulator, d_inputData[tid]);
#endif
		stop_time = clock();

		d_start[tid] = start_time;
		d_stop[tid] = stop_time;
	}
}

void generateInputData(unsigned int numInputs, unsigned long long int seed, float * d_data){

	curandGenerator_t rng = NULL;

	// Create RNG, seed RNG and populate device array.
	curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT); // @todo - curand error check
	curandSetPseudoRandomGeneratorSeed(rng, seed); // @todo - curand error check

	curandGenerateUniform(rng, d_data, numInputs); // @todo - curand error check
	
	// Cleanup rng
	curandDestroyGenerator(rng); // @todo - curand error check

}

void generateInputData(unsigned int numInputs, unsigned long long int seed, double * d_data){

	curandGenerator_t rng = NULL;

	// Create RNG, seed RNG and populate device array.
	curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT); // @todo - curand error check
	curandSetPseudoRandomGeneratorSeed(rng, seed); // @todo - curand error check

	curandGenerateUniformDouble(rng, d_data, numInputs); // @todo - curand error check
	
	// Cleanup rng
	curandDestroyGenerator(rng); // @todo - curand error check

}


void printAccumulatorTotal(int v){
	fprintf(stdout, "Accumulator: %d\n", v);
	fflush(stdout);
}
void printAccumulatorTotal(long long int v){
	fprintf(stdout, "Accumulator: %ll\n", v);
	fflush(stdout);
}
void printAccumulatorTotal(unsigned int v){
	fprintf(stdout, "Accumulator: %u\n", v);
	fflush(stdout);
}
void printAccumulatorTotal(unsigned long long int v){
	fprintf(stdout, "Accumulator: %llu\n", v);
	fflush(stdout);
}
void printAccumulatorTotal(float v){
	fprintf(stdout, "Accumulator: %f\n", v);
	fflush(stdout);
}
void printAccumulatorTotal(double v){
	fprintf(stdout, "Accumulator: %f\n", v);
	fflush(stdout);
}

template <typename T, typename U>
void runAtomicAddTest(unsigned int numIterations, unsigned int numInputs, unsigned long long int seed){

	T *h_accumulator = (T*)malloc(1 * sizeof(T));
	U *h_inputData = (U*)malloc(numInputs * sizeof(U)); //@todo - unneeded?

	T *d_accumulator = NULL;
	U *d_inputData = NULL;

	unsigned int *h_start = (unsigned int*)malloc(numInputs * sizeof(unsigned int));
	unsigned int *h_stop = (unsigned int*)malloc(numInputs * sizeof(unsigned int));
	
	unsigned int *d_start = NULL;
	unsigned int *d_stop = NULL;

	fprintf(stdout, "atomicAdd(%s) RNG(%s) %d iterations\n", typeid(*h_accumulator).name(), typeid(*h_inputData).name(), numIterations);
	fflush(stdout);

	// Allocate device data.
	CUDA_CALL(cudaMalloc((void**)&d_accumulator, 1 * sizeof(T)));
	CUDA_CALL(cudaMalloc((void**)&d_inputData, numInputs * sizeof(U)));
	CUDA_CALL(cudaMalloc((void**)&d_start, numInputs * sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc((void**)&d_stop, numInputs * sizeof(unsigned int)));


	// Initialise accumulator
	(*h_accumulator) = (T)0.0;
	CUDA_CALL(cudaMemcpy(d_accumulator, h_accumulator, 1 * sizeof(T), cudaMemcpyHostToDevice));

	// Generate random data
	generateInputData(numInputs, seed, d_inputData);

	// Get a function pointer to the kernel for this data type.
	void (*kernel)(unsigned int, U*, T*, unsigned int*, unsigned int*) = atomicAdd_test;

	// Accumulate values via kernel.
	int blockSize, minGridSize, gridSize;
	CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, numInputs));
	gridSize = (numInputs + blockSize - 1) / blockSize;
	kernel<<<gridSize, blockSize>>>(numInputs, d_inputData, d_accumulator, d_start, d_stop);
	CUDA_CHECK();

	// Copy Data from Device to Host
	CUDA_CALL(cudaMemcpy(h_accumulator, d_accumulator, 1 * sizeof(T), cudaMemcpyDeviceToHost));
	//CUDA_CALL(cudaMemcpy(h_inputData, d_inputData, numInputs * sizeof(U), cudaMemcpyDeviceToHost)); //@todo - remove?
	CUDA_CALL(cudaMemcpy(h_start, d_start, numInputs * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(h_stop, d_stop, numInputs * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
#if defined(VERBOSE) && VERBOSE > 0

	// Find the minimum start time.
	unsigned int min = h_start[0];
	for(unsigned int i = 0; i < numInputs; i++){
		min = (h_start[i] < min) ? h_start[i] : min;
	}

	fprintf(stdout, "thread, warp, start, stop\n");
	for(unsigned int i = 0; i < numInputs; i++){
		if (i > 0 && i % 32 == 0){
			fprintf(stdout, "---------------\n");
		}
		fprintf(stdout, "%d, %d, %u, %u\n", i % 32, i / 32, h_start[i] - min, h_stop[i] - min); 
	}
	fflush(stdout);
#endif 

	// Print output messages
	printAccumulatorTotal(h_accumulator[0]);

	// Free device data
	CUDA_CALL(cudaFree(d_accumulator));
	CUDA_CALL(cudaFree(d_inputData));
	CUDA_CALL(cudaFree(d_start));
	CUDA_CALL(cudaFree(d_stop));

	// Free host data
	free(h_accumulator);
	free(h_inputData);
	free(h_start);
	free(h_stop);

	// Reset the device for profiler output.
	CUDA_CALL(cudaDeviceReset());
}

int main()
{
	runAtomicAddTest<float, float>(1, 128, 0);
	//runAtomicAddTest<double, double>(1, 128, 0);
	//runAtomicAddTest<int, float>(1, 128, 0);
	//runAtomicAddTest<long long int, float>(1, 128, 0);


    return 0;
}
