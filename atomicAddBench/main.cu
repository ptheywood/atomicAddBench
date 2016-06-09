
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"

#include "stdlib.h"
#include "stdio.h"
#include "time.h"



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


__global__ void atomicAdd_test(unsigned int numInputs, const float * __restrict__ d_inputData, float * d_accumulator, unsigned int * d_start, unsigned int * d_stop){
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

void runAtomicAddTest(unsigned int numIterations, unsigned int numInputs, unsigned long long int seed){
	fprintf(stdout, "Running %d iteration\n", numIterations);
	fflush(stdout);

	float * h_accumulator = (float *)malloc(1 * sizeof(float));
	float * h_inputData = (float*)malloc(numInputs * sizeof(float)); //@todo - unneeded?

	unsigned int * h_start = (unsigned int*)malloc(numInputs * sizeof(unsigned int));
	unsigned int * h_stop = (unsigned int*)malloc(numInputs * sizeof(unsigned int));


	float * d_accumulator = NULL;
	float * d_inputData = NULL;
	
	unsigned int * d_start = NULL;
	unsigned int * d_stop = NULL;

	curandGenerator_t rng = NULL;


	// Allocate device data.
	CUDA_CALL(cudaMalloc((void**)&d_accumulator, 1 * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_inputData, numInputs * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_start, numInputs * sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc((void**)&d_stop, numInputs * sizeof(unsigned int)));


	// Initialise accumulator
	(*h_accumulator) = 0.0f;
	CUDA_CALL(cudaMemcpy(d_accumulator, h_accumulator, 1 * sizeof(float), cudaMemcpyHostToDevice));

	// Create RNG, seed RNG and populate device array.
	curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT); // @todo - curand error check
	curandSetPseudoRandomGeneratorSeed(rng, seed); // @todo - curand error check
	curandGenerateUniform(rng, d_inputData, numInputs); // @todo - curand error check

	// Accumulate values via kernel.
	int blockSize, minGridSize, gridSize;
	CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, atomicAdd_test, 0, numInputs));
	gridSize = (numInputs + blockSize - 1) / blockSize;
	atomicAdd_test<<<gridSize, blockSize>>>(numInputs, d_inputData, d_accumulator, d_start, d_stop);
	CUDA_CHECK();

	// Copy Data from Device to Host
	CUDA_CALL(cudaMemcpy(h_accumulator, d_accumulator, 1 * sizeof(float), cudaMemcpyDeviceToHost));
	//CUDA_CALL(cudaMemcpy(h_inputData, d_inputData, numInputs * sizeof(float), cudaMemcpyDeviceToHost)); //@todo - remove?
	CUDA_CALL(cudaMemcpy(h_start, d_start, numInputs * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(h_stop, d_stop, numInputs * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
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

	// Print output messages

	fprintf(stdout, "Accumulator: %f\n", h_accumulator[0]);
	fflush(stdout);

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
	runAtomicAddTest(1, 128, 0);
    return 0;
}
