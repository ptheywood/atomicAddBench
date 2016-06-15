
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"

#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include <typeinfo>

#define VERBOSE 0
#define INTEGER_SCALE_FACTOR 100

#define DEFAULT_NUM_REPEATS 1
#define DEFAULT_NUM_ITERATIONS 1
#define DEFAULT_NUM_ELEMENTS 128
#define DEFAULT_SEED 0
#define DEFAULT_DEVICE 0

#ifdef WIN32
#define EXE_NAME "atomicAddBench.exe"
#else
#define EXE_NAME "atomicAddBench"
#endif

#define MIN_ARGS 1
#define MAX_ARGS 6
#define ARG_REPEATS 1
#define ARG_ITERATIONS 2
#define ARG_ELEMENTS 3
#define ARG_SEED 4
#define ARG_DEVICE 5

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


__device__ double atomicAddViaCAS(double* address, double val)
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

/*
__device__ double atomicAddViaCAS(float* address, float val)
{
	//@todo
	unsigned int* address_as_u = (unsigned int*)address;
	unsigned int old = *address_as_u;
	unsigned int assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_u, assumed, reinterpret_cast<unsigned int>(val + reinterpret_cast<float>(assumed)));
	} while (assumed != old);
	return reinterpret_cast<float>(old);
}
*/

__global__ void atomicAdd_intrinsic(unsigned int numIterations, unsigned int numInputs, float * d_inputData, float * d_accumulator){
	unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if (tid < numInputs){
		for (int iteration = 0; iteration < numIterations; iteration++){
			atomicAdd(d_accumulator, d_inputData[tid]);
		}
	}
}


__global__ void atomicAdd_intrinsic(unsigned int numIterations, unsigned int numInputs, float * d_inputData, double * d_accumulator){
	unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if(tid < numInputs){
		for (int iteration = 0; iteration < numIterations; iteration++){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 8
			atomicAdd(d_accumulator, d_inputData[tid]);
#endif 
		}
	}
}


__global__ void atomicAdd_cas(unsigned int numIterations, unsigned int numInputs, float * d_inputData, float * d_accumulator){
	unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if (tid < numInputs){
		for (int iteration = 0; iteration < numIterations; iteration++){
			//@todo
		}
	}
}


__global__ void atomicAdd_cas(unsigned int numIterations, unsigned int numInputs, float * d_inputData, double * d_accumulator){
	unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if (tid < numInputs){
		for (int iteration = 0; iteration < numIterations; iteration++){
			atomicAddViaCAS(d_accumulator, (double)d_inputData[tid]);
		}
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

void fprintAccumulatorTotal(FILE* f, int v){
	fprintf(f, "%d\n", v);
}
void fprintAccumulatorTotal(FILE* f, unsigned int v){
	fprintf(f, "%u\n", v);
}
void fprintAccumulatorTotal(FILE* f, unsigned long long int v){
	fprintf(f, "%llu\n", v);
}
void fprintAccumulatorTotal(FILE* f, float v){
	fprintf(f, "%f\n", v);
}
void fprintAccumulatorTotal(FILE* f, double v){
	fprintf(f, "%f\n", v);
}

void checkUsage(
	int argc,
	char *argv[],
	unsigned int *numRepeats,
	unsigned int *numIterations,
	unsigned int *numElements,
	unsigned long long int *seed,
	unsigned int *device
	){
		// If an incorrect number of arguments is specified, print usage.
		if (argc < MIN_ARGS || argc > MAX_ARGS ){
			const char *usage = "Usage: \n"
				"%s <num_iterations> <num_elements> <seed>\n"
				"\n"
				"    <num_iterations> number of atomicAdd iterations to repeat (default %u)\n"
				"    <num_elements>   number of threads to launch (default %u)\n"
				"    <seed>           seed for RNG (default %llu)\n"
				"\n";
			fprintf(stdout, usage, EXE_NAME, DEFAULT_NUM_ITERATIONS, DEFAULT_NUM_ELEMENTS, DEFAULT_SEED);
			fflush(stdout);
			exit(EXIT_FAILURE);
		}

		// If there are more than 1 arg (the filename)5
		if(argc > MIN_ARGS){
			// Extract the number of repeats
			(*numRepeats) = (unsigned int) atoi(argv[ARG_REPEATS]);
			// Extract the number of iterations
			(*numIterations) = (unsigned int) atoi(argv[ARG_ITERATIONS]);
			// Extract the number of elements
			(*numElements) = (unsigned int) atoi(argv[ARG_ELEMENTS]);
			// Extract the seed
			(*seed) = strtoull(argv[ARG_SEED], nullptr, 0);
			if (argc >= ARG_DEVICE + 1){
				// Extract the device
				(*device) = (unsigned int)atoi(argv[ARG_DEVICE]);
			}

		}

		printf("repeats:    %u\n", (*numRepeats));
		printf("iterations: %u\n", (*numIterations));
		printf("threads:    %u\n", (*numElements));
		printf("seed:       %llu\n", (*seed));

}

void initDevice(unsigned int device, int *major, int *minor){
	int deviceCount = 0;
	cudaError_t status;
	// Get the number of cuda device.
	status = cudaGetDeviceCount(&deviceCount);
	if (status != cudaSuccess){
		fprintf(stderr, "Cuda Error getting device count.\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
	// If there are any devices
	if (deviceCount > 0){
		// Ensure the device count is not bad.
		if (device >= (unsigned int)deviceCount){
			device = DEFAULT_DEVICE;
			fprintf(stdout, "Warning: device %d is invalid, using device %d\n", device, DEFAULT_DEVICE);
			fflush(stdout);
		}
		// Set the device
		status = cudaSetDevice(device);
		// If there were no errors, proceed.
		if (status == cudaSuccess){
			// Get properties
			cudaDeviceProp props;
			status = cudaGetDeviceProperties(&props, device);
			// If we have properties, print the device.
			if (status == cudaSuccess){
				fprintf(stdout, "Device: %s\n  pci %d bus %d\n  tcc %d\n  SM %d%d\n\n", props.name, props.pciDeviceID, props.pciBusID, props.tccDriver, props.major, props.minor);
				(*major) = props.major;
				(*minor) = props.minor;
			}
		}
		else {
			fprintf(stderr, "Error setting CUDA Device %d.\n", device);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}
	}
	else {
		fprintf(stderr, "Error: No CUDA Device found.\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}			
}

// @todo - change intrinsic to compile time value (Partial template?)
// @todo - repeat and average the time taken.
template <typename T, bool intrinsic>
void test(unsigned int numRepeats, unsigned int numIterations, unsigned int numElements, unsigned long long int seed, T *h_accumulator, T *d_accumulator, float *d_inputData){
	
	if (intrinsic){
		fprintf(stdout, "%s intrinsic \n", typeid(*h_accumulator).name());
	}
	else {
		fprintf(stdout, "%s atomicCAS \n", typeid(*h_accumulator).name());
	}

	float milliTotal = 0.0f;
	for (unsigned int repeat = 0; repeat < numRepeats; repeat++){
		// Reset accumulator.
		(*h_accumulator) = (T)0.0;
		CUDA_CALL(cudaMemcpy(d_accumulator, h_accumulator, 1 * sizeof(T), cudaMemcpyHostToDevice));

		// Create timing elements
		cudaEvent_t start, stop;
		float milliseconds = 0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		// Get pointer to kernel

		void(*kernel)(unsigned int, unsigned int, float*, T*);

		if (intrinsic){
			kernel = atomicAdd_intrinsic;
		}
		else {
			kernel = atomicAdd_cas;
		}

		// Compute launch args and launch kernel
		int blockSize, minGridSize, gridSize;
		CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, numElements));
		gridSize = (numElements + blockSize - 1) / blockSize;

		// Accumulate values via kernel.

		CUDA_CALL(cudaEventRecord(start));
		kernel << <gridSize, blockSize >> >(numIterations, numElements, d_inputData, d_accumulator);
		CUDA_CHECK();
		cudaDeviceSynchronize();
		CUDA_CALL(cudaEventRecord(stop));

		// Output timing 
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);

		// Copy out results
		CUDA_CALL(cudaMemcpy(h_accumulator, d_accumulator, 1 * sizeof(T), cudaMemcpyDeviceToHost));

#if defined(VERBOSE) && VERBOSE > 0
		fprintf(stdout, "  > time %fms value ", milliseconds);
		fprintAccumulatorTotal(stdout, h_accumulator[0]);
#endif
		fflush(stdout);
		milliTotal += milliseconds;
	}

	float milliAverage = milliTotal / numRepeats;


	fprintf(stdout, "  Value: ");
	fprintAccumulatorTotal(stdout, h_accumulator[0]);
	fprintf(stdout, "  Total  : %fms\n", milliTotal);
	fprintf(stdout, "  Average: %fms\n\n", milliAverage);
	fflush(stdout);
}

int main(int argc, char *argv[])
{
	unsigned int numRepeats = DEFAULT_NUM_REPEATS;
	unsigned int numIterations = DEFAULT_NUM_ITERATIONS;
	unsigned int numElements = DEFAULT_NUM_ELEMENTS;
	unsigned long long int seed = DEFAULT_SEED;
	unsigned int device = DEFAULT_DEVICE;
	int major = 0;
	int minor = 0;

	checkUsage(argc, argv, &numRepeats, &numIterations, &numElements, &seed, &device);

	// Initialise the device
	initDevice(device, &major, &minor);
 
	// Alloc Rands.
	float *d_inputData = NULL;
	CUDA_CALL(cudaMalloc((void**)&d_inputData, numElements * sizeof(float)));

	// Alloc accumulator as double
	void *h_accumulator = (void *)malloc(sizeof(double));
	void *d_accumulator = NULL;
	CUDA_CALL(cudaMalloc((void**)&d_accumulator, numElements * sizeof(double)));
	
	// Generate rands
	generateInputData(numElements, seed, d_inputData);

	// Test float intrinsic
	test<float, true>(numRepeats, numIterations, numElements, seed, reinterpret_cast<float*>(h_accumulator), reinterpret_cast<float*>(d_accumulator), d_inputData);

#if __CUDACC_VER_MAJOR__ >= 8
	if (major >= 6){
		// Test double intrinsic if possible
		test<double, true>(numRepeats, numIterations, numElements, seed, reinterpret_cast<double*>(h_accumulator), reinterpret_cast<double*>(d_accumulator), d_inputData);
	}
	else {
		printf("double intrinsic not available SM %d.%d\n\n", major, minor);
	}
#endif

	//test<float, false>(numRepeats, numIterations, numElements, seed, reinterpret_cast<float*>(h_accumulator), reinterpret_cast<float*>(d_accumulator), d_inputData);

	// Test double atomicCAS.
	test<double, false>(numRepeats, numIterations, numElements, seed, reinterpret_cast<double*>(h_accumulator), reinterpret_cast<double*>(d_accumulator), d_inputData);


	// Free arrays.
	CUDA_CALL(cudaFree(d_inputData));
	CUDA_CALL(cudaFree(d_accumulator));
	free(h_accumulator);


	// Reset the device.
	CUDA_CALL(cudaDeviceReset());

    return 0;
}
