/*
 * cubffa (CUda Binary Finite Field Arithmetic library) provides 
 * functions for large binary galois field arithmetic on GPUs. 
 * Besides CUDA it is also possible to extend cubffa to any other 
 * underlying framework.
 * Copyright (C) 2016  Dominik Stamm
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "GF2nArithmeticCudaKernels.h"

#define SIZE_CHUNK 64

////////////////////////////////////////////////////////////////////////////////
//
//  Device Contants
//
////////////////////////////////////////////////////////////////////////////////
__constant__ uint32 dMaxThreadsPerBlock;
__constant__ uint32 dSharedMemPerBlock;
__constant__ uint32 dMultiProcessorCount;
__constant__ uint32 dWarpSize;
__constant__ uint32 dMaxThreadsPerMultiProcessor;

////////////////////////////////////////////////////////////////////////////////
//
//	Host functions
//
////////////////////////////////////////////////////////////////////////////////
__host__ void cudaGetDeviceProperies( 
	uint32 *hMaxThreadsPerBlock, 
	uint32 *hSharedMemPerBlock, 
	uint32 *hMultiProcessorCount, 
	uint32 *hWarpSize, 
	uint32 *hMaxThreadsPerMultiProcessor
	)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	CudaSafeCall(cudaMemcpyToSymbol(dMaxThreadsPerBlock, (const char *)&deviceProp.maxThreadsPerBlock, sizeof(uint32), 0, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpyToSymbol(dSharedMemPerBlock, &deviceProp.sharedMemPerBlock, sizeof(uint32)));
	CudaSafeCall(cudaMemcpyToSymbol(dMultiProcessorCount, &deviceProp.multiProcessorCount, sizeof(uint32)));
	CudaSafeCall(cudaMemcpyToSymbol(dWarpSize, &deviceProp.warpSize, sizeof(uint32)));
	CudaSafeCall(cudaMemcpyToSymbol(dMaxThreadsPerMultiProcessor, &deviceProp.maxThreadsPerMultiProcessor, sizeof(uint32)));

	*hMaxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
	*hSharedMemPerBlock = deviceProp.sharedMemPerBlock;
	*hMultiProcessorCount = deviceProp.multiProcessorCount;
	*hWarpSize = deviceProp.warpSize;
	*hMaxThreadsPerMultiProcessor = deviceProp.maxThreadsPerMultiProcessor;
}

__global__ void EmptyKernel() { }

/**************************************************************************\

                                   Addition

\**************************************************************************/

////////////////////////////////////////////////////////////////////////////////
//
//	Adds a and b and stores the result in res
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaParAddKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks ) 
{
	int32 thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < num_chunks )
		res[thid] = a[thid] ^ b[thid];
}

__global__ void cudaParAddKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, uint32 num_chunks ) 
{
	int32 thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < num_chunks )
		a[thid] = a[thid] ^ b[thid];
}

__global__ void cudaParAddLoopKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks )
{
    for( int32 thid = blockIdx.x * blockDim.x + threadIdx.x; 
         thid < num_chunks; 
         thid += blockDim.x * gridDim.x) 
      {
          res[thid] = a[thid] ^ b[thid];
      }
}

__global__ void cudaParAddTimeKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks, timing_stats *time )
{
#ifdef LINUXINTEL32
	int32 thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < num_chunks )
	{
		// define registers
		asm(".reg .s32 %t<14>;\n\t"
			"mov.u32 %t0, %clock;");
        
        // load a and b to registers
        asm("cvta.to.global.u32 %%t4, %0;\n\t"
            "shl.b32 %%t5, %%r1, 2;\n\t"
            "add.s32 %%t6, %%t5, %%t4;\n\t"
            "cvta.to.global.u32 %%t7, %1;\n\t"
            "add.s32 %%t8, %%t7, %%t5;\n\t"
            "ld.global.u32 %%t9, [%%t8];\n\t"
            "ld.global.u32 %%t10, [%%t6];\n\t"
            "mov.u32 %%t1, %%clock;"
            :: "r"(a), "r"(b));

        asm("xor.b32 %t11, %t9, %t10;\n\t"
        	"mov.u32 %t2, %clock;");

        asm("cvta.to.global.u32 %%t12, %0;\n\t"
            "add.s32 %%t13, %%t12, %%t5;\n\t"
            "st.global.u32 [%%t13], %%t11;\n\t"
            "mov.u32 %%t3, %%clock;"
            :: "r"(res));

        asm volatile("mov.u32 %0, %%t0;" : "=r"(time[thid].time1) :: "memory");
        asm volatile("mov.u32 %0, %%t1;" : "=r"(time[thid].time2) :: "memory");
        asm volatile("mov.u32 %0, %%t2;" : "=r"(time[thid].time3) :: "memory");
        asm volatile("mov.u32 %0, %%t3;" : "=r"(time[thid].time4) :: "memory");
	}
#endif
}

__global__ void cuda2ParAddKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks )
{
	int32 thid = (blockIdx.x * blockDim.x * 2) + threadIdx.x;

	if( thid + blockDim.x < num_chunks )
	{
		res[thid + 0 * blockDim.x] = a[thid + 0 * blockDim.x] ^ b[thid + 0 * blockDim.x];
		res[thid + 1 * blockDim.x] = a[thid + 1 * blockDim.x] ^ b[thid + 1 * blockDim.x];
	}
}

__global__ void cuda4ParAddKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks )
{
	int32 thid = (blockIdx.x * blockDim.x * 4) + threadIdx.x;

	if( thid + 3 * blockDim.x < num_chunks )
	{
		res[thid + 0 * blockDim.x] = a[thid + 0 * blockDim.x] ^ b[thid + 0 * blockDim.x];
		res[thid + 1 * blockDim.x] = a[thid + 1 * blockDim.x] ^ b[thid + 1 * blockDim.x];
		res[thid + 2 * blockDim.x] = a[thid + 2 * blockDim.x] ^ b[thid + 2 * blockDim.x];
		res[thid + 3 * blockDim.x] = a[thid + 3 * blockDim.x] ^ b[thid + 3 * blockDim.x];
	}
}

__global__ void cuda8ParAddKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks )
{
	int32 thid = (blockIdx.x * blockDim.x * 8) + threadIdx.x;

	if( thid + 7 * blockDim.x < num_chunks )
	{
		res[thid + 0 * blockDim.x] = a[thid + 0 * blockDim.x] ^ b[thid + 0 * blockDim.x];
		res[thid + 1 * blockDim.x] = a[thid + 1 * blockDim.x] ^ b[thid + 1 * blockDim.x];
		res[thid + 2 * blockDim.x] = a[thid + 2 * blockDim.x] ^ b[thid + 2 * blockDim.x];
		res[thid + 3 * blockDim.x] = a[thid + 3 * blockDim.x] ^ b[thid + 3 * blockDim.x];
		res[thid + 4 * blockDim.x] = a[thid + 4 * blockDim.x] ^ b[thid + 4 * blockDim.x];
		res[thid + 5 * blockDim.x] = a[thid + 5 * blockDim.x] ^ b[thid + 5 * blockDim.x];
		res[thid + 6 * blockDim.x] = a[thid + 6 * blockDim.x] ^ b[thid + 6 * blockDim.x];
		res[thid + 7 * blockDim.x] = a[thid + 7 * blockDim.x] ^ b[thid + 7 * blockDim.x];
	}
}

__global__ void cudaParAddSharedMemKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks ) 
{
	extern __shared__ CUDA_BIGNUM sa[];
	extern __shared__ CUDA_BIGNUM sb[];
	extern __shared__ CUDA_BIGNUM sr[];

	int32 thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < num_chunks )
		res[thid] = a[thid] ^ b[thid];
}

/**************************************************************************\

                               Multiplication

\**************************************************************************/

__global__ void cudaParMulKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks, CUDA_BIGNUM *irred_poly, uint32 indx_mask_bit ) 
{
	for( uint32 i=num_chunks; i>0; --i ) 
	{
		for( uint32 j=0; j<CUDA_BIGNUM_SIZE_BITS; ++j ) 
		{
			if( isbitset(b[i-1], j) ) 
			{
				for( uint32 k=0; k<num_chunks; ++k )
				{
					res[k] ^= a[k];
				}
			} 
			
			cudaBitShiftLeft1(a, num_chunks);
			cudaReducePoly(a, num_chunks, irred_poly, indx_mask_bit);
		}
	}
}

__global__ void cudaCreateChunkProdArrayKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *arr, uint32 numChunks )
{
	// thid_x is the column of the result array
	int32 thid_x = (blockIdx.x * blockDim.x) + threadIdx.x;
	// thid_y is the row of the result array
	int32 thid_y = (blockIdx.y * blockDim.y) + threadIdx.y;
	// thid inside of the current block
	// this is used to index the shared memory of the current thread
	int32 thid_inBlock = (threadIdx.y * blockDim.x) + threadIdx.x;

	// The temporary result of each thread:
	// -> shared memory is allocated for the entire block
	// -> 64 threads per block * 2 chunks per result
	// The temporary result is used to store the intermediate
	// result of the multiplication. If no shared memory
	// value would be used, every intermediate result has to
	// be stored in global memory which would result in 
	// decreased memory efficiency ans performance.
	__shared__ CUDA_BIGNUM res[2 * 64];
	// Because every block has to shift the value of a
	// it has to be stored in a temporary shared memory value.
	__shared__ CUDA_BIGNUM a_tmp[2 * 64];

	// Only exeute if the two dimensional thread indices
	// reference an element of the result array.
	if( thid_x < numChunks && thid_y < numChunks )
	{
		// Initialize the result with 0
		res[2 * thid_inBlock] 	  = 0;
		res[2 * thid_inBlock + 1] = 0;

		// Copy the value of a to a_tmp
		a_tmp[2 * thid_inBlock] 	= 0;
		a_tmp[2 * thid_inBlock + 1] = a[thid_x];

		// For every bit of the value b ...
		for( uint32 j=0; j<CUDA_BIGNUM_SIZE_BITS; ++j ) 
		{
			// ... check if the bit is set
			if( isbitset(b[thid_y], j) ) 
			{
				// If the bit in b is set -> add the current content of 
				// a_tmp to the temporary result
				res[2 * thid_inBlock] 	  ^= a_tmp[2 * thid_inBlock];
				res[2 * thid_inBlock + 1] ^= a_tmp[2 * thid_inBlock + 1];
			} 
			
			// Shift the content of a_tmp one bit to the left
			cudaBitShiftLeft1(&a_tmp[2 * thid_inBlock], 2);
		}

		// Copy the temporary result to the global result array
		arr[(thid_y + thid_x + 1) * 2 * numChunks + thid_x] = res[2 * thid_inBlock + 1];
		// Copy the carry to the global result array
		arr[(thid_y + thid_x) * 2 * numChunks + numChunks + thid_x] = res[2 * thid_inBlock];
	}
}

// TODO: Array Zeile des Blocks in Shared Memory puffern
__global__ void cudaParDiagBlockSumKernel( CUDA_BIGNUM *arr, uint32 numChunks ) 
{
	// thid_x is the column of the array
	int32 thid_x = (blockIdx.x * blockDim.x) + threadIdx.x;
	// thid_y is the row of the array
	int32 thid_y = (blockIdx.y * blockDim.y) + threadIdx.y;

	// Only exeute if the two dimensional thread indices
	// reference an element of the array.
	if( thid_x < numChunks && thid_y < 2 * numChunks )
	{
		// Each row consists of 2 * numChunks elements
		int32 chunksRow = 2 * numChunks;
		// rowOffset is the index of the first element
		// in row thid_y if the matrix arr is 
		// transformed to a 1D array
		int32 rowOffset = thid_y * chunksRow;
		// colOffset is the index of the column 
		// calculated over all blocks. Because every block
		// sums twice as much elements as the number of 
		// threads per block, the blockDix.x is multiplied 
		// by 2.
		int32 colOffset = blockIdx.x * blockDim.x * 2;
		// currArr is the part of the array used by the
		// current block in respect of the row and col offset
		// Example:
		// - Block 0: currArr = arr[0 ... 127]
		// - Block 1: currArr = arr[128 ... 255]
		// ...
		CUDA_BIGNUM *currArr = &arr[rowOffset + colOffset];
		// numElemLeftInRow is the number of elements left
		// in the current row, starting at the current
		// column offset.
		int32 numElemLeftInRow = chunksRow - colOffset;

		// there are 64 threads per block
		for( int32 offset = 1; offset <= blockDim.x; offset = offset<<1 )
		{
			if( threadIdx.x * 2 * offset + offset < numElemLeftInRow )	
		    	currArr[threadIdx.x * 2 * offset] ^= currArr[threadIdx.x * 2 * offset + offset];

		  	// only syncs the threads of a block
		  	// => kernel results in one result per block
		  	__syncthreads();
		}
	}
}

__global__ void cudaParChunkSumKernel( CUDA_BIGNUM *arr, uint32 numChunks, 
	uint32 numElements, uint32 offset, CUDA_BIGNUM *res ) 
{
	// thid is the row index of the array
	int32 thid = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// Only exeute if the row indes reference 
	// a valid row of the array.
	if( thid < 2 * numChunks )
	{
		// currArr points at the first element
		// of the current row indexed by the thid
		CUDA_BIGNUM *currArr = &arr[thid * 2 * numChunks];
		// res is initialized with the first row entry.
		// This ensures that res containts valid data and
		// the other chunks can be added
		res[thid] = currArr[0];

		// Now the results of all blocks created by the
		// function cudaParDiagBlockSumKernel are summed up.
		// numElements is equal to the number of blocks
		// in a row, whereas offset is the space between two 
		// blocks represented as number of chunks.
		for( uint32 i = 1; i < numElements; ++i )
			res[thid] ^= currArr[i * offset];
	}
}

__global__ void cudaShiftRightKernel( 
	CUDA_BIGNUM *a, uint32 numChunksA, 
	uint32 numBitsToShiftInBlock /*b*/, 
	uint32 blockOffset /*o*/, 
	CUDA_BIGNUM maskLeft /*l*/, 
	CUDA_BIGNUM maskRight /*r*/, 
	CUDA_BIGNUM *res, uint32 numChunksRes )
{
	// thid is the chunk index of the result
	uint32 thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Only exeute if the index references
	// a valid chunk of res.
	if(thid < numChunksRes) 
	{
		// Clear current result chunk
		res[thid] = 0;

		// First the low part of the result chunk is created.
		// Therefor the calculated source chunk of a has to be valid.
		if( blockOffset < numChunksA + thid + 1 )
			res[thid] = (*(a + numChunksRes + thid - blockOffset) & maskRight) >> numBitsToShiftInBlock;

		// As second step the high part of the result chunk is added.
		// Again the source chunk of a has to be valid.
		if( blockOffset < numChunksA + thid )
			res[thid] |= (*(a + numChunksRes + thid - (blockOffset + 1)) & maskLeft) << (CUDA_BIGNUM_SIZE_BITS - numBitsToShiftInBlock);
	}
}

__global__ void cudaShiftLeftKernel( 
	CUDA_BIGNUM *a, uint32 numChunksA, 
	uint32 numBitsToShiftInBlock /*b*/, 
	uint32 blockOffset /*o*/, 
	CUDA_BIGNUM maskLeft /*l*/, 
	CUDA_BIGNUM maskRight /*r*/, 
	CUDA_BIGNUM *res, uint32 numChunksRes )
{
	// thid is the chunk index of the result
	uint32 thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Only exeute if the index references
	// a valid chunk of res.
	if( thid < numChunksRes ) 
	{
		// Clear current result chunk
		res[thid] = 0;

		// First the high part of the result chunk is created.
		// Therefor the calculated source chunk of a has to be valid.
		if( thid + blockOffset >= numChunksRes - numChunksA && 
			thid + blockOffset - (numChunksRes - numChunksA) < numChunksA )
			res[thid] = (*(a + thid + blockOffset - (numChunksRes - numChunksA)) & maskLeft) << numBitsToShiftInBlock;

		// As second step the kiw part of the result chunk is added.
		// Again the source chunk of a has to be valid.
		if( thid + blockOffset + 1 >= numChunksA - numChunksA &&
			thid + blockOffset - (numChunksRes - numChunksA) + 1 < numChunksA )
			res[thid] |= (*(a + thid + blockOffset - (numChunksRes - numChunksA) + 1) & maskRight) >> (CUDA_BIGNUM_SIZE_BITS - numBitsToShiftInBlock);
	}
}

__global__ void cudaGetFirstNonEmptyChunkIndexKernel( CUDA_BIGNUM *x, uint32 numChunks, CUDA_BIGNUM *foundChunkIndex )
{
	// thid is the chunk index of the result
	uint32 thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid == 0 )
		*foundChunkIndex = ~0u;

	if( thid < numChunks )
	{
		if( x[thid] > 0 )
		{
#ifdef LINUXINTEL64
			atomicMin((unsigned long long *)foundChunkIndex, (unsigned long long)thid);
#else
			atomicMin((unsigned int *)foundChunkIndex, (unsigned int)thid);
#endif
		}
	}
}

__global__ void cudaCalcElemDegreeKernel( CUDA_BIGNUM *x, uint32 numChunks, CUDA_BIGNUM *foundChunkIndex, CUDA_BIGNUM *elemDegree )
{
	if( *foundChunkIndex == ~((CUDA_BIGNUM)0) )
		*elemDegree = 0;
	else
	{
		CUDA_BIGNUM index = 1;
		CUDA_BIGNUM h = ~(~((CUDA_BIGNUM)0) >> 1); 
		while( !(x[*foundChunkIndex] & h) ) 
		{
			h >>= 1;
			++index;
		}

		*elemDegree = (numChunks - 1 - (*foundChunkIndex)) * CUDA_BIGNUM_SIZE_BITS + (CUDA_BIGNUM_SIZE_BITS - index);
	}
}

__global__ void cudaPrintKernel( CUDA_BIGNUM *x, uint32 num_chunks )
{
	printf("x = %u\n", *x);
}

__global__ void cudaSwapBytes( CUDA_BIGNUM *x, uint32 num_chunks )
{
	uint32 thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < num_chunks )
	{
		x[thid] = swapBytes(x[thid]);
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//	Sets x to 0
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaSet0Kernel( CUDA_BIGNUM *x, uint32 length ) 
{
	uint32 thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < length )
	{
		x[thid] = 0;
	}
}

__global__ void cudaSet1Kernel( CUDA_BIGNUM *x, uint32 length ) 
{
	uint32 thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < length - 1 )
	{
		x[thid] = 0;
	}
	else if( thid == length - 1 )
	{
		x[thid] = 1;
	}
}

__global__ void cudaMaskFirstChunkKernel( CUDA_BIGNUM *x, CUDA_BIGNUM mask )
{
	x[0] = x[0] & mask;
}

__global__ void cudaHexDumpKernel( char *desc, void *addr, int len )
{
	cudaHexDump(desc, addr, len);
}

// ////////////////////////////////////////////////////////////////////////////////
// //
// //	Copies a to b
// //
// ////////////////////////////////////////////////////////////////////////////////
// __global__ void cudaCopyKernel( CUDA_BIGNUM *a, uint32 num_chunks_a, CUDA_BIGNUM *b, uint32 num_chunks_b ) 
// {
// 	uint32 thid = (blockIdx.x * blockDim.x) + threadIdx.x;

// 	if( thid < num_chunks_b ) {
// 		if( thid < num_chunks_a )
// 			b[thid] = a[thid];
// 		else
// 			b[thid] = 0;
// 	}
// }

// ////////////////////////////////////////////////////////////////////////////////
// /*
// 	Device Functions
// */
// ////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
//	Shift a 1 Bit to the left
//
////////////////////////////////////////////////////////////////////////////////
__device__ void cudaBitShiftLeft1( CUDA_BIGNUM *a, uint32 num_chunks ) 
{
	CUDA_BIGNUM tmp = 0;
	CUDA_BIGNUM carry = 0;
	CUDA_BIGNUM carry_last = 0;

	CUDA_BIGNUM lmask = pow((double)2, (double)CUDA_BIGNUM_SIZE_BITS) - 1;
	CUDA_BIGNUM umask = pow((double)2, (double)CUDA_BIGNUM_SIZE_BITS - 1);

	for( uint32 i = num_chunks; i > 0; --i ) 
	{
		tmp = 0;
		tmp = a[i-1];
		carry_last = carry;
		carry = (tmp&umask) >> (CUDA_BIGNUM_SIZE_BITS - 1);
		tmp <<= 1;
		a[i-1] = (tmp&lmask) | carry_last;
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//	Checks if the bit at pos bitnum is set (idx from right to left)
//
////////////////////////////////////////////////////////////////////////////////
__device__ bool isbitset( CUDA_BIGNUM val, uint32 bitnum ) 
{
	return (val & ((CUDA_BIGNUM)1 << bitnum)) != 0;
}

////////////////////////////////////////////////////////////////////////////////
//
//	Reduce the extended field polynomial
//	TODO: Algorithmus der nur die Chunks reduziert in denen irred_poly mind
//		  ein Bit == 1 besitzt
//
////////////////////////////////////////////////////////////////////////////////
__device__ void cudaReducePoly(
	CUDA_BIGNUM *value,
	uint32 num_chunks,
	CUDA_BIGNUM *irred_poly,
	uint32 indx_mask_bit
	) 
{
	if( isbitset(value[0], CUDA_BIGNUM_SIZE_BITS - indx_mask_bit) ) 
	{
		for( uint32 i=0; i<num_chunks; ++i )
		{
			value[i] ^= irred_poly[i];
		}
	}
}

__device__ void cudaHexDump( char *desc, void *addr, int len ) 
{
	int i;
	unsigned char buff[17];       // stores the ASCII data
	unsigned char *pc = (unsigned char *)addr;     // cast to make the code cleaner.

	// Output description if given.
	if (desc != NULL)
		printf ("%s:\n", desc);

	// Process every byte in the data.

	for (i = 0; i < len; i++) {
		// Multiple of 16 means new line (with line offset).

		if ((i % 16) == 0) {
		// Just don't print ASCII for the zeroth line.

			if (i != 0)
				printf ("  %s\n", buff);

			// Output the offset.

			printf ("  %04x ", i);
		}

		// Now the hex code for the specific character.

		printf (" %02x", pc[i]);

		// And store a printable ASCII character for later.

		if ((pc[i] < 0x20) || (pc[i] > 0x7e))
			buff[i % 16] = '.';
		else
			buff[i % 16] = pc[i];
    	
    	buff[(i % 16) + 1] = '\0';
	}

	// Pad out last line if not exactly 16 characters.

	while ((i % 16) != 0) {
		printf ("   ");
		i++;
	}

	// And print the final ASCII bit.

	printf ("  %s\n", buff);
}


__device__ uint16 swapBytes( uint16 val ) 
{
    return (val << 8) | (val >> 8 );
}

__device__ int16 swapBytes( int16 val ) 
{
    return (val << 8) | ((val >> 8) & 0xFF);
}

__device__ uint32 swapBytes( uint32 val )
{
    val = ((val << 8) & 0xFF00FF00 ) | ((val >> 8) & 0xFF00FF ); 
    return (val << 16) | (val >> 16);
}

__device__ int32 swapBytes( int32 val )
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF ); 
    return (val << 16) | ((val >> 16) & 0xFFFF);
}

__device__ int64 swapBytes( int64 val )
{
    val = ((val << 8) & 0xFF00FF00FF00FF00ULL ) | ((val >> 8) & 0x00FF00FF00FF00FFULL );
    val = ((val << 16) & 0xFFFF0000FFFF0000ULL ) | ((val >> 16) & 0x0000FFFF0000FFFFULL );
    return (val << 32) | ((val >> 32) & 0xFFFFFFFFULL);
}

__device__ uint64 swapBytes( uint64 val )
{
    val = ((val << 8) & 0xFF00FF00FF00FF00ULL ) | ((val >> 8) & 0x00FF00FF00FF00FFULL );
    val = ((val << 16) & 0xFFFF0000FFFF0000ULL ) | ((val >> 16) & 0x0000FFFF0000FFFFULL );
    return (val << 32) | (val >> 32);
}