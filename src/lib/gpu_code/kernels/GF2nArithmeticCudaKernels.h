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
 
#ifndef __GF2N_ARITHMETIC_CUDA_KERNELS_H__
#define __GF2N_ARITHMETIC_CUDA_KERNELS_H__

#include "CudaBignum.h"
#include "CudaUtils.h"

__host__ void cudaGetDeviceProperies( uint32 *hMaxThreadsPerBlock, uint32 *hSharedMemPerBlock, uint32 *hMultiProcessorCount, uint32 *hWarpSize, uint32 *hMaxThreadsPerMultiProcessor );

////////////////////////////////////////////////////////////////////////////////
//
//	Adds a and b and stores the result in a
//
////////////////////////////////////////////////////////////////////////////////
__global__ void EmptyKernel();

__global__ void cudaParAddKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks );
__global__ void cudaParAddKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, uint32 num_chunks );
__global__ void cudaParAddLoopKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks );
__global__ void cudaParAddTimeKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks, timing_stats *time );
__global__ void cuda2ParAddKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks );
__global__ void cuda4ParAddKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks );
__global__ void cuda8ParAddKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks );
__global__ void cudaParAddSharedMemKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks );

__global__ void cudaParMulKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *res, uint32 num_chunks, CUDA_BIGNUM *irred_poly, uint32 indx_mask_bit );

// Helper functions for parallel multiplication
__global__ void cudaCreateChunkProdArrayKernel( CUDA_BIGNUM *a, CUDA_BIGNUM *b, CUDA_BIGNUM *arr, uint32 numChunks );
__global__ void cudaParDiagBlockSumKernel( CUDA_BIGNUM *arr, uint32 numChunks ) ;
__global__ void cudaParChunkSumKernel( CUDA_BIGNUM *arr, uint32 numChunks, uint32 numElements, uint32 offset, CUDA_BIGNUM *res );
__global__ void cudaShiftRightKernel( CUDA_BIGNUM *a, uint32 numChunksA, uint32 numBitsToShiftInBlock, uint32 blockOffset, CUDA_BIGNUM maskLeft, CUDA_BIGNUM maskRight, CUDA_BIGNUM *res, uint32 numChunksRes );
__global__ void cudaShiftLeftKernel( CUDA_BIGNUM *a, uint32 numChunksA, uint32 numBitsToShiftInBlock, uint32 blockOffset, CUDA_BIGNUM maskLeft, CUDA_BIGNUM maskRight, CUDA_BIGNUM *res, uint32 numChunksRes );
__global__ void cudaGetFirstNonEmptyChunkIndexKernel( CUDA_BIGNUM *x, uint32 numChunks, CUDA_BIGNUM *foundChunkIndex );
__global__ void cudaCalcElemDegreeKernel( CUDA_BIGNUM *x, uint32 numChunks, CUDA_BIGNUM *foundChunkIndex, CUDA_BIGNUM *elemDegree );

__global__ void cudaSwapBytes( CUDA_BIGNUM *x, uint32 num_chunks );
__global__ void cudaSet0Kernel( CUDA_BIGNUM *x, uint32 length );
__global__ void cudaSet1Kernel( CUDA_BIGNUM *x, uint32 length );
__global__ void cudaMaskFirstChunkKernel( CUDA_BIGNUM *x, CUDA_BIGNUM mask );
__global__ void cudaPrintKernel( CUDA_BIGNUM *x, uint32 num_chunks );
__global__ void cudaHexDumpKernel( char *desc, void *addr, int len );

__device__ bool isbitset(CUDA_BIGNUM val, uint32 bitnum);
__device__ void cudaBitShiftLeft1( CUDA_BIGNUM *a, uint32 num_chunks );
__device__ void cudaReducePoly(CUDA_BIGNUM *value, uint32 num_chunks, CUDA_BIGNUM *irred_poly, uint32 indx_mask_bit); 
__device__ void cudaPrintbincharpad( CUDA_BIGNUM* ca, uint32 n );
__device__ void cudaHexDump( char *desc, void *addr, int len );

__device__ uint16 swapBytes( uint16 val );
__device__ int16 swapBytes( int16 val );
__device__ uint32 swapBytes( uint32 val );
__device__ int32 swapBytes( int32 val );
__device__ int64 swapBytes( int64 val );
__device__ uint64 swapBytes( uint64 val );

#endif //__GF2N_ARITHMETIC_CUDA_KERNELS_H__