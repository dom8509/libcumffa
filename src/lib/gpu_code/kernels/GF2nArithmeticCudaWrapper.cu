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
#include "GF2nArithmeticCudaWrapper.h"
#include <iostream>
#include <sys/time.h>

namespace libcumffa {
	namespace gpu {

		namespace cuda {

			uint32 g_hMaxThreadsPerBlock = 0;
			uint32 g_hSharedMemPerBlock = 0;
			uint32 g_hMultiProcessorCount = 0;
			uint32 g_hWarpSize = 0;
			uint32 g_hMaxThreadsPerMultiProcessor = 0;
			cudaStream_t g_stream_0;
			cudaStream_t g_stream_1;


			// return the time in milliseconds
			double cpuSecond()
			{
			    struct timeval tp;
			    gettimeofday(&tp, NULL);
			    return ((double)tp.tv_sec * 1000 + (double)tp.tv_usec * 1.e-3);
			}

			void loadPoroperties() 
			{
				cudaGetDeviceProperies(&g_hMaxThreadsPerBlock, &g_hSharedMemPerBlock, &g_hMultiProcessorCount, &g_hWarpSize, &g_hMaxThreadsPerMultiProcessor);
			}

			void device_init()
			{
				//cudaDeviceReset();
				loadPoroperties();
				cudaStreamCreate(&g_stream_0);
				cudaStreamCreate(&g_stream_1);
			}

			/**************************************************************************\

                               sync calls

			\**************************************************************************/

			double device_allocate( CUDA_BIGNUM **d_x, uint32 num_bytes )
			{
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start);

				// allocate space for the device value
				CudaSafeCall(cudaMalloc((void **)d_x, num_bytes));
				
				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double device_allocate_pinned( CUDA_BIGNUM **d_x, uint32 num_bytes )
			{
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start);
				
				CudaSafeCall(cudaHostAlloc((void **)d_x, num_bytes, cudaHostAllocDefault));
				
				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;	
			}

			double device_delete( CUDA_BIGNUM *d_x )
			{
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start);
				
				// free all device values
				CudaSafeCall(cudaFree(d_x));
				
				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double device_delete_pinned( CUDA_BIGNUM *d_x )
			{
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start);
				
				CudaSafeCall(cudaFreeHost(d_x));
				
				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double device_set( CUDA_BIGNUM *d_x, CUDA_BIGNUM *h_x, uint32 num_bytes )
			{
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start);

				// copy values to the device
				CudaSafeCall(cudaMemcpy(d_x, h_x, num_bytes, cudaMemcpyHostToDevice));
				
				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double device_swapBytes( CUDA_BIGNUM *d_x, uint32 num_chunks )
			{
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				uint32 num_threads = min(g_hMaxThreadsPerBlock, (uint32)num_chunks);
				uint32 num_blocks = ceil((double)num_chunks / g_hMaxThreadsPerBlock);

				cudaEventRecord(start);

				cudaSwapBytes<<<num_blocks, num_threads>>>(d_x, num_chunks);			

				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double device_copy( CUDA_BIGNUM *d_y, CUDA_BIGNUM *d_x, uint32 num_bytes )	
			{
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start);

				// copy values from one device value to another
				CudaSafeCall(cudaMemcpy(d_y, d_x, num_bytes, cudaMemcpyDeviceToDevice));
				
				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double device_get( CUDA_BIGNUM *h_x, CUDA_BIGNUM *d_x, uint32 num_bytes )
			{
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start);

				CudaSafeCall(cudaMemcpy(h_x, d_x, num_bytes, cudaMemcpyDeviceToHost));
				
				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double device_clear( CUDA_BIGNUM *d_x, uint32 num_chunks )
			{
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				dim3 threadsPerBlock(min(g_hMaxThreadsPerBlock, (uint32)num_chunks));
				dim3 numBlocks(ceil((double)num_chunks / threadsPerBlock.x));

				cudaEventRecord(start);

				// allocate space for the device value
				cudaSet0Kernel<<<numBlocks, threadsPerBlock>>>(d_x, num_chunks);
				
				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			/**************************************************************************\

                               async calls

			\**************************************************************************/

			void device_set_async( CUDA_BIGNUM *d_x, CUDA_BIGNUM *h_x, uint32 num_bytes )
			{
				// copy values to the device
				CudaSafeCall(cudaMemcpyAsync(d_x, h_x, num_bytes, cudaMemcpyHostToDevice, g_stream_0));
			}

			void device_copy_async( CUDA_BIGNUM *d_y, CUDA_BIGNUM *d_x, uint32 num_bytes )	
			{
				// copy values from one device value to another
				CudaSafeCall(cudaMemcpyAsync(d_y, d_x, num_bytes, cudaMemcpyDeviceToDevice, g_stream_0));
			}

			void device_get_async( CUDA_BIGNUM *h_x, CUDA_BIGNUM *d_x, uint32 num_bytes )
			{
				CudaSafeCall(cudaMemcpyAsync(h_x, d_x, num_bytes, cudaMemcpyDeviceToHost, g_stream_0));
			}

			void device_hexDump( char *h_desc, void *d_addr, int len )
			{
				std::cout << h_desc << std::endl;
				std::cout << strlen(h_desc) << std::endl;
				char *d_desc;
				CudaSafeCall(cudaMalloc((void **)&d_desc, strlen(h_desc) + 1));
				CudaSafeCall(cudaMemcpy(d_desc, h_desc, strlen(h_desc) + 1, cudaMemcpyHostToDevice));
				cudaHexDumpKernel<<<1, 1>>>(d_desc, d_addr, len);
				CudaSafeCall(cudaFree(d_desc));
				cudaDeviceSynchronize();
				CudaCheckError();			
			}

			/**************************************************************************\

                               Addition

			\**************************************************************************/			

			double parAdd( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res )
			{
				uint32 num_threads = min(g_hMaxThreadsPerBlock, (uint32)num_chunks);
				uint32 num_blocks = ceil((double)num_chunks / g_hMaxThreadsPerBlock);

				double iStart, iElaps;
				iStart = cpuSecond();

				cudaParAddKernel<<<num_blocks, num_threads>>>(x, y, res, num_chunks);
				cudaDeviceSynchronize();

				iElaps = cpuSecond() - iStart;

				CudaCheckError();

				return iElaps;
			}

			double parAddLoop( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, uint32 num_threads, uint32 num_blocks, CUDA_BIGNUM *res )
			{
				double iStart, iElaps;
				iStart = cpuSecond();

				cudaParAddLoopKernel<<<num_blocks, num_threads>>>(x, y, res, num_chunks);
				cudaDeviceSynchronize();

				iElaps = cpuSecond() - iStart;

				CudaCheckError();

				return iElaps;				
			}

			double parAddTime( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res, uint32 *times )
			{
				uint32 num_threads = min(g_hMaxThreadsPerBlock, (uint32)num_chunks);
				uint32 num_blocks = ceil((double)num_chunks / g_hMaxThreadsPerBlock);

				// allocate memory for timer
				timing_stats *d_timer = NULL;
				CudaSafeCall(cudaMalloc((void **)&d_timer, sizeof(timing_stats)));

				double iStart, iElaps;
				iStart = cpuSecond();

				cudaParAddTimeKernel<<<num_blocks, num_threads>>>(x, y, res, num_chunks, d_timer);
				cudaDeviceSynchronize();

				iElaps = cpuSecond() - iStart;

				CudaCheckError();

				timing_stats h_timer;
				CudaSafeCall(cudaMemcpy(&h_timer, d_timer, sizeof(timing_stats), cudaMemcpyDeviceToHost));

				times[0] = (uint32)(h_timer.time2 - h_timer.time1);
				times[1] = (uint32)(h_timer.time3 - h_timer.time2);
				times[2] = (uint32)(h_timer.time4 - h_timer.time3);

				CudaSafeCall(cudaFree(d_timer));

				return iElaps;
			}

			double parAddWithEvents( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res )
			{
				uint32 num_threads = min(g_hMaxThreadsPerBlock, (uint32)num_chunks);
				uint32 num_blocks = ceil((double)num_chunks / g_hMaxThreadsPerBlock);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start);

				cudaParAddKernel<<<num_blocks, num_threads>>>(x, y, res, num_chunks);
				
				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double parAddOwnStream( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res )
			{
				uint32 num_threads = min(g_hMaxThreadsPerBlock, (uint32)num_chunks);
				uint32 num_blocks = ceil((double)num_chunks / g_hMaxThreadsPerBlock);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, g_stream_0);

				cudaParAddKernel<<<num_blocks, num_threads, 0, g_stream_0>>>(x, y, res, num_chunks);
				
				cudaEventRecord(stop, g_stream_0);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double parAddOwnStream1024Threads( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res )
			{
				uint32 num_threads = min(1024, (uint32)num_chunks);
				uint32 num_blocks = ceil((double)num_chunks / 1024);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, g_stream_0);

				cudaParAddKernel<<<num_blocks, num_threads, 0, g_stream_0>>>(x, y, res, num_chunks);
				
				cudaEventRecord(stop, g_stream_0);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double parAddOwnStream512Threads( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res )
			{
				uint32 num_threads = min(512, (uint32)num_chunks);
				uint32 num_blocks = ceil((double)num_chunks / 512);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, g_stream_0);

				cudaParAddKernel<<<num_blocks, num_threads, 0, g_stream_0>>>(x, y, res, num_chunks);
				
				cudaEventRecord(stop, g_stream_0);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double parAddOwnStream256Threads( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res )
			{
				uint32 num_threads = min(256, (uint32)num_chunks);
				uint32 num_blocks = ceil((double)num_chunks / 256);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, g_stream_0);

				cudaParAddKernel<<<num_blocks, num_threads, 0, g_stream_0>>>(x, y, res, num_chunks);
				
				cudaEventRecord(stop, g_stream_0);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}			

			double parAddOwnStream128Threads( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res )
			{
				uint32 num_threads = min(128, (uint32)num_chunks);
				uint32 num_blocks = ceil((double)num_chunks / 128);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, g_stream_0);

				cudaParAddKernel<<<num_blocks, num_threads, 0, g_stream_0>>>(x, y, res, num_chunks);
				
				cudaEventRecord(stop, g_stream_0);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}					

			double parAddMultiStreams( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res )
			{
				uint32 num_threads = min(g_hMaxThreadsPerBlock, (uint32)num_chunks);
				uint32 num_blocks = ceil((double)num_chunks / g_hMaxThreadsPerBlock);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, g_stream_0);

				cudaParAddKernel<<<num_blocks, num_threads, 0, g_stream_0>>>(x, y, res, num_chunks);
				
				cudaEventRecord(stop, g_stream_0);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double parAdd2OwnStream( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res )
			{
				uint32 num_threads = min(g_hMaxThreadsPerBlock, (uint32)ceil((double)num_chunks / 2));
				uint32 num_blocks = ceil((double)num_chunks / g_hMaxThreadsPerBlock);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, g_stream_0);

				cuda2ParAddKernel<<<num_blocks, num_threads, 0, g_stream_0>>>(x, y, res, num_chunks);
				
				cudaEventRecord(stop, g_stream_0);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double parAdd4OwnStream( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res )
			{
				uint32 num_threads = min(g_hMaxThreadsPerBlock, (uint32)ceil((double)num_chunks / 4));
				uint32 num_blocks = ceil((double)num_chunks / g_hMaxThreadsPerBlock);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, g_stream_0);

				cuda4ParAddKernel<<<num_blocks, num_threads, 0, g_stream_0>>>(x, y, res, num_chunks);
				
				cudaEventRecord(stop, g_stream_0);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			double parAdd8OwnStream( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res )
			{
				uint32 num_threads = min(g_hMaxThreadsPerBlock, (uint32)ceil((double)num_chunks / 8));
				uint32 num_blocks = ceil((double)num_chunks / g_hMaxThreadsPerBlock);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, g_stream_0);

				cuda8ParAddKernel<<<num_blocks, num_threads, 0, g_stream_0>>>(x, y, res, num_chunks);
				
				cudaEventRecord(stop, g_stream_0);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}	

			double parAddSharedMem( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res )
			{
				uint32 num_threads = min(g_hMaxThreadsPerBlock, (uint32)num_chunks);
				uint32 num_blocks = ceil((double)num_chunks / g_hMaxThreadsPerBlock);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaParAddKernel<<<1, 1>>>(x, y, res, num_chunks);

				cudaEventRecord(start, g_stream_0);

				cudaParAddKernel<<<num_blocks, num_threads, num_chunks*num_threads, g_stream_0>>>(x, y, res, num_chunks);
				
				cudaEventRecord(stop, g_stream_0);

				cudaEventSynchronize(stop);
				
				float iElaps;
				cudaEventElapsedTime(&iElaps, start, stop);

				CudaCheckError();

				return (double)iElaps;
			}

			/**************************************************************************\

                               Multiplication

			\**************************************************************************/			

			double parMul( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM* irred_poly, uint32 indx_mask_bit, CUDA_BIGNUM *res )
			{
				device_clear(res, num_chunks);

				double iStart, iElaps;
				iStart = cpuSecond();

				cudaParMulKernel<<<1, 1>>>(x, y, res, num_chunks, irred_poly, indx_mask_bit);	
				cudaDeviceSynchronize();

				iElaps = cpuSecond() - iStart;

				CudaCheckError();

				return iElaps;
			}

			void parRedImpl( CUDA_BIGNUM *a, uint32 numChunksA, CUDA_BIGNUM maskRed, CUDA_BIGNUM *res, uint32 numChunksRes)
			{
				// res(x) = a(x) % x^n -> only take n bits of the right side
				// First copy all chunks that remain from a to res
				device_copy(res, &a[numChunksA - numChunksRes], numChunksRes * CUDA_BIGNUM_SIZE_BYTES);
				
				// Then ajust the first chunk of res with the mask
				cudaMaskFirstChunkKernel<<<1, 1>>>(res, maskRed);
			}

			void parShiftLeftImpl( CUDA_BIGNUM *x, uint32 numChunks, uint32 numBitsToShift, CUDA_BIGNUM *res, uint32 numChunksRes )
			{
				dim3 threadsPerBlock(min(g_hMaxThreadsPerBlock, numChunksRes));
				dim3 numBlocks(ceil((double)numChunksRes / threadsPerBlock.x));

				uint32 numBitsToShiftInBlock = numBitsToShift % CUDA_BIGNUM_SIZE_BITS;
				uint32 blockOffset = (uint32)(numBitsToShift / CUDA_BIGNUM_SIZE_BITS);

				CUDA_BIGNUM maskRight = ~((CUDA_BIGNUM)0);
				if( CUDA_BIGNUM_SIZE_BITS - numBitsToShiftInBlock < CUDA_BIGNUM_SIZE_BITS ) 
					maskRight -= ((CUDA_BIGNUM)pow((double)2, (double)CUDA_BIGNUM_SIZE_BITS - numBitsToShiftInBlock) - 1);
				else
					maskRight = 0;
				CUDA_BIGNUM maskLeft = ~((CUDA_BIGNUM)0) - maskRight;

				cudaShiftLeftKernel<<<numBlocks, threadsPerBlock>>>(x, numChunks, numBitsToShiftInBlock, blockOffset, maskLeft, maskRight, res, numChunksRes);
			}

			void parShiftRightImpl( CUDA_BIGNUM *x, uint32 numChunks, uint32 numBitsToShift, CUDA_BIGNUM *res, uint32 numChunksRes )
			{
				dim3 threadsPerBlock(min(g_hMaxThreadsPerBlock, numChunksRes));
				dim3 numBlocks(ceil((double)numChunksRes / threadsPerBlock.x));

				uint32 numBitsToShiftInBlock = numBitsToShift % CUDA_BIGNUM_SIZE_BITS;
				uint32 blockOffset = (uint32)(numBitsToShift / CUDA_BIGNUM_SIZE_BITS);
				CUDA_BIGNUM maskLeft  =  (CUDA_BIGNUM)pow((double)2, (double)(numBitsToShiftInBlock)) - 1;
				CUDA_BIGNUM maskRight =  ~((CUDA_BIGNUM)0) - maskLeft;

				cudaShiftRightKernel<<<numBlocks, threadsPerBlock>>>(x, numChunks, numBitsToShiftInBlock, blockOffset, maskLeft, maskRight, res, numChunksRes);
			}

			void parGetElemDegreeImpl( CUDA_BIGNUM *x, uint32 numChunks, CUDA_BIGNUM *foundChunkIndex, CUDA_BIGNUM *elemDegree )
			{
				dim3 threadsPerBlock(min(g_hMaxThreadsPerBlock, numChunks));
				dim3 numBlocks(ceil((double)numChunks / threadsPerBlock.x));

				cudaGetFirstNonEmptyChunkIndexKernel<<<numBlocks, threadsPerBlock>>>(x, numChunks, foundChunkIndex);
				cudaCalcElemDegreeKernel<<<1, 1>>>(x, numChunks, foundChunkIndex, elemDegree);
			}

			void parMulChunkedImpl( CUDA_BIGNUM *x, CUDA_BIGNUM *y, CUDA_BIGNUM *tmpArr, uint32 numChunks, CUDA_BIGNUM *res )
			{
				device_clear(tmpArr, 4 * numChunks * numChunks);

				dim3 threadsPerBlock(4, 16);
				dim3 numBlocks(ceil((double)numChunks / threadsPerBlock.x), ceil((double)2 * numChunks / threadsPerBlock.y));

				cudaCreateChunkProdArrayKernel<<<numBlocks, threadsPerBlock>>>(x, y, tmpArr, numChunks);	

				dim3 threadsPerBlock2(64, 1); 
				dim3 numBlocks2(ceil((double)numChunks / threadsPerBlock2.x), 
					ceil((double)2 * numChunks / threadsPerBlock2.y));

				cudaParDiagBlockSumKernel<<<numBlocks2, threadsPerBlock2>>>(tmpArr, numChunks);

				dim3 threadsPerBlock3(min(g_hMaxThreadsPerBlock, (uint32)2 * numChunks));
				dim3 numBlocks3(ceil((double)2 * numChunks / threadsPerBlock3.x));

				cudaParChunkSumKernel<<<numBlocks3, threadsPerBlock3>>>(tmpArr, numChunks, numBlocks2.x, 2 * threadsPerBlock2.x, res);

				cudaDeviceSynchronize();

				CudaCheckError();
			}

			/*
				Q1(x) = A(x) / x^n -> Shift um n Bits nach rechts
				Q2(x) = M(x) * Q1(x) -> Erg der Größe 2n
				Q3(x) = Q2(x) / x^n -> Shift um n Bits nach rechts
				R1(x) = A(x) % x^n -> nur rechten n Bits
				R2(x) = M(x) * Q3(x) % x^n
				R(x) = R1(x) + R2(x)	
			*/
			void parBarRedImpl( CUDA_BIGNUM *a, uint32 numChunksA, CUDA_BIGNUM *tmpArr, CUDA_BIGNUM *tmp,
				CUDA_BIGNUM *q1, CUDA_BIGNUM *q2, CUDA_BIGNUM *q3, CUDA_BIGNUM *r1, CUDA_BIGNUM *r2, 
				CUDA_BIGNUM *irredPoly, uint32 fieldSize, 
				uint32 numBitsToShiftInBlock, uint32 blockOffset,
				CUDA_BIGNUM maskLeft, CUDA_BIGNUM maskRight,
				CUDA_BIGNUM *res, uint32 numChunksRes )
			{	
				dim3 threadsPerBlock(min(g_hMaxThreadsPerBlock, numChunksRes));
				dim3 numBlocks(ceil((double)numChunksRes / threadsPerBlock.x));

				// Q1(x) = A(x) / x^n -> shift n bits right
				parShiftRightImpl(a, numChunksA, fieldSize, q1, numChunksRes);
				//cudaShiftRightKernel<<<numBlocks, threadsPerBlock>>>(a, numChunksA, numBitsToShiftInBlock, blockOffset, maskLeft, maskRight, q1, numChunksRes);

				// Q2(x) = M(x) * Q1(x) -> q2 has 2n bits
				parMulChunkedImpl(q1, irredPoly, tmpArr, numChunksRes, q2);

				// Q3(x) = Q2(x) / x^n -> shift n bits left
				parShiftRightImpl(q2, numChunksA, fieldSize, q3, numChunksRes);
				//cudaShiftRightKernel<<<numBlocks, threadsPerBlock>>>(q2, numChunksA, numBitsToShiftInBlock, blockOffset, maskLeft, maskRight, q3, numChunksRes);

				// R1(x) = A(x) % x^n -> take n lsb bits
				parRedImpl(a, numChunksA, maskLeft, r1, numChunksRes);

				// R2(x) = M(x) * Q3(x) % x^n
				parMulChunkedImpl(q3, irredPoly, tmpArr, numChunksRes, tmp);
				parRedImpl(tmp, numChunksA, maskLeft, r2, numChunksRes);

				// R(x) = R1(x) + R2(x)
				cudaParAddKernel<<<numBlocks, threadsPerBlock>>>(r1, r2, res, numChunksRes );

				cudaDeviceSynchronize();

				CudaCheckError();
			}

			void parInverseElementImpl( CUDA_BIGNUM *x, uint32 numChunks, 
				CUDA_BIGNUM *irredPoly, CUDA_BIGNUM *tmp,
				CUDA_BIGNUM *s, CUDA_BIGNUM *r, CUDA_BIGNUM *v, CUDA_BIGNUM *u,
				CUDA_BIGNUM *degR, CUDA_BIGNUM *degS, CUDA_BIGNUM *foundChunkIndex, 
				CUDA_BIGNUM *res )
			{
				dim3 threadsPerBlock(min(g_hMaxThreadsPerBlock, numChunks));
				dim3 numBlocks(ceil((double)numChunks / threadsPerBlock.x));

				// S(x) = G(x)
				device_copy(s, irredPoly, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				// R(X) = A(x)
				device_copy(r, x, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				// V(x) = 0
				device_clear(v, numChunks);
				// U(x) = 1
				cudaSet1Kernel<<<numBlocks, threadsPerBlock>>>(u, numChunks);

				CUDA_BIGNUM h_degR, h_degS, h_degDelta;

				// degR = deg(R(x))
				parGetElemDegreeImpl(r, numChunks, foundChunkIndex, degR);
				device_get(&h_degR, degR, CUDA_BIGNUM_SIZE_BYTES);

				while( h_degR )
				{
					parGetElemDegreeImpl(s, numChunks, foundChunkIndex, degS);
					device_get(&h_degS, degS, CUDA_BIGNUM_SIZE_BYTES);

					if( h_degR > h_degS )
					{
						// printf("hit swap\n");
						// swap pointers
						CUDA_BIGNUM *ptrTmp = NULL;
						// temp := S(x); S(x) := R(x); R(x) := temp;
						ptrTmp = s; s = r, r = ptrTmp;
						// temp := V (x); V (x) := U(x); U(x) := temp;
						ptrTmp = v; v = u; u = ptrTmp;
						
						h_degDelta = h_degR - h_degS;
					}
					else
					{
						h_degDelta = h_degS - h_degR;
					}

					// S(x) = S(x) − x^degDelta * R(x)
					device_clear(tmp, numChunks);
					parShiftLeftImpl(r, numChunks, h_degDelta, tmp, numChunks);
					cudaParAddKernel<<<numBlocks, threadsPerBlock>>>(s, tmp, numChunks);

					// V(x) = V(x) - x^degDelta * U(x)
					device_clear(tmp, numChunks);
					parShiftLeftImpl(u, numChunks, h_degDelta, tmp, numChunks);
					cudaParAddKernel<<<numBlocks, threadsPerBlock>>>(v, tmp, numChunks);

					// degR = deg(R(x))
					parGetElemDegreeImpl(r, numChunks, foundChunkIndex, degR);
					device_get(&h_degR, degR, CUDA_BIGNUM_SIZE_BYTES);
				}

				cudaDeviceSynchronize();

				// return U(x) as the result
				device_copy(res, u, CUDA_BIGNUM_SIZE_BYTES * numChunks);

				CudaCheckError();
			}

			double parMulChunkedBarRed( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 numChunks, CUDA_BIGNUM *irredPoly, uint32 fieldSize, CUDA_BIGNUM *res )
			{
				double iStart, iElaps;

				uint32 matrixWidth = 2 * numChunks;
				uint32 matrixHeight = 2 * numChunks;

				device_clear(res, numChunks);

				// allocate result array
				CUDA_BIGNUM *arr = NULL;
				device_allocate(&arr, CUDA_BIGNUM_SIZE_BYTES * matrixWidth * matrixHeight);

				// allocate space for tmp result befor reduction
				// of size 2 * num_chunks
				CUDA_BIGNUM *tmp, *resTmp = NULL;
				device_allocate(&resTmp, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);
				device_allocate(&tmp, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);

				CUDA_BIGNUM *q1, *q2, *q3, *r1, *r2 = NULL;
				device_allocate(&q1, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&q2, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);
				device_allocate(&q3, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&r1, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&r2, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);

				uint32 numBitsToShiftInBlock = fieldSize % CUDA_BIGNUM_SIZE_BITS;
				uint32 blockOffset = (uint32)(fieldSize / CUDA_BIGNUM_SIZE_BITS);
				CUDA_BIGNUM maskLeft  =  (CUDA_BIGNUM)pow((double)2, (double)(numBitsToShiftInBlock)) - 1;
				CUDA_BIGNUM maskRight =  ~((CUDA_BIGNUM)0) - maskLeft;

				// {
				// 	uint32 *foundChunkIndex, *elemDegree;
				// 	device_allocate(&foundChunkIndex, 4);
				// 	device_allocate(&elemDegree, 4);

				// 	uint32 h_elemDegree;

				// 	parGetElemDegreeImpl(x, numChunks, foundChunkIndex, elemDegree);
				// 	device_get(&h_elemDegree, elemDegree, 4);

				// 	std::cout << "element has degree " << h_elemDegree << std::endl;

				// 	device_delete(elemDegree);
				// 	device_delete(foundChunkIndex);
				// }
				
				//Test Left Shift
				{
					CUDA_BIGNUM *tmp_x = NULL;
					device_allocate(&tmp_x, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);

					parShiftLeftImpl(x, numChunks, fieldSize, tmp_x, 2 * numChunks);
					parShiftRightImpl(tmp_x, 2 * numChunks, fieldSize, x, numChunks);

					cudaDeviceSynchronize();
					
					device_delete(tmp_x);
				}

				iStart = cpuSecond();

				parMulChunkedImpl(x, y, arr, numChunks, resTmp);
				parBarRedImpl(resTmp, matrixHeight, arr, tmp, q1, q2, q3, r1, r2, irredPoly, fieldSize, numBitsToShiftInBlock, blockOffset, maskLeft, maskRight, res, numChunks);

				iElaps = cpuSecond() - iStart;

				device_delete(q1);
				device_delete(q2);
				device_delete(q3);
				device_delete(r1);
				device_delete(r2);

				device_delete(tmp);
				device_delete(resTmp);
				device_delete(arr);

				return iElaps;				
			}		

			/**************************************************************************\

                               Exponentiation

			\**************************************************************************/

            double parExponentiation( CUDA_BIGNUM *x, uint32 k, uint32 numChunks, CUDA_BIGNUM *irredPoly, uint32 fieldSize, CUDA_BIGNUM *res )
            {
				double iStart, iElaps;

				uint32 matrixWidth = 2 * numChunks;
				uint32 matrixHeight = 2 * numChunks;

				device_clear(res, CUDA_BIGNUM_SIZE_BYTES * numChunks);

				// allocate result array
				CUDA_BIGNUM *arr = NULL;
				device_allocate(&arr, CUDA_BIGNUM_SIZE_BYTES * matrixWidth * matrixHeight);

				// allocate space for tmp result befor reduction
				// of size 2 * num_chunks
				CUDA_BIGNUM *tmp, *resTmp = NULL;
				device_allocate(&resTmp, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);
				device_allocate(&tmp, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);

				CUDA_BIGNUM *q1, *q2, *q3, *r1, *r2 = NULL;
				device_allocate(&q1, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&q2, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);
				device_allocate(&q3, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&r1, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&r2, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);

				uint32 numBitsToShiftInBlock = fieldSize % CUDA_BIGNUM_SIZE_BITS;
				uint32 blockOffset = (uint32)(fieldSize / CUDA_BIGNUM_SIZE_BITS);
				CUDA_BIGNUM maskLeft  =  (CUDA_BIGNUM)pow((double)2, (double)(numBitsToShiftInBlock)) - 1;
				CUDA_BIGNUM maskRight =  ~((CUDA_BIGNUM)0) - maskLeft;

				iStart = cpuSecond();

				if( !k )
				{
					dim3 threadsPerBlock(min(g_hMaxThreadsPerBlock, numChunks));
					dim3 numBlocks(ceil((double)numChunks / threadsPerBlock.x));
					cudaSet1Kernel<<<numBlocks, threadsPerBlock>>>(res, numChunks);
				}
				else
				{
					// Calculate the mask 
				    uint32 h = ~(~0u >> 1); 
	    			while( !(k & h) ) 
	    				h >>= 1;

	    			// initial copy x to res
	    			device_copy(res, x, CUDA_BIGNUM_SIZE_BYTES * numChunks);

	    			// As long as the mask isn't 0 -> do exponentiation
				    while( h >>= 1 )
				    {
				    	// Calculate res = res^2
				    	parMulChunkedImpl(res, res, arr, numChunks, resTmp);
						parBarRedImpl(resTmp, matrixHeight, arr, tmp, q1, q2, q3, r1, r2, irredPoly, fieldSize, numBitsToShiftInBlock, blockOffset, maskLeft, maskRight, res, numChunks);

						// If bit is set -> calculate res = res * x
				        if( k & h )
				        {
				        	parMulChunkedImpl(res, x, arr, numChunks, resTmp);
							parBarRedImpl(resTmp, matrixHeight, arr, tmp, q1, q2, q3, r1, r2, irredPoly, fieldSize, numBitsToShiftInBlock, blockOffset, maskLeft, maskRight, res, numChunks);
				        }
				    }
				}

				iElaps = cpuSecond() - iStart;

				device_delete(q1);
				device_delete(q2);
				device_delete(q3);
				device_delete(r1);
				device_delete(r2);

				device_delete(tmp);
				device_delete(resTmp);
				device_delete(arr);

				return iElaps;	
            }

			/**************************************************************************\

                               Inverse

			\**************************************************************************/

			double parInverseElement( CUDA_BIGNUM *x, uint32 numChunks, CUDA_BIGNUM *irredPoly, CUDA_BIGNUM *res )
			{
				double iStart, iElaps;

				CUDA_BIGNUM *tmp, *s, *r, *v, *u;
				device_allocate(&tmp, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&s, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&r, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&v, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&u, CUDA_BIGNUM_SIZE_BYTES * numChunks);

				CUDA_BIGNUM *degR, *degS, *foundChunkIndex;
				device_allocate(&degR, CUDA_BIGNUM_SIZE_BYTES);
				device_allocate(&degS, CUDA_BIGNUM_SIZE_BYTES);
				device_allocate(&foundChunkIndex, CUDA_BIGNUM_SIZE_BYTES);

				iStart = cpuSecond();

				parInverseElementImpl(x, numChunks, irredPoly, tmp, s, r, v, u, degR, degS, foundChunkIndex, res);

				iElaps = cpuSecond() - iStart;

				device_delete(foundChunkIndex);
				device_delete(degS);
				device_delete(degR);
				
				device_delete(u);
				device_delete(v);
				device_delete(r);
				device_delete(s);
				device_delete(tmp);

				return iElaps;
			}

			double parInverseElementWithExp( CUDA_BIGNUM *x, uint32 numChunks, CUDA_BIGNUM *irredPoly, uint32 fieldSize, CUDA_BIGNUM *res )
            {
				double iStart, iElaps;

				uint32 matrixWidth = 2 * numChunks;
				uint32 matrixHeight = 2 * numChunks;

				device_clear(res, CUDA_BIGNUM_SIZE_BYTES * numChunks);

				// allocate result array
				CUDA_BIGNUM *arr = NULL;
				device_allocate(&arr, CUDA_BIGNUM_SIZE_BYTES * matrixWidth * matrixHeight);

				// allocate space for tmp result befor reduction
				// of size 2 * num_chunks
				CUDA_BIGNUM *tmp, *resTmp = NULL;
				device_allocate(&resTmp, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);
				device_allocate(&tmp, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);

				CUDA_BIGNUM *q1, *q2, *q3, *r1, *r2 = NULL;
				device_allocate(&q1, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&q2, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);
				device_allocate(&q3, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&r1, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&r2, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);

				uint32 numBitsToShiftInBlock = fieldSize % CUDA_BIGNUM_SIZE_BITS;
				uint32 blockOffset = (uint32)(fieldSize / CUDA_BIGNUM_SIZE_BITS);
				CUDA_BIGNUM maskLeft  =  (CUDA_BIGNUM)pow((double)2, (double)(numBitsToShiftInBlock)) - 1;
				CUDA_BIGNUM maskRight =  ~((CUDA_BIGNUM)0) - maskLeft;

				iStart = cpuSecond();

    			// initial copy x to res
    			device_copy(res, x, CUDA_BIGNUM_SIZE_BYTES * numChunks);

    			// Mask is always (2^fieldSize)-2
			    for( int i=0; i<fieldSize - 2; ++i )
			    {
			    	// Calculate res = res^2
			    	parMulChunkedImpl(res, res, arr, numChunks, resTmp);
					parBarRedImpl(resTmp, matrixHeight, arr, tmp, q1, q2, q3, r1, r2, irredPoly, fieldSize, numBitsToShiftInBlock, blockOffset, maskLeft, maskRight, res, numChunks);

					// Calculate res = res * x
			        parMulChunkedImpl(res, x, arr, numChunks, resTmp);
					parBarRedImpl(resTmp, matrixHeight, arr, tmp, q1, q2, q3, r1, r2, irredPoly, fieldSize, numBitsToShiftInBlock, blockOffset, maskLeft, maskRight, res, numChunks);
			    }

			    // Calculate res = res^2
			    parMulChunkedImpl(res, res, arr, numChunks, resTmp);
				parBarRedImpl(resTmp, matrixHeight, arr, tmp, q1, q2, q3, r1, r2, irredPoly, fieldSize, numBitsToShiftInBlock, blockOffset, maskLeft, maskRight, res, numChunks);

				iElaps = cpuSecond() - iStart;

				device_delete(q1);
				device_delete(q2);
				device_delete(q3);
				device_delete(r1);
				device_delete(r2);

				device_delete(tmp);
				device_delete(resTmp);
				device_delete(arr);

				return iElaps;	
            }

            /**************************************************************************\

                               Polynomial Evaluation

			\**************************************************************************/

            double parEvaluatePoly( CUDA_BIGNUM *coeffs, uint32 numCoeffs, CUDA_BIGNUM *x, uint32 numChunks, CUDA_BIGNUM *irredPoly, uint32 fieldSize, CUDA_BIGNUM *res )
            {
				double iStart, iElaps;

				uint32 matrixWidth = 2 * numChunks;
				uint32 matrixHeight = 2 * numChunks;

				device_clear(res, CUDA_BIGNUM_SIZE_BYTES * numChunks);

				// allocate result array
				CUDA_BIGNUM *arr = NULL;
				device_allocate(&arr, CUDA_BIGNUM_SIZE_BYTES * matrixWidth * matrixHeight);

				// allocate space for tmp result befor reduction
				// of size 2 * num_chunks
				CUDA_BIGNUM *tmp, *resTmp, *xTmp, *resTmp2 = NULL;
				device_allocate(&resTmp, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);
				device_allocate(&tmp, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);
				device_allocate(&xTmp, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&resTmp2, CUDA_BIGNUM_SIZE_BYTES * numChunks);

				CUDA_BIGNUM *q1, *q2, *q3, *r1, *r2 = NULL;
				device_allocate(&q1, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&q2, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);
				device_allocate(&q3, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&r1, CUDA_BIGNUM_SIZE_BYTES * numChunks);
				device_allocate(&r2, CUDA_BIGNUM_SIZE_BYTES * 2 * numChunks);

				uint32 numBitsToShiftInBlock = fieldSize % CUDA_BIGNUM_SIZE_BITS;
				uint32 blockOffset = (uint32)(fieldSize / CUDA_BIGNUM_SIZE_BITS);
				CUDA_BIGNUM maskLeft  =  (CUDA_BIGNUM)pow((double)2, (double)(numBitsToShiftInBlock)) - 1;
				CUDA_BIGNUM maskRight =  ~((CUDA_BIGNUM)0) - maskLeft;

				iStart = cpuSecond();

    			// initial xTmp = x^0
    			dim3 threadsPerBlock(min(g_hMaxThreadsPerBlock, numChunks));
				dim3 numBlocks(ceil((double)numChunks / threadsPerBlock.x));

				// Initialize xTmp with 1
				cudaSet1Kernel<<<numBlocks, threadsPerBlock>>>(xTmp, numChunks);

				// Set res to last coeff
				device_copy(res, &coeffs[(numCoeffs - 1) * numChunks], CUDA_BIGNUM_SIZE_BYTES * numChunks);

    			// Iterate over all coeffs
			    for( int i=numCoeffs - 2; i>=0; --i )
			    {
			    	// Calculate xTmp = xTmp * x
					parMulChunkedImpl(xTmp, x, arr, numChunks, resTmp);
					parBarRedImpl(resTmp, matrixHeight, arr, tmp, q1, q2, q3, r1, r2, irredPoly, fieldSize, numBitsToShiftInBlock, blockOffset, maskLeft, maskRight, xTmp, numChunks);

			    	// Calculate resTmp2 = coeffs[i] * xTmp
			    	parMulChunkedImpl(&coeffs[i * numChunks], xTmp, arr, numChunks, resTmp);
					parBarRedImpl(resTmp, matrixHeight, arr, tmp, q1, q2, q3, r1, r2, irredPoly, fieldSize, numBitsToShiftInBlock, blockOffset, maskLeft, maskRight, resTmp2, numChunks);

					// Calculate res = res + resTmp2
					cudaParAddKernel<<<numBlocks, threadsPerBlock>>>(res, resTmp2, numChunks);
			    }

				iElaps = cpuSecond() - iStart;

				device_delete(q1);
				device_delete(q2);
				device_delete(q3);
				device_delete(r1);
				device_delete(r2);

				device_delete(resTmp2);
				device_delete(xTmp);
				device_delete(tmp);
				device_delete(resTmp);
				device_delete(arr);

				return iElaps;	
            }

			/**************************************************************************\

                               Helper

			\**************************************************************************/

			double createChunkProdArray( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM **res )	
			{
				double iStart, iElaps;
				iStart = cpuSecond();

				cudaCreateChunkProdArrayKernel<<<1, 1>>>(x, y, *res, num_chunks);
				cudaDeviceSynchronize();

				iElaps = cpuSecond() - iStart;

				CudaCheckError();

				return iElaps;
			}

			double measureKernelLaunchOverhead()
			{
				double iStart, iElaps;
				iStart = cpuSecond();

				EmptyKernel<<<1, 1>>>();	
				cudaDeviceSynchronize();

				iElaps = cpuSecond() - iStart;

				CudaCheckError();

				return iElaps;	
			}


			void print( CUDA_BIGNUM *x, uint32 num_chunks )
			{
				cudaPrintKernel<<<1, num_chunks>>>(x, num_chunks);
				CudaCheckError();
			}

		}
	}
}
