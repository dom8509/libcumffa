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

#ifndef __GF2N_ARITHMETIC_CUDA_WRAPPER_H__
#define __GF2N_ARITHMETIC_CUDA_WRAPPER_H__

#include "CudaBignum.h"

namespace libcumffa {
	namespace gpu {

		namespace cuda {

			/**************************************************************************\

                               Core functions

			\**************************************************************************/
			void device_init();
			double device_allocate( CUDA_BIGNUM **d_x, uint32 num_bytes );
			double device_allocate_pinned( CUDA_BIGNUM **d_x, uint32 num_bytes );
			double device_delete( CUDA_BIGNUM *d_x );
			double device_delete_pinned( CUDA_BIGNUM *d_x );
			double device_set( CUDA_BIGNUM *d_x, CUDA_BIGNUM *h_x, uint32 num_bytes );
			double device_swapBytes( CUDA_BIGNUM *d_x, uint32 num_chunks );
			double device_copy( CUDA_BIGNUM *d_y, CUDA_BIGNUM *d_x, uint32 num_bytes );
			double device_get( CUDA_BIGNUM *h_x, CUDA_BIGNUM *d_x, uint32 num_bytes );
			double device_clear( CUDA_BIGNUM *d_x, uint32 num_chunks );
			void device_set_async( CUDA_BIGNUM *d_x, CUDA_BIGNUM *h_x, uint32 num_bytes );
			void device_copy_async( CUDA_BIGNUM *d_y, CUDA_BIGNUM *d_x, uint32 num_bytes );
			void device_get_async( CUDA_BIGNUM *h_x, CUDA_BIGNUM *d_x, uint32 num_bytes );
			void device_hexDump( char *h_desc, void *d_addr, int len );


			/**************************************************************************\

                               Addition

			\**************************************************************************/
			double parAdd( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res );
			double parAddLoop( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, uint32 num_threads, uint32 num_blocks, CUDA_BIGNUM *res );
			double parAddTime( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res, uint32 *times );
			double parAddWithEvents( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res );
			double parAddOwnStream( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res );
			double parAddOwnStream1024Threads( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res );
			double parAddOwnStream512Threads( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res );
			double parAddOwnStream256Threads( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res );
			double parAddOwnStream128Threads( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res );
			double parAdd2OwnStream( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res );
			double parAdd4OwnStream( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res );
			double parAdd8OwnStream( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res );
			double parAddSharedMem( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *res );


			/**************************************************************************\

                               Multiplication

			\**************************************************************************/
			double parMul( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM *irred_poly, uint32 indx_mask_bit, CUDA_BIGNUM *res );
			double parMulChunkedBarRed( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 numChunks, CUDA_BIGNUM *irredPoly, uint32 fieldSize, CUDA_BIGNUM *res );


			/**************************************************************************\

                               Exponentiation

			\**************************************************************************/
			double parExponentiation( CUDA_BIGNUM *x, uint32 k, uint32 numChunks, CUDA_BIGNUM *irredPoly, uint32 fieldSize, CUDA_BIGNUM *res );


			/**************************************************************************\

                               Inverse

			\**************************************************************************/
			double parInverseElement( CUDA_BIGNUM *x, uint32 numChunks, CUDA_BIGNUM *irredPoly, CUDA_BIGNUM *res );
			double parInverseElementWithExp( CUDA_BIGNUM *x, uint32 numChunks, CUDA_BIGNUM *irredPoly, uint32 fieldSize, CUDA_BIGNUM *res );
			
			
			/**************************************************************************\

                               Polynomial Evaluation

			\**************************************************************************/
			double parEvaluatePoly( CUDA_BIGNUM *coeffs, uint32 numCoeffs, CUDA_BIGNUM *x, uint32 numChunks, CUDA_BIGNUM *irredPoly, uint32 fieldSize, CUDA_BIGNUM *res );

			/**************************************************************************\

                               	Helper

			\**************************************************************************/
			double createChunkProdArray( CUDA_BIGNUM *x, CUDA_BIGNUM *y, uint32 num_chunks, CUDA_BIGNUM **res )	;
			double measureKernelLaunchOverhead();
			void print( CUDA_BIGNUM *x, uint32 num_chunks );
		}
		
	}
}

#endif //__GF2N_ARITHMETIC_CUDA_WRAPPER_H__