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

#ifndef __GF2N_ARITHMETIC_CUDA_DATAPOOL_H__
#define __GF2N_ARITHMETIC_CUDA_DATAPOOL_H__

#include <exception>
#include <map>
#include <list>
#include <memory>

#include "CudaBignum.h"

namespace libcumffa {
	namespace gpu {		

		class EmptyDataPoolException : public std::exception 
		{
			virtual const char* what() const throw() 
			{
				return "The datapool has no more free elements!!!";
			}
		};

		class InvalidDataPoolElementException : public std::exception
		{
			virtual const char* what() const throw()
			{
				return "The datapool element is not valid anymore!!!";
			}
		};

		class GF2nArithmeticCudaDataPoolElement;

		class GF2nArithmeticCudaDataPool
		{
			friend class GF2nArithmeticCudaDataPoolElementData;
		public:
			GF2nArithmeticCudaDataPool();
			/**
			 * @brief      Creates a new data pool
			 *
			 * @param[in]  num_bytes       the size of a data
			 * 							   pool element in bytes
			 * @param[in]  data_pool_size  the number of data pool elements
			 * 							   that will be created
			 */
			GF2nArithmeticCudaDataPool( uint32 num_bytes, uint32 data_pool_size );
			~GF2nArithmeticCudaDataPool();
			/**
			 * @brief      Returns the next free data pool element
			 *
			 * @return     the next free data pool element
			 * 
			 * @throw    EmptyDataPoolException This exception is thrown if now more 
			 * 									free elements are available in the data pool
			 */
			GF2nArithmeticCudaDataPoolElement get();
		private:
			void free( uint32 pool_element );
		private:
			// Size of a data pool elmenent
			uint32 m_h_num_bytes;

			// Current size of the data pool
			uint32 m_h_data_pool_size;

			// Pointer to the first element of the allocated data pool
			CUDA_BIGNUM *m_d_data_pool;

			// Contains the indices of all free data pool elements
			std::list<CUDA_BIGNUM> m_h_data_pool_free;

			// Contains the indices of all data pool elements that are
			// currently in use
			std::list<CUDA_BIGNUM> m_h_data_pool_used;

			// Contains a map of all currently useed data pool elements.
			// All elements are GF2nArithmeticCudaDataPoolElementData types.
			// Elements can be accessed by there data pool index.
			std::map<CUDA_BIGNUM, void*> m_h_data_pool_refs;
		};

		class GF2nArithmeticCudaDataPoolElementData
		{
		public:
			GF2nArithmeticCudaDataPoolElementData( CUDA_BIGNUM *data, uint32 pool_element, GF2nArithmeticCudaDataPool *parent );
			~GF2nArithmeticCudaDataPoolElementData();
			void invalidate();
			CUDA_BIGNUM *getData();
		private:
			CUDA_BIGNUM *m_data;
			CUDA_BIGNUM m_pool_element;
			GF2nArithmeticCudaDataPool *m_parent;
			bool m_isValid;
		};

		class GF2nArithmeticCudaDataPoolElement
		{
		public:
			GF2nArithmeticCudaDataPoolElement( GF2nArithmeticCudaDataPoolElementData *element );
			~GF2nArithmeticCudaDataPoolElement();
			CUDA_BIGNUM *operator*() const;
			operator bool();
		private:
			std::shared_ptr<GF2nArithmeticCudaDataPoolElementData> m_element;
		};
	}
}

#endif //__GF2N_ARITHMETIC_CUDA_DATAPOOL_H__