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

#include "GF2nArithmeticCudaDataPool.h"
#include "GF2nArithmeticCudaWrapper.h"

namespace libcumffa {
	namespace gpu {

		GF2nArithmeticCudaDataPool::GF2nArithmeticCudaDataPool()
		: m_h_num_bytes(0)
		, m_h_data_pool_size(0) {}

		GF2nArithmeticCudaDataPool::GF2nArithmeticCudaDataPool( uint32 num_bytes, uint32 data_pool_size ) 
		: m_h_num_bytes(num_bytes)
		, m_h_data_pool_size(data_pool_size)
		{
			cuda::device_allocate(&m_d_data_pool, m_h_num_bytes * m_h_data_pool_size);
			for( uint32 i=0; i<data_pool_size; ++i ) 
			{
				m_h_data_pool_free.push_back(i);
			}
		}

		GF2nArithmeticCudaDataPool::~GF2nArithmeticCudaDataPool()
		{
			cuda::device_delete(m_d_data_pool);
			
			for( std::map<CUDA_BIGNUM, void *>::iterator it = m_h_data_pool_refs.begin(); it != m_h_data_pool_refs.end(); ++it )
			{
				reinterpret_cast<GF2nArithmeticCudaDataPoolElementData *>(it->second)->invalidate();	
				m_h_data_pool_refs.erase(it->first);
			}
			m_h_data_pool_free.clear();
			m_h_data_pool_used.clear();
		}

		GF2nArithmeticCudaDataPoolElement GF2nArithmeticCudaDataPool::get() 
		{
			// TODO: no free data element available? -> create more?
			if( m_h_data_pool_free.empty() )
				throw EmptyDataPoolException();	

			CUDA_BIGNUM next_data_pool_el = m_h_data_pool_free.front();
			m_h_data_pool_used.push_back(next_data_pool_el);
			m_h_data_pool_free.pop_front();

			GF2nArithmeticCudaDataPoolElementData *newData = new GF2nArithmeticCudaDataPoolElementData(&m_d_data_pool[next_data_pool_el * m_h_num_bytes / sizeof(CUDA_BIGNUM)], next_data_pool_el, this);
			m_h_data_pool_refs[next_data_pool_el] = newData;
			return GF2nArithmeticCudaDataPoolElement(newData);
		}		

		void GF2nArithmeticCudaDataPool::free( uint32 pool_element )
		{
			m_h_data_pool_used.remove(pool_element);
			m_h_data_pool_refs.erase(pool_element);
			m_h_data_pool_free.push_back(pool_element);
		}

		///////////////////////////////////////////////////////////////////////
		/*
			implementations of GF2nArithmeticCudaDataPoolElementData
		*/
		GF2nArithmeticCudaDataPoolElementData::GF2nArithmeticCudaDataPoolElementData( CUDA_BIGNUM *data, uint32 pool_element, GF2nArithmeticCudaDataPool *parent )
		: m_data(data)
		, m_pool_element(pool_element)
		, m_parent(parent) 
		, m_isValid(true)
		{
		}


		GF2nArithmeticCudaDataPoolElementData::~GF2nArithmeticCudaDataPoolElementData()
		{
			if( m_isValid )
			{
				m_parent->free(m_pool_element);
			}
		}

		void GF2nArithmeticCudaDataPoolElementData::invalidate()
		{
			m_isValid = false;
		}

		CUDA_BIGNUM *GF2nArithmeticCudaDataPoolElementData::getData()
		{
			return m_data;
		}

		///////////////////////////////////////////////////////////////////////
		/*
			implementations of GF2nArithmeticCudaDataPoolElement
		*/
		GF2nArithmeticCudaDataPoolElement::GF2nArithmeticCudaDataPoolElement( GF2nArithmeticCudaDataPoolElementData *element )
		{
			m_element.reset(element);
		}

		GF2nArithmeticCudaDataPoolElement::~GF2nArithmeticCudaDataPoolElement()
		{
		}

		CUDA_BIGNUM *GF2nArithmeticCudaDataPoolElement::operator*() const
		{
			return m_element->getData();
		}		

		GF2nArithmeticCudaDataPoolElement::operator bool()
		{
			return (m_element != NULL);
		}
	}
}