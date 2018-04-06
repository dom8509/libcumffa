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

#include <cstring>
#include <cassert>
#include <sstream>
#include <arpa/inet.h>

#include "../include/GF2nArithmeticCuda.h"
#include "../kernels/GF2nArithmeticCudaWrapper.h"


void hexDump2( char *desc, void *addr, int len ) 
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

void set_irrep_cuda(std::vector<ufixn> &, uint32);

//cuda
namespace libcumffa {
	namespace gpu {

		void cudaPrintbincharpad( ufixn* ca, unsigned int n )
		{
			for( unsigned int j=0; j<n; j++ ) 
			{
				ufixn c = ca[j];
			    for( int i = sizeof(ufixn)*8-1; i >= 0; --i )
			    {
			        if( c & (1 << i) )
			     	{
						printf("%c", '1');
			     	} 
					else
					{
						printf("%c", '0');
					}
			    }
			    printf("%c", ' ');
			}
			printf("\n");
		}

		///////////////////////////////////////////////////////////////////////
		/*
			implementations of GF2nArithmeticCuda
		*/
		GF2nArithmeticCuda::GF2nArithmeticCuda()
			: m_h_field_size(0)
			, m_h_num_chunks(0)
			, m_h_num_bytes(0)
			, m_h_indx_mask_bit(0)
			, m_device_initialized(false) {}

		GF2nArithmeticCuda::~GF2nArithmeticCuda()
		{
			clearDevice();
		}

		void GF2nArithmeticCuda::setFieldSize( const uint32 field_size )
		{
			if( m_h_field_size != field_size )
			{
				initChunkSizes(field_size);

				if( !m_h_irred_poly.empty() )
					m_h_irred_poly.clear();

				set_irrep_cuda(m_h_irred_poly, m_h_field_size);
			}

			initDevice();
		}

		/*
			set dummy parameters for testing
		*/		
		void GF2nArithmeticCuda::setDummyParameters( const uint32 field_size, const std::string irred_poly )
		{
			if( m_h_field_size != field_size )
				initChunkSizes(field_size);

			if( !m_h_irred_poly.empty() )
				m_h_irred_poly.clear();

			std::vector<std::string> str_arr_irred_poly;
			utils::convertStringToArray(irred_poly, sizeof(CUDA_BIGNUM) * 8, m_h_num_chunks, str_arr_irred_poly);

			if( m_h_num_chunks != str_arr_irred_poly.size() )
				throw WrongIrredPolyException();

			for( auto elem : str_arr_irred_poly )
				m_h_irred_poly.push_back(atoui64(elem));

			initDevice();
		}

		void GF2nArithmeticCuda::setDummyParameters( const uint32 field_size, const unsigned char *irred_poly, const uint32 chunks_irred_poly )
		{
			if( m_h_field_size != field_size )
				initChunkSizes(field_size);

			if( !m_h_irred_poly.empty() )
				m_h_irred_poly.clear();

			m_h_irred_poly.resize(m_h_num_chunks, 0);

			assert(chunks_irred_poly <= m_h_num_chunks * CUDA_BIGNUM_SIZE_BYTES);

			std::copy_backward(irred_poly, irred_poly + chunks_irred_poly, ((unsigned char*)&m_h_irred_poly[m_h_num_chunks]));
			for(uint32 i=0; i<m_h_num_chunks; ++i) m_h_irred_poly[i] = ntohl(m_h_irred_poly[i]);

			initDevice();
		}

		void GF2nArithmeticCuda::setDummyParameters( const uint32 field_size, const void *irred_poly, const uint32 chunks_irred_poly )
		{
			if( m_h_field_size != field_size )
				initChunkSizes(field_size);

			if( !m_h_irred_poly.empty() )
				m_h_irred_poly.clear();

			m_h_irred_poly.resize(m_h_num_chunks, 0);

			assert(chunks_irred_poly <= m_h_num_chunks * CUDA_BIGNUM_SIZE_BYTES);

			CUDA_BIGNUM *irred_poly_p = (CUDA_BIGNUM *)irred_poly;

			std::copy(irred_poly_p, irred_poly_p + chunks_irred_poly, &m_h_irred_poly[0]);

			initDevice();
		}		

		void GF2nArithmeticCuda::setFlags( const unsigned char flags )
		{
			m_async = (flags & 0x2) > 0;
		}

		GF2nArithmeticElement GF2nArithmeticCuda::getElement( const std::string value )
		{
			std::vector<std::string> str_arr_value;
			utils::convertStringToArray(value, CUDA_BIGNUM_SIZE_BITS, 
				utils::calcNumberChunks<uint32>((CUDA_BIGNUM)value.length(), CUDA_BIGNUM_SIZE_BITS), str_arr_value);

			CUDA_BIGNUM* vec_value = new CUDA_BIGNUM[m_h_num_bytes];
			uint32 i = 0;

			for( auto str_arr_value_elem : str_arr_value )
				vec_value[i++] = atoui64(str_arr_value_elem);

			GF2nArithmeticCudaDataPoolElement d_value(NULL);
			GF2nCudaMetrics metrics;

			GF2nArithmeticElement element = GF2nArithmeticElement(
				new GF2nArithmeticElementCuda(vec_value, d_value, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_async, metrics));

			return element;
		}

		GF2nArithmeticElement GF2nArithmeticCuda::getElement( const unsigned char *value, const uint32 bytes_value )
		{
			assert(bytes_value <= m_h_num_chunks * CUDA_BIGNUM_SIZE_BYTES);

			CUDA_BIGNUM* vec_value = new CUDA_BIGNUM[m_h_num_chunks];
			memset(vec_value, 0, m_h_num_chunks * CUDA_BIGNUM_SIZE_BYTES);
			
			std::copy_backward(value, value + bytes_value, ((unsigned char *)&vec_value[m_h_num_chunks]));
			//for(uint32 i=0; i<m_h_num_chunks; ++i) vec_value[i] = ntohl(vec_value[i]);

			GF2nArithmeticCudaDataPoolElement d_value(NULL);
			GF2nCudaMetrics metrics;

			GF2nArithmeticElement element = GF2nArithmeticElement(
				new GF2nArithmeticElementCuda(vec_value, d_value, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_async, metrics));

			return element;			
		}

		GF2nArithmeticElement GF2nArithmeticCuda::getElement( const void *value, const uint32 chunks_value )
		{
			assert(chunks_value <= m_h_num_chunks * CUDA_BIGNUM_SIZE_BYTES);

			CUDA_BIGNUM *vec_value = new CUDA_BIGNUM[m_h_num_chunks];
			memset(vec_value, 0, m_h_num_chunks * CUDA_BIGNUM_SIZE_BYTES);

			CUDA_BIGNUM *value_p = (CUDA_BIGNUM *)value;
			
			std::copy(value_p, value_p + chunks_value, &vec_value[0]);

			GF2nArithmeticCudaDataPoolElement d_value(NULL);
			GF2nCudaMetrics metrics;

			GF2nArithmeticElement element = GF2nArithmeticElement(
				new GF2nArithmeticElementCuda(vec_value, d_value, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_async, metrics));

			return element;			
		}		

		void GF2nArithmeticCuda::initDevice()
		{
			clearDevice();

			cuda::device_init();

			cuda::device_allocate(&m_d_irred_poly, m_h_num_bytes);
			
			cuda::device_set(m_d_irred_poly, &m_h_irred_poly[0], m_h_num_bytes);

			m_d_data_pool = new GF2nArithmeticCudaDataPool(m_h_num_bytes, DATA_POOL_SIZE);

			m_device_initialized = true;
		}

		void GF2nArithmeticCuda::clearDevice()
		{
			if( m_device_initialized )
			{
				cuda::device_delete(m_d_irred_poly);
				delete m_d_data_pool;
			}
		}

		void GF2nArithmeticCuda::initChunkSizes( uint32 field_size )
		{
			assert(field_size != 0);

			m_h_field_size = field_size;
			m_h_num_chunks = utils::calcNumberChunks<uint32>(m_h_field_size + 1, CUDA_BIGNUM_SIZE_BITS);
			m_h_num_bytes = m_h_num_chunks * CUDA_BIGNUM_SIZE_BYTES;
			m_h_indx_mask_bit = CUDA_BIGNUM_SIZE_BITS - (m_h_field_size % CUDA_BIGNUM_SIZE_BITS);
		}

		///////////////////////////////////////////////////////////////////////
		/*
			implementations of MethodNotFoundException
		*/
		MethodNotFoundException::MethodNotFoundException( std::string method ) 
		{
			m_msg = "Error: The called method " + method + " does not exist!";
		}

		const char* MethodNotFoundException::what() const throw()
		{
			return m_msg.c_str();
		}

		///////////////////////////////////////////////////////////////////////
		/*
			implementations of GF2nArithmeticElementCuda
		*/
		GF2nArithmeticElementCuda::GF2nArithmeticElementCuda( 
			CUDA_BIGNUM *h_value, 
			GF2nArithmeticCudaDataPoolElement d_value, 
			uint32 h_field_size, 
			uint32 h_num_chunks, 
			uint32 h_num_bytes,
			CUDA_BIGNUM *d_irred_poly, 
			uint32 h_indx_mask_bit, 
			GF2nArithmeticCudaDataPool *d_data_pool,
			const bool async,
			GF2nCudaMetrics h_metrics, 
			uint8 byteOrder )
			: m_h_value(h_value)
			, m_h_field_size(h_field_size)
			, m_h_num_chunks(h_num_chunks)
			, m_h_num_bytes(h_num_bytes)
			, m_h_async(async)
			, m_h_metrics(h_metrics)
			, m_d_value(d_value)
			, m_d_irred_poly(d_irred_poly)
			, m_h_indx_mask_bit(h_indx_mask_bit)
			, m_d_data_pool(d_data_pool)
			, m_byteOrder(byteOrder)
		{
			m_h_metrics.copyToDevice_time = 0.;
			m_h_metrics.copyToHost_time = 0.;

			// set device values
			if( m_h_value && !m_d_value)
			{
				m_d_value = m_d_data_pool->get();
				if( m_h_async )
				{
					cuda::device_set_async(*m_d_value, m_h_value, m_h_num_bytes);
				}
				else
				{
					m_h_metrics.copyToDevice_time = cuda::device_set(*m_d_value, m_h_value, m_h_num_bytes);	
				}

				if( m_byteOrder == LENDIAN )
				{
					cuda::device_swapBytes(*m_d_value, m_h_num_chunks);
				}
			}
		}

		GF2nArithmeticElementCuda::~GF2nArithmeticElementCuda()
		{
			// delete host values
			if( m_h_value )
				delete [] m_h_value;
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::add( GF2nArithmeticElementInterface *other )
		{	
			return parAdd(other);
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::sub( GF2nArithmeticElementInterface *other )
		{
			return parAdd(other);
		}		

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::mul( GF2nArithmeticElementInterface *other )
		{
			return parMul(other);		
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::div( GF2nArithmeticElementInterface *other )
		{
			return NULL;
		}		

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::runWithElement( const std::string &what, GF2nArithmeticElementInterface *other )
		{
			if( what.compare("add") == 0 )
			{
				return parAdd(other);
			}
			if( what.compare("parAdd") == 0 )
			{
				return parAdd(other);
			}
			if( what.compare("parAddLoop") == 0 )
			{
				return parAddLoop(other);
			}
			if( what.compare("parAddTime") == 0 )
			{
				return parAddTime(other);
			}
			if( what.compare("parAddWithEvents") == 0 )
			{
				return parAddWithEvents(other);
			}
			if( what.compare("parAddOwnStream") == 0 )
			{
				return parAddOwnStream(other);
			}	
			if( what.compare("parAddOwnStream1024Threads") == 0 )
			{
				return parAddOwnStream1024Threads(other);
			}	
			if( what.compare("parAddOwnStream512Threads") == 0 )
			{
				return parAddOwnStream512Threads(other);
			}	
			if( what.compare("parAddOwnStream256Threads") == 0 )
			{
				return parAddOwnStream256Threads(other);
			}	
			if( what.compare("parAddOwnStream128Threads") == 0 )
			{
				return parAddOwnStream128Threads(other);
			}	
			if( what.compare("parAdd2OwnStream") == 0 )
			{
				return parAdd2OwnStream(other);
			}			
			if( what.compare("parAdd4OwnStream") == 0 )
			{
				return parAdd4OwnStream(other);
			}			
			if( what.compare("parAdd8OwnStream") == 0 )
			{
				return parAdd8OwnStream(other);
			}			
			if( what.compare("parAddSharedMem") == 0 )
			{
				return parAddSharedMem(other);
			}
			else if( what.compare("parMul") == 0 )
			{
				return parMul(other);
			}
			else if( what.compare("parMulChunkedBarRed") == 0 )
			{
				return parMulChunkedBarRed(other);
			}
			else if( what.compare("measureKernelLaunchOverhead") == 0 )
			{
				return measureKernelLaunchOverhead(other);
			}
			else
			{
				throw MethodNotFoundException(what);
			}	
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::runWithValue( const std::string &what, uint32 value )
		{
			if( what.compare("parExponentiation") == 0 )
			{
				return parExponentiation(value);
			}	
			else if( what.compare("parInverseElement") == 0 )
			{
				return parInverseElement(value);
			}	
			else if( what.compare("parInverseElementWithExp") == 0 )
			{
				return parInverseElementWithExp(value);
			}	
			else
			{
				throw MethodNotFoundException(what);
			}	
		}		

		std::string GF2nArithmeticElementCuda::toString()
		{
			if( !m_h_value )
			{
				m_h_value = new CUDA_BIGNUM[m_h_num_bytes];
				memset(m_h_value, 0, m_h_num_bytes);
				m_h_metrics.copyToHost_time = cuda::device_get(m_h_value, *m_d_value, m_h_num_bytes);
			}

			std::vector<std::string> value_arr;

			for( uint32 i=0; i<m_h_num_chunks; ++i )
			{
				value_arr.push_back(std::to_string(m_h_value[i]));
			}

			std::string ret;
			utils::convertArrayToString(value_arr, sizeof(CUDA_BIGNUM) * 8, m_h_num_chunks, ret);
			
			return ret;
		}

		void GF2nArithmeticElementCuda::getValue( std::vector<uint8> &value )
		{
			GF2nArithmeticCudaDataPoolElement device_value = m_d_value;

			if( m_byteOrder == LENDIAN )
			{
				device_value = m_d_data_pool->get();
				cuda::device_copy(*device_value, *m_d_value, m_h_num_bytes);
				cuda::device_swapBytes(*device_value, m_h_num_chunks);
			}

			if( !m_h_value )
			{
				m_h_value = new CUDA_BIGNUM[m_h_num_bytes];
				memset(m_h_value, 0, m_h_num_bytes);
				m_h_metrics.copyToHost_time = cuda::device_get(m_h_value, *device_value, m_h_num_bytes);
			}

			uint32 num_uint8_chunks = utils::calcNumberChunks<uint32>(m_h_field_size, sizeof(uint8) * 8);

			value.resize(num_uint8_chunks, 0);

			memcpy(&value[0], reinterpret_cast<uint8 *>(m_h_value)+(m_h_num_bytes-num_uint8_chunks), num_uint8_chunks);
		}
		
		std::string GF2nArithmeticElementCuda::getMetrics()
		{
			std::stringstream ss;

			ss << "copyToDevice_time=" << m_h_metrics.copyToDevice_time << std::endl;
			ss << "copyToHost_time=" << m_h_metrics.copyToHost_time << std::endl;
			ss << "creation_time=" << m_h_metrics.creation_time << std::endl;
			ss << "load_time=" << m_h_metrics.load_time << std::endl;
			ss << "exec_time=" << m_h_metrics.exec_time << std::endl;
			ss << "store_time=" << m_h_metrics.store_time << std::endl;

			std::string metrics = ss.str();

			return metrics;
		}

		std::string GF2nArithmeticElementCuda::getMetrics( const std::string &metrics_name )
		{
			double value = 0.;

			if( metrics_name.compare("copyToDevice_time") == 0 )
			{
				value = m_h_metrics.copyToDevice_time;	
			} 
			else if( metrics_name.compare("copyToHost_time") == 0 )
			{
				value = m_h_metrics.copyToHost_time;	
			} 
			else if( metrics_name.compare("creation_time") == 0 )
			{
				value = m_h_metrics.creation_time;	
			} 
			else if( metrics_name.compare("load_time") == 0 )
			{
				value = m_h_metrics.load_time;	
			} 
			else if( metrics_name.compare("exec_time") == 0 )
			{
				value = m_h_metrics.exec_time;	
			} 
			else if( metrics_name.compare("store_time") == 0 )
			{
				value = m_h_metrics.store_time;	
			}

			std::string value_str = std::to_string(value);
			return value_str;
		}

		void GF2nArithmeticElementCuda::setProperty( const std::string &property_name, const std::string &property_value )
		{
			if( property_name.compare("num_threads") == 0 )
			{
				m_h_properties.num_threads = (CUDA_BIGNUM)atoi(property_value.c_str());
			}
			else if( property_name.compare("num_blocks") == 0 )
			{
				m_h_properties.num_blocks = (CUDA_BIGNUM)atoi(property_value.c_str());
			}
		}

		// GF2nArithmeticElementCuda functions
		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parAdd( GF2nArithmeticElementInterface *other )
		{	
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parAdd(*m_d_value, other_d_value, m_h_num_chunks, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parAddLoop( GF2nArithmeticElementInterface *other )
		{	
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parAddLoop(*m_d_value, other_d_value, m_h_num_chunks, m_h_properties.num_threads, m_h_properties.num_blocks, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parAddTime( GF2nArithmeticElementInterface *other )
		{
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			uint32 times[3];

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parAddTime(*m_d_value, other_d_value, m_h_num_chunks, *d_res, times);

			metrics.load_time = times[0];
			metrics.exec_time = times[1];
			metrics.store_time = times[2];

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parAddWithEvents( GF2nArithmeticElementInterface *other )
		{	
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parAddWithEvents(*m_d_value, other_d_value, m_h_num_chunks, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parAddOwnStream( GF2nArithmeticElementInterface *other )
		{	
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parAddOwnStream(*m_d_value, other_d_value, m_h_num_chunks, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;
		}		

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parAddOwnStream1024Threads( GF2nArithmeticElementInterface *other )
		{	
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parAddOwnStream1024Threads(*m_d_value, other_d_value, m_h_num_chunks, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;
		}		

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parAddOwnStream512Threads( GF2nArithmeticElementInterface *other )
		{	
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parAddOwnStream512Threads(*m_d_value, other_d_value, m_h_num_chunks, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;
		}		

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parAddOwnStream256Threads( GF2nArithmeticElementInterface *other )
		{	
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parAddOwnStream256Threads(*m_d_value, other_d_value, m_h_num_chunks, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;
		}		

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parAddOwnStream128Threads( GF2nArithmeticElementInterface *other )
		{	
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parAddOwnStream128Threads(*m_d_value, other_d_value, m_h_num_chunks, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;
		}		

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parAdd2OwnStream( GF2nArithmeticElementInterface *other )
		{
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parAdd2OwnStream(*m_d_value, other_d_value, m_h_num_chunks, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;			
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parAdd4OwnStream( GF2nArithmeticElementInterface *other )
		{
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parAdd4OwnStream(*m_d_value, other_d_value, m_h_num_chunks, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parAdd8OwnStream( GF2nArithmeticElementInterface *other )
		{
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parAdd8OwnStream(*m_d_value, other_d_value, m_h_num_chunks, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parAddSharedMem( GF2nArithmeticElementInterface *other )
		{	
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parAddSharedMem(*m_d_value, other_d_value, m_h_num_chunks, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;
		}


		/**************************************************************************\

                               Multiplication

		\**************************************************************************/

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parMul( GF2nArithmeticElementInterface *other )
		{	
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parMul(*m_d_value, other_d_value, m_h_num_chunks, m_d_irred_poly, m_h_indx_mask_bit, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics, m_byteOrder);

			return new_element;
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parMulChunkedBarRed( GF2nArithmeticElementInterface *other )
		{
			CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parMulChunkedBarRed(*m_d_value, other_d_value, m_h_num_chunks, m_d_irred_poly, m_h_field_size, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics, m_byteOrder);

			return new_element;
		}

		/**************************************************************************\

                               Exponentiation

		\**************************************************************************/

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parExponentiation( uint32 value )
		{
			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parExponentiation(*m_d_value, value, m_h_num_chunks, m_d_irred_poly, m_h_field_size, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics, m_byteOrder);

			return new_element;
		}				

		/**************************************************************************\

                               Inverse

		\**************************************************************************/

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parInverseElement( uint32 value )
		{
			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parInverseElement(*m_d_value, m_h_num_chunks, m_d_irred_poly, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics, m_byteOrder);

			return new_element;
		}	

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::parInverseElementWithExp( uint32 value )
		{
			GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::parInverseElementWithExp(*m_d_value, m_h_num_chunks, m_d_irred_poly, m_h_field_size, *d_res);

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics, m_byteOrder);

			return new_element;
		}

		/**************************************************************************\

                               Helper

		\**************************************************************************/

		// GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::createChunkProdArray( GF2nArithmeticElementInterface *other )
		// {
		// 	CUDA_BIGNUM *other_d_value = reinterpret_cast<GF2nArithmeticElementCuda *>(other)->getDeviceValue();

		// 	GF2nArithmeticCudaDataPoolElement d_res = m_d_data_pool->get();

		// 	uint32 factor = * 4 * m_h_num_chunks * m_h_num_chunks;
		// 	/*
		// 	 * allocate the array
		// 	 * 0 	... 	2k
		// 	 * ...
		// 	 * 2k
		// 	 */
		// 	CUDA_BIGNUM *d_res;
		// 	cuda::device_allocate(&d_res, CUDA_BIGNUM_SIZE_BYTES * factor);

		// 	GF2nCudaMetrics metrics;
		// 	metrics.creation_time = cuda::createChunkProdArray(*m_d_value, other_d_value, m_h_num_chunks, d_res);

		// 	GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, d_res, m_h_field_size, m_h_num_chunks * factor, m_h_num_bytes * factor, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics, m_byteOrder);

		// 	return new_element;
		// }

		GF2nArithmeticElementInterface *GF2nArithmeticElementCuda::measureKernelLaunchOverhead( GF2nArithmeticElementInterface *other )
		{	
			GF2nCudaMetrics metrics;
			metrics.creation_time = cuda::measureKernelLaunchOverhead();

			GF2nArithmeticElementCuda *new_element = new GF2nArithmeticElementCuda(NULL, m_d_value, m_h_field_size, m_h_num_chunks, m_h_num_bytes, m_d_irred_poly, m_h_indx_mask_bit, m_d_data_pool, m_h_async, metrics);

			return new_element;
		}

		CUDA_BIGNUM *GF2nArithmeticElementCuda::getValue()
		{
			if( !m_h_value )
			{
				m_h_value = new CUDA_BIGNUM[m_h_num_bytes];
				memset(m_h_value, 0, m_h_num_bytes);
				m_h_metrics.copyToHost_time = cuda::device_get(m_h_value, *m_d_value, m_h_num_bytes);
			}

			return m_h_value;
		}

		CUDA_BIGNUM *GF2nArithmeticElementCuda::getDeviceValue()
		{
			return *m_d_value;
		}
	}
}
