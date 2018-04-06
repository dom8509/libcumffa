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

#include <openssl/crypto.h>
#include <cstring>
#include <cstdlib>
#include <sys/time.h>
#include <cassert>

#include "../include/GF2nArithmeticOpenSSL.h"

void set_irrep(std::vector<int>&, unsigned int);

//openssl
namespace libcumffa {
	namespace cpu {

		namespace openssl {

			// return the time in milliseconds
			double cpuSecond()
			{
			    struct timeval tp;
			    gettimeofday(&tp, NULL);
			    return ((double)tp.tv_sec * 1000 + (double)tp.tv_usec * 1.e-3);
			}

			double add( BIGNUM *x, BIGNUM *y, BIGNUM *res )
			{
				double iStart, iElaps;
				iStart = cpuSecond();

				BN_GF2m_add(res, x, y);

				iElaps = cpuSecond() - iStart;

				return iElaps;
			}

			double mul( BIGNUM *x, BIGNUM *y, int *irred_poly, BIGNUM *res )
			{
				BN_CTX *ctx = BN_CTX_new();

				double iStart, iElaps;
				iStart = cpuSecond();

				BN_GF2m_mod_mul_arr(res, x, y, reinterpret_cast<const int*>(irred_poly), ctx);

				iElaps = cpuSecond() - iStart;

				BN_CTX_free(ctx);

				return iElaps;
			}

			double inverse( BIGNUM *x, int *irred_poly, BIGNUM *res )
			{
				BN_CTX *ctx = BN_CTX_new();

				double iStart, iElaps;
				iStart = cpuSecond();

				BN_GF2m_mod_inv_arr(res, x, reinterpret_cast<const int*>(irred_poly), ctx);
				
				iElaps = cpuSecond() - iStart;

				BN_CTX_free(ctx);

				return iElaps;
			}

			// res = x^k % irred_poly
			double exp( BIGNUM *x, BIGNUM *k, int *irred_poly, BIGNUM *res )
			{
				BN_CTX *ctx = BN_CTX_new();

				double iStart, iElaps;
				iStart = cpuSecond();

				BN_GF2m_mod_exp_arr(res, x, k, reinterpret_cast<const int*>(irred_poly), ctx);

				iElaps = cpuSecond() - iStart;

				BN_CTX_free(ctx);

				return iElaps;
			}
		}

		GF2nArithmeticOpenSSL::GF2nArithmeticOpenSSL()
		{
		}

		GF2nArithmeticOpenSSL::~GF2nArithmeticOpenSSL()
		{
		}

		void GF2nArithmeticOpenSSL::setFieldSize( const uint32 field_size )
		{
			m_field_size = field_size;
			set_irrep(m_irred_poly, static_cast<unsigned int>(m_field_size));

			// The irred poly has to be terminated with -1 
			// because inside of the function BN_GF2m_mod_inv_arr, 
			// the irred poly is converted to a BIGNUM 
			// using the function BN_GF2m_arr2poly. This function
			// iterates over all chunks of the irred poly
			// until it hits a -1.
			m_irred_poly.push_back(-1);
		}

		void GF2nArithmeticOpenSSL::setDummyParameters( const uint32 field_size, const std::string irred_poly )
		{
			m_field_size = field_size;		

			uint32 num_chunks = utils::calcNumberChunks(field_size + 1, static_cast<uint32>(sizeof(int) * 8));
			std::vector<std::string> str_arr_irred_poly;
			utils::convertStringToArray(irred_poly, sizeof(int) * 8, static_cast<int>(num_chunks), str_arr_irred_poly);

			if( !m_irred_poly.empty() )
			{
				m_irred_poly.clear();	
			}

			for( uint32 i=0; i<num_chunks; ++i )
			{
				unsigned int current_chunk = (unsigned int)atoi32(str_arr_irred_poly[i].c_str());
				unsigned int x = current_chunk;

			 	x = x - ((x >> 1) & 0x55555555);
			    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
			    x = (x + (x >> 4)) & 0x0F0F0F0F;
			    x = x + (x >> 8);
			    x = x + (x >> 16);
			    
			    unsigned int num_bits_set = x & 0x0000003F;

				if( num_bits_set > 0 )
				{
					for( unsigned int j=0; j<num_bits_set; ++j )
					{
						unsigned int bit=2147483648, pos=1;
						for( ; bit>=1; bit=bit>>1, pos++ )
						{
							if( current_chunk & bit )
							{
								break;
							}
						}
							// calc the absolute bit position in the irred_poly string
						unsigned int abs_pos = ((sizeof(int) * 8) * (num_chunks - 1 - i)) + ((sizeof(int) * 8) - pos);
						m_irred_poly.push_back(abs_pos);

						// remove bit from x
						current_chunk=current_chunk^bit;
					}
				}
			}	

			m_irred_poly.push_back(-1);
		}

		void GF2nArithmeticOpenSSL::setDummyParameters( const uint32 field_size, const unsigned char *irred_poly, const uint32 chunks_irred_poly )
		{
			m_field_size = field_size;

			uint32 num_chunks = utils::calcNumberChunks(m_field_size + 1, static_cast<uint32>(sizeof(int) * 8));

			if( !m_irred_poly.empty() )
			{
				m_irred_poly.clear();		
			}

			m_irred_poly.resize(num_chunks);

			assert(chunks_irred_poly <= num_chunks * sizeof(int));

			std::copy_backward(irred_poly, irred_poly + chunks_irred_poly, ((unsigned char *)&m_irred_poly[0]) + num_chunks * sizeof(int) / sizeof(unsigned char));

			m_irred_poly.push_back(-1);
		}

		void GF2nArithmeticOpenSSL::setDummyParameters( const uint32 field_size, const void *irred_poly, const uint32 chunks_irred_poly )
		{
			m_field_size = field_size;

			uint32 num_chunks = utils::calcNumberChunks(m_field_size + 1, static_cast<uint32>(sizeof(int) * 8));
			assert(chunks_irred_poly <= num_chunks * sizeof(int));

			if( !m_irred_poly.empty() )
			{
				m_irred_poly.clear();		
			}

			m_irred_poly.resize(chunks_irred_poly);

			int *irred_poly_p = (int *)irred_poly;

			std::copy(irred_poly_p, irred_poly_p + chunks_irred_poly, &m_irred_poly[0]);

			m_irred_poly.push_back(-1);
		}		

		GF2nArithmeticElement GF2nArithmeticOpenSSL::getElement( const std::string value )
		{
			BIGNUM *bn_value = NULL;
			BN_dec2bn(&bn_value, value.c_str());

			GF2nOpenSSLMetrics metrics;

			GF2nArithmeticElement element = GF2nArithmeticElement(
				new GF2nArithmeticElementOpenSSL(bn_value, m_field_size, m_irred_poly, metrics));

			return element;
		}

		GF2nArithmeticElement GF2nArithmeticOpenSSL::getElement( const unsigned char *value, const uint32 chunks_value )
		{
			BIGNUM *bn_value = BN_bin2bn(value, static_cast<int>(chunks_value), NULL);

			GF2nOpenSSLMetrics metrics;

			GF2nArithmeticElement element = GF2nArithmeticElement(
				new GF2nArithmeticElementOpenSSL(bn_value, m_field_size, m_irred_poly, metrics));

			return element;
		}

		GF2nArithmeticElement GF2nArithmeticOpenSSL::getElement( const void *value, const uint32 chunks_value )
		{
			BIGNUM *bn_value = BN_bin2bn((unsigned char *)value, static_cast<int>(chunks_value), NULL);

			GF2nOpenSSLMetrics metrics;

			GF2nArithmeticElement element = GF2nArithmeticElement(
				new GF2nArithmeticElementOpenSSL(bn_value, m_field_size, m_irred_poly, metrics));

			return element;
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
			implementations of GF2nArithmeticElementOpenSSL
		*/
		GF2nArithmeticElementOpenSSL::GF2nArithmeticElementOpenSSL( BIGNUM *value, const uint32 field_size, std::vector<int> irred_poly, GF2nOpenSSLMetrics metrics )
		: m_value(value)
		, m_field_size(field_size)
		, m_irred_poly(irred_poly)
		, m_metrics(metrics) {}

		GF2nArithmeticElementOpenSSL::~GF2nArithmeticElementOpenSSL()
		{
			if( m_value )
				BN_free(m_value);
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementOpenSSL::add( GF2nArithmeticElementInterface *other )
		{
			BIGNUM *res = BN_new();
			BIGNUM *other_value = reinterpret_cast<GF2nArithmeticElementOpenSSL *>(other)->getValue();

			m_metrics.creation_time = openssl::add(m_value, other_value, res);

			GF2nArithmeticElementInterface *new_element = new GF2nArithmeticElementOpenSSL(res, m_field_size, m_irred_poly, m_metrics);

			BN_free(other_value);

			return new_element;
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementOpenSSL::sub( GF2nArithmeticElementInterface *other )
		{
			return add(other);
		}		

		GF2nArithmeticElementInterface *GF2nArithmeticElementOpenSSL::mul( GF2nArithmeticElementInterface *other )
		{
			BIGNUM *res = BN_new();
			BIGNUM *other_value = reinterpret_cast<GF2nArithmeticElementOpenSSL *>(other)->getValue();

			m_metrics.creation_time = openssl::mul(m_value, other_value, &m_irred_poly[0], res);

			GF2nArithmeticElementInterface *new_element = new GF2nArithmeticElementOpenSSL(res, m_field_size, m_irred_poly, m_metrics);

			BN_free(other_value);

			return new_element;
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementOpenSSL::div( GF2nArithmeticElementInterface *other )
		{
			return NULL;
		}	

		GF2nArithmeticElementInterface *GF2nArithmeticElementOpenSSL::exp( uint32 value )
		{
			BIGNUM *res = BN_new();
			BIGNUM *k = BN_new();

			std::ostringstream value_str;
			value_str << value;
			BN_dec2bn(&k, value_str.str().c_str());

			m_metrics.creation_time = openssl::exp(m_value, k, &m_irred_poly[0], res);

			GF2nArithmeticElementInterface *new_element = new GF2nArithmeticElementOpenSSL(res, m_field_size, m_irred_poly, m_metrics);

			BN_free(k);

			return new_element;
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementOpenSSL::inverse( uint32 value )
		{
			BIGNUM *res = BN_new();

			m_metrics.creation_time = openssl::inverse(m_value, &m_irred_poly[0], res);

			GF2nArithmeticElementInterface *new_element = new GF2nArithmeticElementOpenSSL(res, m_field_size, m_irred_poly, m_metrics);

			return new_element;
		}		

		GF2nArithmeticElementInterface *GF2nArithmeticElementOpenSSL::runWithElement( const std::string &what, GF2nArithmeticElementInterface *other )
		{
			if( what.compare("add") == 0 )
			{
				return add(other);
			}
			else if( what.compare("sub") == 0 )
			{
				return sub(other);
			}
			else if( what.compare("mul") == 0 )
			{
				return mul(other);
			}
			else
			{
				throw MethodNotFoundException(what);
			}	
		}

		GF2nArithmeticElementInterface *GF2nArithmeticElementOpenSSL::runWithValue( const std::string &what, uint32 value )
		{
			if( what.compare("exp") == 0 )
			{
				return exp(value);
			}
			else if( what.compare("inverse") == 0 )
			{
				return inverse(value);
			}
			else
			{
				throw MethodNotFoundException(what);
			}
		}		

		std::string GF2nArithmeticElementOpenSSL::toString()
		{
			std::string ret(BN_bn2dec(m_value));
			return ret;
		}

		void GF2nArithmeticElementOpenSSL::getValue( std::vector<uint8> &value )
		{
			uint32 num_uint8_chunks = utils::calcNumberChunks<uint32>(m_field_size, sizeof(uint8) * 8);

			uint32 bn_num_bytes = static_cast<uint32>(BN_num_bytes(m_value));
			assert(bn_num_bytes <= num_uint8_chunks);

			value.resize(num_uint8_chunks, 0);
			BN_bn2bin(m_value, &value[num_uint8_chunks - bn_num_bytes]);
		}

		std::string GF2nArithmeticElementOpenSSL::getMetrics()
		{
			std::stringstream ss;

			ss << "creation_time=" << m_metrics.creation_time << std::endl;

			std::string metrics = ss.str();

			return metrics;
		}

		std::string GF2nArithmeticElementOpenSSL::getMetrics( const std::string &metrics_name )
		{
			double value = 0.;

			if( metrics_name.compare("creation_time") == 0 )
			{
				value = m_metrics.creation_time;	
			}

			std::string value_str = std::to_string(value);
			return value_str;
		}

		void GF2nArithmeticElementOpenSSL::setProperty( const std::string &property_name, const std::string &property_value )
		{}

		BIGNUM *GF2nArithmeticElementOpenSSL::getValue()
		{
			BIGNUM *res = BN_dup(m_value);
			return res;
		}
	}
}