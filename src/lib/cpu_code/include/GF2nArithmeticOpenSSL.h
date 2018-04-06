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

#ifndef __GF2N__ARITHMETIC_OPENSSL_H__
#define __GF2N__ARITHMETIC_OPENSSL_H__

#include <openssl/bn.h>

#include "GF2nArithmetic.h"

namespace libcumffa {
	namespace cpu {

		struct GF2nOpenSSLMetrics 
		{
			double creation_time;
		
			GF2nOpenSSLMetrics() 
				: creation_time(0.)
				{}
		};

		class GF2nArithmeticOpenSSL : public GF2nArithmeticInterface
		{
		public:
			GF2nArithmeticOpenSSL();
			~GF2nArithmeticOpenSSL();
			void setFieldSize( const uint32 field_size );
			void setDummyParameters( const uint32 field_size, const std::string irred_poly );
			void setDummyParameters( const uint32 field_size, const unsigned char *irred_poly, const uint32 chunks_irred_poly );
			void setDummyParameters( const uint32 field_size, const void *irred_poly, const uint32 chunks_irred_poly );
			void setFlags( const unsigned char flags ) {}
			GF2nArithmeticElement getElement( const std::string value );
			GF2nArithmeticElement getElement( const unsigned char *value, const uint32 chunks_value );
			GF2nArithmeticElement getElement( const void *value, const uint32 chunks_value );
		private:
			uint32 m_field_size;
			std::vector<int> m_irred_poly;
		};

		class MethodNotFoundException : public std::exception
		{
		public:
			MethodNotFoundException( std::string method );

		private:
			virtual const char* what() const throw();

		private:
			std::string m_msg;
		};

		class GF2nArithmeticElementOpenSSL : public GF2nArithmeticElementInterface
		{
		public:
			GF2nArithmeticElementOpenSSL( BIGNUM *value, const uint32 field_size, std::vector<int> irred_poly, GF2nOpenSSLMetrics metrics );
			~GF2nArithmeticElementOpenSSL();

		public:
			GF2nArithmeticElementInterface *add( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *sub( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *mul( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *div( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *exp( uint32 value );
			GF2nArithmeticElementInterface *inverse( uint32 value );
			GF2nArithmeticElementInterface *runWithElement( const std::string &what, GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *runWithValue( const std::string &what, uint32 value );
			std::string toString();
			void getValue( std::vector<uint8_t> &value );
			std::string getMetrics();
			std::string getMetrics( const std::string &metrics_name );
			void setProperty( const std::string &property_name, const std::string &property_value );

		public:
			BIGNUM *getValue();

		private:
			BIGNUM *m_value;
			uint32 m_field_size;
			std::vector<int> m_irred_poly;
			GF2nOpenSSLMetrics m_metrics;
		};

	}
}

#endif //__GF2N__ARITHMETIC_OPENSSL_H__