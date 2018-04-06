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

#ifndef __GF2N_ARITHMETIC_CUDA_H__
#define __GF2N_ARITHMETIC_CUDA_H__

#include <vector>
#include <map>
#include <list>
#include <exception>

#include "GF2nArithmetic.h"
#include "CudaBignum.h"
#include "GF2nArithmeticCudaDataPool.h"

void set_irrep(std::vector<int>&, unsigned int);

#define LENDIAN 0
#define BENDIAN 1
#define DATA_POOL_SIZE 20

namespace libcumffa {
	namespace gpu {

		struct GF2nCudaMetrics 
		{
			uint32 copyToDevice_time;
			uint32 copyToHost_time;
			uint32 creation_time;
			uint32 load_time;
			uint32 exec_time;
			uint32 store_time;

			GF2nCudaMetrics() 
				: copyToDevice_time(0.)
				, copyToHost_time(0.)
				, creation_time(0.)
				, load_time(0.)
				, exec_time(0.)
				, store_time(0.)
				{}
		};

		struct GF2nCudaProperties 
		{
			uint32 num_threads;
			uint32 num_blocks;

			GF2nCudaProperties() 
				: num_threads(0)
				, num_blocks(0)
				{}
		};		

		/*

			Cuda Arithmetic

		*/
		class WrongIrredPolyException : public std::exception
		{
			virtual const char* what() const throw()
			{
				return "The passed irreducible polynomial does not match to the field!!!";
			}
		};

		/**
		 * 
		 */
		class GF2nArithmeticCuda : public GF2nArithmeticInterface
		{
		public:
			/**
			 * @brief [brief description]
			 * @details [long description]
			 */
			GF2nArithmeticCuda();
			~GF2nArithmeticCuda();
			void setFieldSize( const uint32 field_size );
			void setDummyParameters( const uint32 field_size, const std::string irred_poly );
			void setDummyParameters( const uint32 field_size, const unsigned char *irred_poly, const uint32 chunks_irred_poly );
			void setDummyParameters( const uint32 field_size, const void *irred_poly, const uint32 chunks_irred_poly );
			void setFlags( const unsigned char flags );

			/** @brief Prints character ch at the current location
			 *         of the cursor.
			 *
			 *  If the character is a newline ('\n'), the cursor should
			 *  be moved to the next line (scrolling if necessary).  If
			 *  the character is a carriage return ('\r'), the cursor
			 *  should be immediately reset to the beginning of the current
			 *  line, causing any future output to overwrite any existing
			 *  output on the line.  If backsapce ('\b') is encountered,
			 *  the previous character should be erased (write a space
			 *  over it and move the cursor back one column).  It is up
			 *  to you how you want to handle a backspace occurring at the
			 *  beginning of a line.
			 *
			 *  @param ch the character to print
			 *  @return The input character
			 */
			GF2nArithmeticElement getElement( const std::string value );
			GF2nArithmeticElement getElement( const unsigned char *value, const uint32 bytes_value );
			GF2nArithmeticElement getElement( const void *value, const uint32 chunks_value );

		private:
			void initChunkSizes( uint32 field_size );
			void initDevice();
			void clearDevice();

		private:
			bool m_async;
			uint32 m_h_field_size;
			uint32 m_h_num_chunks;
			uint32 m_h_num_bytes;
			uint32 m_h_indx_mask_bit;
			std::vector<CUDA_BIGNUM> m_h_irred_poly;

		private:
			bool m_device_initialized;
			CUDA_BIGNUM *m_d_irred_poly;
			GF2nArithmeticCudaDataPool *m_d_data_pool;
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

		class GF2nArithmeticElementCuda : public GF2nArithmeticElementInterface
		{
		public:
			GF2nArithmeticElementCuda( 
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
				uint8 byteOrder = LENDIAN );
			~GF2nArithmeticElementCuda();

		public: /* GF2nArithmeticElementInterface functions */
			GF2nArithmeticElementInterface *add( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *sub( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *mul( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *div( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *runWithElement( const std::string &what, GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *runWithValue( const std::string &what, uint32 value );
			std::string toString();
			void getValue( std::vector<uint8> &value );
			double getCreationTime();
			double getCopyToDeviceTime();
			std::string getMetrics();
			std::string getMetrics( const std::string &metrics_name );
			/**
			 * @brief      { function_description }
			 *
			 * @param[in]  property_name   { parameter_description }
			 * @param[in]  property_value  { parameter_description }
			 */
			void setProperty( const std::string &property_name, const std::string &property_value );

		public: /* GF2nArithmeticElementCuda */
			// Addition
			GF2nArithmeticElementInterface *parAdd( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parAddLoop( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parAddTime( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parAddWithEvents( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parAddOwnStream( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parAddOwnStream1024Threads( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parAddOwnStream512Threads( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parAddOwnStream256Threads( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parAddOwnStream128Threads( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parAdd2OwnStream( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parAdd4OwnStream( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parAdd8OwnStream( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parAddSharedMem( GF2nArithmeticElementInterface *other );

			// Multiplication
			GF2nArithmeticElementInterface *parMul( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parMulChunkedBarRed( GF2nArithmeticElementInterface *other );
			GF2nArithmeticElementInterface *parMulChunkedMontgRed( GF2nArithmeticElementInterface *other );

			// Exponentiation
			GF2nArithmeticElementInterface *parExponentiation( uint32 value );

			// Inverse
			GF2nArithmeticElementInterface *parInverseElement( uint32 value );
			GF2nArithmeticElementInterface *parInverseElementWithExp( uint32 value );

			// Helper (Debug)
			//GF2nArithmeticElementInterface *createChunkProdArray( GF2nArithmeticElementInterface *other );	

			// Other Stuff
			GF2nArithmeticElementInterface *measureKernelLaunchOverhead( GF2nArithmeticElementInterface *other );

		public:
			CUDA_BIGNUM *getValue();

		private: // host values
			CUDA_BIGNUM *m_h_value;
			uint32 m_h_field_size; 
			uint32 m_h_num_chunks; 					// the number chunks needed for the given field size
			uint32 m_h_num_bytes;	
			uint32 m_h_indx_mask_bit;
			bool m_h_async;
			double m_h_creation_time;
			double m_h_copyToDevice_time;
			GF2nCudaMetrics m_h_metrics;
			GF2nCudaProperties m_h_properties;
			uint8 m_byteOrder;

		private: // device values
			GF2nArithmeticCudaDataPoolElement m_d_value;
			CUDA_BIGNUM *m_d_irred_poly;
			GF2nArithmeticCudaDataPool *m_d_data_pool;

		public:
			CUDA_BIGNUM *getDeviceValue();
		};
		
	}
}

#endif //__GF2N_ARITHMETIC_CUDA_H__
