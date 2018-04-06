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

#include <GF2nArithmetic_C.h>

#include <gmpxx.h>
#include <cstring>
#include <chrono>
#include <iterator>
#include <sys/time.h>
#include <iomanip>

#include <iostream>

using namespace libcumffa;

GF2nArithmeticElement bn_a;
GF2nArithmeticElement bn_b;
GF2nArithmeticElement res;

extern "C" 
{
	void *createInstance( const char *mode )
	{
		GF2nArithmetic *inst = GF2nArithmeticFactory::createInstancePtr(mode);
		return reinterpret_cast<void *>(inst);
	}

	void setFieldSize( void *inst, const unsigned long field_size )
	{
		reinterpret_cast<GF2nArithmetic *>(inst)->setFieldSize(field_size);
	}

	void setDummyParameters( void *inst, const unsigned long field_size, const unsigned char *irred_poly, const unsigned long chunks_irred_poly )
	{
		if( irred_poly )
		{
			if( chunks_irred_poly == 0)
			{
				reinterpret_cast<GF2nArithmetic *>(inst)->setDummyParameters(field_size, (const char*)irred_poly);
			}
			else
			{
				reinterpret_cast<GF2nArithmetic *>(inst)->setDummyParameters(field_size, irred_poly, chunks_irred_poly);	
			}
		} 
		else
		{
			uint32 num_chunks = 0;
			uint32 size_irred_poly = field_size + 1;

			std::string mode = reinterpret_cast<GF2nArithmetic *>(inst)->getMode();
			
			if( mode.compare("Cuda") == 0 )
			{
				std::vector<ufixn> rand_vec;
				num_chunks = create_randomness(size_irred_poly, 23, rand_vec);
				reinterpret_cast<GF2nArithmetic *>(inst)->setDummyParameters(field_size, &rand_vec[0], num_chunks);	
			}
			else
			{
				std::vector<int> rand_vec;
				num_chunks = create_randomness(size_irred_poly, 23, rand_vec);
				reinterpret_cast<GF2nArithmetic *>(inst)->setDummyParameters(field_size, &rand_vec[0], num_chunks);	
			}
		}
	}

	void run( 
		void *inst, 
		const unsigned char *what, 
		const unsigned long value,
		const unsigned long field_size, 
		unsigned char flags, 
		int runs, 
		double *results )
	{
		std::string what_str((const char*)what);
		 
		if( (flags & 0x2) > 0 )
		{
			reinterpret_cast<GF2nArithmetic *>(inst)->setFlags((char)0x2);
		}

		// if flag 0x1 is not set -> create new variables
		uint32 num_chunks;
		if( (flags & 0x1) == 0 )
		{
			std::vector<uint8> rand_a;
			std::vector<uint8> rand_b;

			create_randomness(field_size, 42, rand_a);
			num_chunks = create_randomness(field_size, 84, rand_b);

			bn_a = reinterpret_cast<GF2nArithmetic *>(inst)->getElement(&rand_a[0], num_chunks);
			bn_b = reinterpret_cast<GF2nArithmetic *>(inst)->getElement(&rand_b[0], num_chunks);
		}
		
		double res_vec[runs];
		
		if( (flags & 0x4) > 0 )
		{
			for( int i=0; i<runs; ++i )
			{
				res = bn_a.runWithValue(what_str, value);
				res_vec[i] = std::stod(res.getMetrics("creation_time"));
			}
		}
		else
		{
			for( int i=0; i<runs; ++i )
			{
				res = bn_a.runWithElement(what_str, bn_b);
				res_vec[i] = std::stod(res.getMetrics("creation_time"));
			}
		}

		memcpy(results, &res_vec[0], sizeof(res_vec));
	}

	void getResult( unsigned long num_chunks, char *c )
	{
		std::vector<uint8> res_vec;
		res.getValue(res_vec);
		memcpy(c, &res_vec[0], num_chunks);		
	}

	int getMetricsSize( const unsigned char *value_name )
	{
		std::string value_name_str((const char*)value_name);
		int metrics_size = 0;

		if( value_name_str.compare("res") == 0)
		{
			metrics_size = res.getMetrics().length();	
		}
		if( value_name_str.compare("bn_a") == 0)
		{
			metrics_size = bn_a.getMetrics().length();	
		}
		if( value_name_str.compare("bn_b") == 0)
		{
			metrics_size = bn_b.getMetrics().length();	
		}

		return metrics_size;
	}

	void getMetrics( const unsigned char *value_name, char *metrics )
	{
		std::string value_name_str((const char*)value_name);
		
		if( value_name_str.compare("res") == 0)
		{
			memcpy(metrics, res.getMetrics().data(), res.getMetrics().length());
		}
		if( value_name_str.compare("bn_a") == 0)
		{
			memcpy(metrics, bn_a.getMetrics().data(), bn_a.getMetrics().length());
		}
		if( value_name_str.compare("bn_b") == 0)
		{
			memcpy(metrics, bn_b.getMetrics().data(), bn_b.getMetrics().length());
		}
	}

	void setProperty( const unsigned char *value_name, char *property_name, char *property_value )
	{
		std::string value_name_str((const char*)value_name);
		
		if( value_name_str.compare("res") == 0)
		{
			res.setProperty(property_name, property_value);
		}
		if( value_name_str.compare("bn_a") == 0)
		{
			bn_a.setProperty(property_name, property_value);
		}
		if( value_name_str.compare("bn_b") == 0)
		{
			bn_b.setProperty(property_name, property_value);
		}
	}

	void getRandomNumber( unsigned long num_bits, int seed, unsigned char *rn )
	{
		std::vector<unsigned char> rand_vec;
		
		uint32 num_chunks = create_randomness(num_bits, seed, rand_vec);

		memcpy(rn, rand_vec.data(), num_chunks);
	}

	void destroyInstance( void *inst )
	{
		if( inst )
		{
			delete reinterpret_cast<GF2nArithmetic *>(inst);
		}
	}

	double convertStringToArray( char *a, int size_chunk_bits, int num_chunks, char *arr)
	{
		std::vector<std::string> res;
		std::string stra(a);

		auto t_start = std::chrono::high_resolution_clock::now();

		libcumffa::utils::convertStringToArray(stra, size_chunk_bits, num_chunks, res);

		auto t_end = std::chrono::high_resolution_clock::now();

		for( int i=0; i<num_chunks; ++i )
		{
			mpz_class current_chunk(res[i], 10);
			std::string current_chunk_bit_str = current_chunk.get_str(2);

			if( current_chunk_bit_str.length() < (unsigned int)size_chunk_bits )
			{
  				current_chunk_bit_str.insert(0, size_chunk_bits - current_chunk_bit_str.length(), '0');
			}
			memcpy(arr+(i*size_chunk_bits), current_chunk_bit_str.c_str(), size_chunk_bits);
		}

		return std::chrono::duration<double, std::milli>(t_end - t_start).count();
	}

	double convertStringToArray2( char *a, int size_chunk_bits, int num_chunks, char *arr)
	{
		std::vector<std::string> res;
		std::string stra(a);

		auto t_start = std::chrono::high_resolution_clock::now();

		libcumffa::utils::convertStringToArray2(stra, size_chunk_bits, num_chunks, res);

		auto t_end = std::chrono::high_resolution_clock::now();

		for( int i=0; i<num_chunks; ++i )
		{
			mpz_class current_chunk(res[i], 10);
			std::string current_chunk_bit_str = current_chunk.get_str(2);

			if( current_chunk_bit_str.length() < (unsigned int)size_chunk_bits )
			{
  				current_chunk_bit_str.insert(0, size_chunk_bits - current_chunk_bit_str.length(), '0');
			}
			memcpy(arr+(i*size_chunk_bits), current_chunk_bit_str.c_str(), size_chunk_bits);
		}

		return std::chrono::duration<double, std::milli>(t_end - t_start).count();
	}	

	double convertArrayToString( char **arr, int size_chunk_bits, int num_chunks, char *a )
	{
		std::string res;
		std::vector<std::string> arr_vec;

		for( int i=0; i<num_chunks; ++i )
		{
			arr_vec.push_back(arr[i]);
		}

		auto t_start = std::chrono::high_resolution_clock::now();

		libcumffa::utils::convertArrayToString(arr_vec, size_chunk_bits, num_chunks, res);

		auto t_end = std::chrono::high_resolution_clock::now();

		strncpy(a, res.c_str(), res.size());

		return std::chrono::duration<double, std::milli>(t_end - t_start).count();
	}
}
