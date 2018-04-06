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

#ifndef __GF2N_CPU_ARITHMETIC_C_H__
#define __GF2N_CPU_ARITHMETIC_C_H__

#include <GF2nArithmetic.h>

#include <iostream>

#include <random>
#include <vector>
#include <cstdint>
#include <functional>

void hexDump( char *desc, void *addr, int len ) 
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

template<typename R> 
R bitmask2( const uint32 num_ones )
{
	R res = 0;

	for( uint32 i = 0; i < num_ones; ++i )
	{
		res |= ((R)1 << i);
	}
	
	return res;
}

template<class T>
uint32 create_randomness( uint32 num_bits, int seed, std::vector<T> &rand ) 
{
	uint32 num_chunks = 0;

	if( num_bits > 0 )
	{
		// calculate the number of chunks needed and resize the 
		// return vector to the number of T chunks
		num_chunks = ((num_bits - 1) / (sizeof(T) * 8)) + 1;
		rand.resize(num_chunks, 0);

		// create a distribution from 0 to max_value_of(T)
		std::uniform_int_distribution<T> distribution(0, std::numeric_limits<T>::max());
		// create an engine to select random numbers of a distribution
		std::mt19937 mt_engine(seed);
		// bind the random number engine as parameter of the operator() function
		// of the distribuion object
		auto generator = std::bind(distribution, mt_engine);

		// fill all schunks with random bits
		for( uint32 i = num_chunks - 1; i > 0; --i ) 
		{			
			rand[i] = generator();
			num_bits -= sizeof(T) * 8;
		}
		
		// fill the first chunk with the rest of random bits needed
		rand[0] = generator();
		rand[0] &= bitmask2<T>(num_bits);	
	}

	return num_chunks;
}


#ifdef __cplusplus
extern "C" {
#endif

void *createInstance( const char *mode );
void setFieldSize( void *inst, const unsigned long field_size );
void setDummyParameters( void *inst, const unsigned long field_size, const unsigned char *irred_poly, const unsigned long chunks_irred_poly );
void run( void *inst, const unsigned char *what, const unsigned long value, const unsigned long field_size, unsigned char flags, int runs, double *results );
void getResult( unsigned long num_chunks, char *c );
int getMetricsSize( const unsigned char *value_name );
void getMetrics( const unsigned char *value_name, char *metrics );
void setProperty( const unsigned char *value_name, char *property_name, char *property_value );
void getRandomNumber( unsigned long num_chunks, int seed, unsigned char *rn );
void destroyInstance( void *inst );

double convertStringToArray( char *a, int size_chunk_bits, int num_chunks, char *arr);
double convertStringToArray2( char *a, int size_chunk_bits, int num_chunks, char *arr);
double convertArrayToString( char **arr, int size_chunk_bits, int num_chunks, char *a );

#ifdef __cplusplus
}
#endif

#endif // __GF2N_CPU_ARITHMETIC_C_H__