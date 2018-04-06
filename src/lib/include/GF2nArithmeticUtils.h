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

#ifndef __GF2N_ARITHMETICUTILS_H__
#define __GF2N_ARITHMETICUTILS_H__

namespace libcumffa {
	namespace utils {

		template<typename T>
		T calcNumberChunks( size_t num_bits, T size_chunk_bits )
		{
		    T num_chunks = 0;

		    if( num_bits > 0 ) num_chunks = ((num_bits - 1) / size_chunk_bits) + 1;

		    return num_chunks;
		}

	}
}

#endif // __GF2N_ARITHMETICUTILS_H__