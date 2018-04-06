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

#ifndef __CUMFFA_TYPES_H__
#define __CUMFFA_TYPES_H__

#include <ctype.h>

#ifdef LINUXINTEL32
#undef WINDOWS
#include <inttypes.h>
typedef int8_t			int8;
typedef uint8_t			uint8;
typedef int16_t 		int16;
typedef uint16_t 		uint16;
typedef int32_t 		int32;
typedef uint32_t 	 	uint32;
typedef int64_t         int64;
typedef uint64_t 	 	uint64;
typedef uint32 			ufixn;
typedef int32 			sfixn;
#define atoi32(x) 		atoi(x)
#define atol32(x) 		atol(x)
#define atoui64(x) 		std::stoull(x)
#endif

#ifdef LINUXINTEL64
#undef WINDOWS
#include <inttypes.h>
typedef int8_t			int8;
typedef uint8_t			uint8;
typedef int16_t 		int16;
typedef uint16_t 		uint16;
typedef int32_t 		int32;
typedef uint32_t 	 	uint32;
typedef int64_t         int64;
typedef uint64_t 	 	uint64;
typedef uint64 			ufixn;
typedef int64 			sfixn;
#define atoi32(x) 		atoi(x)
#define atol32(x) 		atol(x)
#define atoui64(x) 		std::stoull(x)
#endif

#ifdef MAC64
#undef WINDOWS
#include <cstdlib>
typedef __int8_t 	int8;
typedef __uint8_t 	uint8;
typedef __int32_t   int32;
typedef __int64_t   int64;
typedef __uint32_t  uint32;
typedef __uint64_t  uint64;
typedef uint64 		ufixn;
typedef int64 		sfixn;
#define atoi32(x) 	atoi(x)
#define atoui64(x) 	std::stoull(x)
#endif

#endif // END_OF_FILE
