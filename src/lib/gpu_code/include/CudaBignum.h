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

#ifndef __CUDA_TYPES_H__
#define __CUDA_TYPES_H__

#include <vector>

#include "CumffaTypes.h"

typedef ufixn CUDA_BIGNUM;
typedef std::vector<CUDA_BIGNUM> CUDA_BIGNUM_VEC;

#define CUDA_BIGNUM_SIZE_BYTES sizeof(CUDA_BIGNUM)
#define CUDA_BIGNUM_SIZE_BITS (CUDA_BIGNUM_SIZE_BYTES * 8)

namespace libcumffa {
    namespace gpu {
        CUDA_BIGNUM *bin2bn( const unsigned char *bin_value, uint32 len );
    }
}

#endif // __CUDA_TYPES_H__