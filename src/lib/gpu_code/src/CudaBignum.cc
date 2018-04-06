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

#include "CudaBignum.h"

namespace libcumffa {
    namespace gpu {

        CUDA_BIGNUM *bin2bn( const unsigned char *bin_value, uint32 len )
        {
            // n is the length in bytes
            uint32 n = len;
            // i is the length in chunks
            uint32 i = ((n - 1) / CUDA_BIGNUM_SIZE_BYTES) + 1;
            // m is the current byte in the chunks
            uint32 m = ((n - 1) % (CUDA_BIGNUM_SIZE_BYTES));
            // l is 1 chunk
            CUDA_BIGNUM l = 0;
            // the return value
            CUDA_BIGNUM *ret = new CUDA_BIGNUM[i];
            // index for the current byte
            uint32 k = 0;

            while( n-- ) { // for every byte
                l = (l << 8L) | bin_value[k++];
                if (m-- == 0) {
                    ret[--i] = l;
                    l = 0;
                    m = CUDA_BIGNUM_SIZE_BYTES - 1;
                }
            }    

            return ret;
        }

    }
}