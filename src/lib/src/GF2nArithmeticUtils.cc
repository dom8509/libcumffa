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

#include "../include/GF2nArithmetic.h"
#include <gmpxx.h>
#include <string>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <sys/time.h>

void libcumffa::utils::convertStringToArray( std::string str, int size_chunk_bits, int num_chunks, std::vector<std::string> &arr )
{
    mpz_class bn_value(str.c_str(), 10);
    mpz_class bn_base_2 = 2;
    mpz_class bn_num_chunks = num_chunks;

    mpz_class bn_num_field_elements;
    mpz_pow_ui(bn_num_field_elements.get_mpz_t(), bn_base_2.get_mpz_t(), size_chunk_bits);
    mpz_class bn_remain;
    mpz_class bn_curr_chunk;

    while( bn_value != 0 )
    {
        bn_curr_chunk = bn_value % bn_num_field_elements;

        std::string curr_chunk_str = bn_curr_chunk.get_str();
        arr.push_back(curr_chunk_str);

        bn_remain = bn_value / bn_num_field_elements;

        bn_value = bn_remain;
    }

    for( int i = arr.size(); i < num_chunks; ++i )
    {
        arr.push_back(std::string(size_chunk_bits, '0'));
    }

    std::reverse(arr.begin(), arr.end());
}

void libcumffa::utils::convertStringToArray2( std::string str, int size_chunk_bits, int num_chunks, std::vector<std::string> &arr )
{
    mpz_class bn_value(str.c_str(), 10);
    std::string str_binary(mpz_get_str(NULL, 2, bn_value.get_mpz_t()));

    int str_pos = 0;

    for( int i=0; i<num_chunks; ++i )
    {
        if( str_binary.length() >= (num_chunks - i) * size_chunk_bits )  
        {
            std::string current_sub_str(str_binary.substr(str_pos, size_chunk_bits));
            mpz_class bn_current_sub_str(current_sub_str.c_str(), 2);
            arr.push_back(bn_current_sub_str.get_str());
            str_pos += size_chunk_bits;
        }
        else if( str_binary.length() < (num_chunks - i) * size_chunk_bits && str_binary.length() >= (num_chunks - i - 1) * size_chunk_bits )
        {
            int length = str_binary.length() - (num_chunks - i - 1) * size_chunk_bits;
            std::string current_sub_str(str_binary.substr(str_pos, length));
            mpz_class bn_current_sub_str(current_sub_str.c_str(), 2);
            arr.push_back(bn_current_sub_str.get_str());
            str_pos += length;
        }
        else
        {
            arr.push_back(std::string(size_chunk_bits, '0'));
        }
    }
}

void libcumffa::utils::convertArrayToString( std::vector<std::string> &arr, int size_chunk_bits, int num_chunks, std::string &str )
{
    mpz_class bn_value;
    mpz_class bn_base_2 = 2;

    mpz_class bn_num_field_elements;
    mpz_pow_ui(bn_num_field_elements.get_mpz_t(), bn_base_2.get_mpz_t(), size_chunk_bits);

    bn_value.set_str(arr.front(), 10);

    for( unsigned int i = 1; i < arr.size(); ++i )
    {
        bn_value *= bn_num_field_elements;

        mpz_class bn_curr_chunk(arr[i], 10);

        bn_value += bn_curr_chunk;
    }
    
    str = bn_value.get_str();
}