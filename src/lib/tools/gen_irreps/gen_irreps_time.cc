/* This file is part of libtrevisan, a modular implementation of
   Trevisan's randomness extraction construction.

   Copyright (C) 2011-2012, Wolfgang Mauerer <wm@linux-kernel.net>

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with libtrevisan. If not, see <http://www.gnu.org/licenses/>. */

// Generate a list of precomputed irreducible polynomials

#include <NTL/GF2X.h>
#include <NTL/GF2XFactoring.h>
#include <NTL/tools.h>
#include <iostream>
#include <sys/time.h>

NTL_CLIENT

unsigned int field_sizes[] = {8192, 16382, 32768, 65536, 131072, 262144, 524288};

// returns time in seconds
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void gen_irreps() 
{
	for( unsigned n = 0; n <= sizeof(field_sizes)/sizeof(unsigned int); n++ ) 
	{
		GF2X P;

		double iStart, iElaps;
		iStart = cpuSecond();

		clog << "calculating irreducible polynomial of size " << std::to_string(field_sizes[n]) << " Bit takes ... ";

		BuildSparseIrred(P, field_sizes[n]);
				
		iElaps = cpuSecond() - iStart;
		clog << std::to_string(iElaps) << "s" << std::endl;
	}
}

int main(int argc, char **argv) {
	gen_irreps();
	return 0;
}
