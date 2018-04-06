#include <GF2nArithmetic.h>
#include <GF2nArithmetic_C.h>
#include <gmpxx.h>
#include <iostream>
#include <string>
#include <sstream>

template <typename T>
std::string tostring(const T& t)
{
    std::ostringstream ss;
    ss << t;
    return ss.str();
}

using namespace libcumffa;
int main( int argc, char *argv[] )
{
	if( argc < 5 || (!std::string(argv[1]).compare("parAddLoop") && argc != 7) )
	{
		std::cout << "Wrong number of parameters!" << std::endl;
		std::cout << "Run a function by calling ./runFunction <fcn> <bits> <runs> <async> <gird_size> <block_size>" << std::endl;
		std::cout << "Example: ./runFunction parAdd 10000 100 0" << std::endl;
	}
	else
	{
		GF2nArithmetic inst = GF2nArithmeticFactory::createInstance("Cuda");
		uint64_t field_size = atoi(argv[2]);

		std::string async_str(argv[4]);
		if( async_str.compare("1") == 0 )
		{
			inst.setFlags((char)0x2);
		}

		uint64_t num_chunks = 0;
		uint64_t size_irred_poly = field_size + 1;

		std::vector<ufixn> irred_poly;

		if( size_irred_poly > 0 ) num_chunks = ((size_irred_poly - 1) / (sizeof(ufixn) * 8)) + 1;
		create_randomness(size_irred_poly, num_chunks, 23, irred_poly);

		if( field_size <= 2048 )
		{
			inst.setFieldSize(field_size);
		}
		else
		{
			inst.setDummyParameters(field_size, &irred_poly[0], num_chunks);
		}

		std::vector<ufixn> rand_a;
		std::vector<ufixn> rand_b;

		if( field_size > 0 ) num_chunks = ((field_size - 1) / (sizeof(ufixn) * 8)) + 1;
		create_randomness(field_size, num_chunks, 42, rand_a);
		create_randomness(field_size, num_chunks, 84, rand_b);


		GF2nArithmeticElement gpu_a = inst.getElement((void *)&rand_a[0], num_chunks);
		GF2nArithmeticElement gpu_b = inst.getElement((void *)&rand_b[0], num_chunks);

		std::string fun_str(argv[1]);

		if( fun_str.compare("parAddLoop") == 0 )
		{
			uint64_t num_grids = strtoull(argv[5], NULL, 0);
			uint64_t num_threads = strtoull(argv[6], NULL, 0);
			uint64_t num_blocks = (size_irred_poly)/(sizeof(uint64_t)*8)/num_grids/num_threads;
			gpu_a.setProperty("num_blocks", tostring(num_blocks));
			gpu_a.setProperty("num_threads", std::string(argv[6]));
		}

		for( int i=0; i<atoi(argv[3]); ++i ) {	
			gpu_a.run(fun_str, gpu_b);
		}
	}

	return 0;
}
