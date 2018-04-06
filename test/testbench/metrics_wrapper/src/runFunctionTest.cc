#include <GF2nArithmetic.h>
#include <GF2nArithmetic_C.h>
#include <gmpxx.h>
#include <iostream>
#include <string>

using namespace libcumffa;
int main( int argc, char *argv[] )
{
    GF2nArithmetic inst = GF2nArithmeticFactory::createInstance("Cuda");
    uint64_t field_size = 2047;

    std::string async_str(0);
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

    GF2nArithmeticElement gpu_a = inst.getElement((void *)&rand_a[0], num_chunks);

    return 0;
}