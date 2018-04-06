#include "../include/GF2nArithmetic.h"
#include <iostream>

using namespace libcumffa;

int main(int argc, char *argv[])
{
	GF2nArithmetic arithm = GF2nArithmeticFactory::createInstance("Cuda", 10);

	GF2nArithmeticElement a = arithm.getElement("1");
	GF2nArithmeticElement b = arithm.getElement("4");

	GF2nArithmeticElement c = a + b;

	std::cout << a << " + " << b << " = " << c << std::endl;

	GF2nArithmeticElement d = arithm.getElement("7");
	GF2nArithmeticElement e = arithm.getElement("3");

	GF2nArithmeticElement f = d + e;

	std::cout << d << " + " << e << " = " << f << std::endl;

	unsigned char arr_g[] = {0x3};
	unsigned char arr_h[] = {0x4};

	GF2nArithmeticElement g = arithm.getElement(arr_g, 1);
	GF2nArithmeticElement h = arithm.getElement(arr_h, 1);

	GF2nArithmeticElement i = g * h;
	std::cout << (uint32)arr_h[0] << " * " << (uint32)arr_g[0] << " = " << i << std::endl;

	GF2nArithmetic arithm_2 = GF2nArithmeticFactory::createInstance("Cuda", 60);

	unsigned char arr_j[] = {0x08, 0x99, 0xc7, 0xbb, 0x2e, 0xf3, 0xcb, 0x5f};
	unsigned char arr_k[] = {0x3};

	GF2nArithmeticElement j = arithm_2.getElement(arr_j, 8);
	GF2nArithmeticElement k = arithm_2.getElement(arr_k, 1);

	GF2nArithmeticElement l = j * k;

	std::vector<uint8> res;
	l.getValue(res);
	std::cout << (uint32)arr_j[0] << " * " << *((uint32 *)(&arr_k[0])) << " " << *((uint32 *)(&arr_k[4])) << " = " << i << std::endl;

	return 0;
}