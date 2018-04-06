# cubffa (CUda Binary Finite Field Arithmetic library) provides
# functions for large binary galois field arithmetic on GPUs.
# Besides CUDA it is also possible to extend cubffa to any other
# underlying framework.
# Copyright (C) 2016  Dominik Stamm
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
sys.path.append("../01_Testbench/")
sys.path.append("../01_Testbench/pyGF2n/")
from GF2nTest import *
import GF2nStub
import GF2n
from irred import set_irrep

import random
import re


class TestOpenSSLAddition( GF2nTest ):

	@SetIterateValue(bits=[100, 1000, 2000])
	@UnitTest()
	def testOpenSSLAddition( self, bits ):
	
		# do OpenSSL arithmetic
		f_cpu = GF2nStub.GF2nStub("OpenSSL", bits, -1)

		a_cpu = f_cpu()
		b_cpu = f_cpu()

		res_cpu = a_cpu + b_cpu

		# calcualte reference
		rand_irred_poly = GF2nStub.getRandomNumber(bits + 1, 23)
		rand_a = GF2nStub.getRandomNumber(bits, 42)
		rand_b = GF2nStub.getRandomNumber(bits, 84)

		f_ref = GF2n.GF2n(bits, rand_irred_poly)

		a_ref = f_ref(rand_a)
		b_ref = f_ref(rand_b)

		res_ref = a_ref + b_ref
		
		# compare results
		self.assertEqual(res_cpu, res_ref)


class TestOpenSSLMultiplication( GF2nTest ):

	@SetIterateValue(bits=[100, 1000, 2000])
	@UnitTest()
	def testOpenSSLMultiplication( self, bits ):
	
		# do OpenSSL arithmetic
		f_cpu = GF2nStub.GF2nStub("OpenSSL", bits)

		a_cpu = f_cpu()
		b_cpu = f_cpu()

		res_cpu = a_cpu * b_cpu

		# calcualte reference
		rand_a = GF2nStub.getRandomNumber(bits, 42)
		rand_b = GF2nStub.getRandomNumber(bits, 84)

		f_ref = GF2n.GF2n(bits)

		a_ref = f_ref(rand_a)
		b_ref = f_ref(rand_b)

		res_ref = a_ref * b_ref
		
		# compare results
		self.assertEqual(res_cpu, res_ref)


class TestOpenSSLExponentiation(GF2nTest):

    @SetIterateValue(bits=[10, 100, 1000, 2000])
    @SetIterateValue(exp=range(1, 10))
    @UnitTest()
    def testOpenSSLExponentiation(self, bits, exp):

        # do cuda arithmetic
        f_gpu = GF2nStub.GF2nStub("OpenSSL", bits)

        a_gpu = f_gpu()

        res_gpu = GF2nStub.run("exp", a_gpu, exp)

        # calcualte reference
        rand_a = GF2nStub.getRandomNumber(bits, 42)

        f_ref = GF2n.GF2n(bits)

        a_ref = f_ref(rand_a)

        res_ref = a_ref ** exp

        # compare results
        self.assertEqual(res_gpu, res_ref)


class TestOpenSSLInverseElement(GF2nTest):

    @SetIterateValue(bits=[10, 100, 1000, 2000])
    @UnitTest()
    def testOpenSSLInverseElement(self, bits):

        # do cuda arithmetic
        f_gpu = GF2nStub.GF2nStub("OpenSSL", bits)

        a_gpu = f_gpu()

        res_gpu = GF2nStub.run("inverse", a_gpu, 0)

        # calcualte reference
        rand_a = GF2nStub.getRandomNumber(bits, 42)

        f_ref = GF2n.GF2n(bits)

        a_ref = f_ref(rand_a)

        res_ref = a_ref.inverse()

        # compare results
        self.assertEqual(res_gpu, res_ref)
