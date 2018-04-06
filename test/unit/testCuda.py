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
import GF2nStub
import GF2n
import random
import re

from GF2nTest import *
from irred import set_irrep

sys.path.append("../01_Testbench/")
sys.path.append("../01_Testbench/pyGF2n/")


class TestCudaAddition(GF2nTest):

    @SetIterateValue(bits=[4, 100, 1000, 10000])
    @SetIterateValue(func=['parAdd', 'parAddWithEvents',
                           'parAddOwnStream',
                           'parAddOwnStream1024Threads',
                           'parAddOwnStream512Threads',
                           'parAddOwnStream256Threads'])
    @UnitTest()
    def testCudaAddition(self, bits, func):

        # do cuda arithmetic
        f_gpu = GF2nStub.GF2nStub("Cuda", bits, -1)

        a_gpu = f_gpu()
        b_gpu = f_gpu()

        res_gpu = GF2nStub.run(func, a_gpu, b_gpu)

        # calcualte reference
        rand_irred_poly = GF2nStub.getRandomNumber(bits + 1, 23)
        rand_a = GF2nStub.getRandomNumber(bits, 42)
        rand_b = GF2nStub.getRandomNumber(bits, 84)

        f_ref = GF2n.GF2n(bits)

        a_ref = f_ref(rand_a)
        b_ref = f_ref(rand_b)

        res_ref = a_ref + b_ref

        # compare results
        self.assertEqual(res_gpu, res_ref)

    # @SetIterateValue(bits=[16777215])
    # @UnitTest()
    # def testBigKernelAddition(self, bits):

    #     runs = 1

    #     rand_irred_poly = GF2nStub.getRandomNumber(bits + 1, 23)
    #     rand_a = GF2nStub.getRandomNumber(bits, 42)
    #     rand_b = GF2nStub.getRandomNumber(bits, 84)

    #     f_ref = GF2n.GF2n(bits, rand_irred_poly)

    #     a_ref = f_ref(rand_a)
    #     b_ref = f_ref(rand_b)

    #     res_ref = a_ref + b_ref

    #     # do cuda arithmetic
    #     f_gpu = GF2nStub.GF2nStub("Cuda", bits, -1)

    #     a_gpu = f_gpu()
    #     b_gpu = f_gpu()

    #     flags = 0

    #     res_gpu = GF2nStub.run("parAdd", a_gpu, b_gpu, flags, runs)

    #     # compare results
    #     self.assertEqual(res_gpu, res_ref)

    #     flags = 1

    #     for num_grids in [2**n for n in range(0, 8)]:
    #         for num_threads in [256, 512, 1024]:
    #             num_blocks = (bits+1)/32/num_grids/num_threads

    #             GF2nStub.setProperty("bn_a", "num_threads", str(num_threads))
    #             GF2nStub.setProperty("bn_a", "num_blocks", str(num_blocks))

    #             res_gpu = GF2nStub.run("parAddLoop", a_gpu, b_gpu, flags, runs)

    #             self.assertEqual(res_gpu, res_ref)


class TestCudaMultiplication(GF2nTest):

    @SetIterateValue(bits=[10, 100, 1000])
    @SetIterateValue(func=['parMul'])
    @UnitTest()
    def testCudaMultiplication(self, bits, func):

        # do cuda arithmetic
        f_gpu = GF2nStub.GF2nStub("Cuda", bits)

        a_gpu = f_gpu()
        b_gpu = f_gpu()

        res_gpu = GF2nStub.run(func, a_gpu, b_gpu)

        # calcualte reference
        rand_a = GF2nStub.getRandomNumber(bits, 42)
        rand_b = GF2nStub.getRandomNumber(bits, 84)

        f_ref = GF2n.GF2n(bits)

        a_ref = f_ref(rand_a)
        b_ref = f_ref(rand_b)

        res_ref = a_ref * b_ref

        # compare results
        self.assertEqual(res_gpu, res_ref)

    @SetIterateValue(bits=range(2, 1000))
    @SetIterateValue(func=['parMulChunkedBarRed'])
    @UnitTest()
    def testChunkedMul(self, bits, func):

        # do cuda arithmetic
        f_gpu = GF2nStub.GF2nStub("Cuda", bits)

        a_gpu = f_gpu()
        b_gpu = f_gpu()

        res_gpu = GF2nStub.run(func, a_gpu, b_gpu)

        rand_a = GF2nStub.getRandomNumber(bits, 42)
        rand_b = GF2nStub.getRandomNumber(bits, 84)

        f_ref = GF2n.GF2n(bits)

        a_ref = f_ref(rand_a)
        b_ref = f_ref(rand_b)

        res_ref = a_ref * b_ref

        # compare results
        self.assertEqual(res_gpu, res_ref)


class TestCudaExponentiation(GF2nTest):

    @SetIterateValue(bits=[10, 100, 1000, 2000])
    @SetIterateValue(k=range(0, 10))
    @SetIterateValue(func=['parExponentiation'])
    @UnitTest()
    def testCudaExponentiation(self, bits, k, func):

        # do cuda arithmetic
        f_gpu = GF2nStub.GF2nStub("Cuda", bits)

        a_gpu = f_gpu()

        res_gpu = GF2nStub.run(func, a_gpu, k)

        # calcualte reference
        rand_a = GF2nStub.getRandomNumber(bits, 42)

        f_ref = GF2n.GF2n(bits)

        a_ref = f_ref(rand_a)

        res_ref = a_ref ** k

        # compare results
        self.assertEqual(res_gpu, res_ref)


class TestCudaInverseElement(GF2nTest):

    @SetIterateValue(bits=[10, 100, 1000, 2000])
    @SetIterateValue(func=['parInverseElement', 'parInverseElementWithExp'])
    @UnitTest()
    def testCudaInverseElement(self, bits, func):

        # do cuda arithmetic
        f_gpu = GF2nStub.GF2nStub("Cuda", bits)

        a_gpu = f_gpu()

        res_gpu = GF2nStub.run(func, a_gpu, 0)

        # calcualte reference
        rand_a = GF2nStub.getRandomNumber(bits, 42)

        f_ref = GF2n.GF2n(bits)

        a_ref = f_ref(rand_a)

        res_ref = a_ref.inverse()

        # compare results
        self.assertEqual(res_gpu, res_ref)
