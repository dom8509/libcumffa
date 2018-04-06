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


class TestMySecondMul(GF2nTest):

    @SetIterateValue(bits=[21])
    @SetIterateValue(func=['parMulChunkedBarRed'])
    @UnitTest()
    def testMySecondMul(self, bits, func):

        # do cuda arithmetic
        f_gpu = GF2nStub.GF2nStub("Cuda", bits)

        a_gpu = f_gpu()
        b_gpu = f_gpu()

        res_gpu = GF2nStub.run(func, a_gpu, b_gpu)

        rand_a = GF2nStub.getRandomNumber(bits, 42)
        rand_b = GF2nStub.getRandomNumber(bits, 84)

        print "a = ", rand_a
        print "b = ", rand_b

        f_ref = GF2n.GF2n(bits)

        a_ref = f_ref(rand_a)
        b_ref = f_ref(rand_b)

        res_ref = a_ref * b_ref

        print "res_ref = ", res_ref

        # compare results
        self.assertEqual(res_gpu, res_ref)
