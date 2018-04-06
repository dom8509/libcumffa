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

from GF2nTest import *
from PerformanceDataLogger import PerformanceDataLogger

sys.path.append("../01_Testbench/")
sys.path.append("../01_Testbench/pyGF2n/")


class TestMulPerformance(GF2nTest):

    @SetIterateValue(framework=["Cuda", "OpenSSL"])
    @SetIterateValue(bits=[128, 512, 1024, 2048, 8191, 16381, 32767])  # , 65535, 131071, 262143, 524287])
    @SetIterateValue(function=["parMulChunkedBarRed", "parMul"])
    @UnitTest()
    def testMulPerformance(self, bits, function, framework):

        runs = 10

        rand_irred_poly = GF2nStub.getRandomNumber(bits + 1, 23)
        f = GF2nStub.GF2nStub(framework, bits, rand_irred_poly | 1)

        a = f()
        b = f()

        if framework == "OpenSSL":
            res = GF2nStub.run("mul", a, b, 0, runs)
        else:
            res = GF2nStub.run(function, a, b, 0, runs)

        times = GF2nStub.getEllapsedTime_ms()
        for time in times:
            PerformanceDataLogger().addPerfResult(function, bits,
                                                  framework, time)
