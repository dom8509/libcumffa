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


class TestAddPerformance(GF2nTest):

    @SetIterateValue(framework=["Cuda", "OpenSSL"])
    @SetIterateValue(bits=[8192, 16382, 32768, 65536,
                           131072, 262144, 524288,
                           1048576, 2097152, 4194304,
                           8388608, 16777216, 33554432,
                           67108864, 134217728, 268435456])
    @SetIterateValue(func=['parAdd', 'parAddWithEvents',
                           'parAddOwnStream',
                           'parAddOwnStream1024Threads',
                           'parAddOwnStream512Threads',
                           'parAddOwnStream256Threads',
                           'parAddOwnStream128Threads'])
    @UnitTest()
    def testAddPerformance(self, bits, framework, func):

        runs = 100

        # do cuda arithmetic
        f_gpu = GF2nStub.GF2nStub(framework, bits, -1)

        a_gpu = f_gpu()
        b_gpu = f_gpu()

        flags = 0

        if (func == "parAddOwnStream" or
           func == "parAdd2OwnStream" or
           func == "parAdd4OwnStream" or
           func == "parAdd8OwnStream" or
           func == 'parAddOwnStream1024Threads' or
           func == 'parAddOwnStream512Threads' or
           func == 'parAddOwnStream256Threads' or
           func == 'parAddOwnStream128Threads') \
           and \
           framework == "Cuda":
            flags = flags | 2

        if framework == "Cuda":
            GF2nStub.run(func, a_gpu, b_gpu, flags, runs)
        else:
            GF2nStub.run("add", a_gpu, b_gpu, flags, runs)

        times = GF2nStub.getEllapsedTime_ms()
        for time in times:
            PerformanceDataLogger().addPerfResult(
              func, bits, framework, time)
