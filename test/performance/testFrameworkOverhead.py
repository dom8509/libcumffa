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
import timeit
sys.path.append("../01_Testbench/")
sys.path.append("../01_Testbench/pyGF2n/")
from GF2nTest import *
import GF2nStub
from PerformanceDataLogger import PerformanceDataLogger


class TestFrameworkOverhead( GF2nTest ):

	@SetIterateValue(bits=[8192, 16382, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456])
	@PerformanceTest(10)
	def testFrameworkOverhead( self, bits ):

		# do cuda arithmetic
		f_gpu = GF2nStub.GF2nStub("Cuda", bits, -1)

		a_gpu = f_gpu()
		b_gpu = f_gpu()

		# measure test run with python framework and prng
		start_stub_run = timeit.default_timer()
		GF2nStub.run("measureKernelLaunchOverhead", a_gpu, b_gpu)
		PerformanceDataLogger().addPerfResult("FrameworkOverhead - All", bits, "Cuda", timeit.default_timer() - start_stub_run)

		# measure test run without python framwork and without prng
		times = GF2nStub.getEllapsedTime_ms()
		PerformanceDataLogger().addPerfResult("FrameworkOverhead - OnlyFunc", bits, "Cuda", times[0])