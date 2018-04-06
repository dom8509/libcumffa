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
from PerformanceDataLogger import PerformanceDataLogger


class TestCopyPerformance( GF2nTest ):

	@SetIterateValue(bits=[8192, 16382, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456])
	@PerformanceTest(100)
	def testCopyHostToDevicePerformance( self, bits ):

		# do cuda arithmetic
		f_gpu = GF2nStub.GF2nStub("Cuda", bits, -1)

		a_gpu = f_gpu()
		b_gpu = f_gpu()

		GF2nStub.run("measureKernelLaunchOverhead", a_gpu, b_gpu)
		
		metrics = GF2nStub.getMetrics("bn_a")
		PerformanceDataLogger().addPerfResult("copy host -> device", bits, "Cuda", metrics["copyToDevice_time"])


	@SetIterateValue(bits=[8192, 16382, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456])
	@PerformanceTest(100)
	def testCopyDeviceToHostPerformance( self, bits ):

		# do cuda arithmetic
		f_gpu = GF2nStub.GF2nStub("Cuda", bits, -1)

		a_gpu = f_gpu()
		b_gpu = f_gpu()

		res = GF2nStub.run("parAdd", a_gpu, b_gpu)
		res_value = res._value
		
		metrics = GF2nStub.getMetrics("res")
		PerformanceDataLogger().addPerfResult("copy device -> host", bits, "Cuda", metrics["copyToHost_time"])