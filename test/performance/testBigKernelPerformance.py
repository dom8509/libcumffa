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
from PerformanceDataLogger import PerformanceDataLogger


class TestBigKernelPerformance( GF2nTest ):

	@SetIterateValue(framework=["Cuda"])
	@SetIterateValue(bits=[16777215, 33554431, 67108863, 134217727, 268435455])
	@UnitTest()
	def testBigKernelPerformance( self, bits, framework ):

		runs = 10

		# do cuda arithmetic
		f_gpu = GF2nStub.GF2nStub(framework, bits, -1)

		a_gpu = f_gpu()
		b_gpu = f_gpu()

		flags = 0
		
		GF2nStub.run("parAdd", a_gpu, b_gpu, flags, runs)
		
		times = GF2nStub.getEllapsedTime_ms()
		for time in times:
			PerformanceDataLogger().addPerfResult("parAdd small", bits, framework, time)
	
		flags = 1
		chunk_size = 32
		if GF2nStub.getRegisterSize() == 64:
			chunk_size = 64

		for num_grids in [2**n for n in range(0,5)]:
			for num_threads in [1024, 512, 256, 128]:
				num_blocks = (bits+1)/chunk_size/num_grids/num_threads
			
				GF2nStub.setProperty("bn_a", "num_threads", str(num_threads))
				GF2nStub.setProperty("bn_a", "num_blocks", str(num_blocks))

				GF2nStub.run("parAddLoop", a_gpu, b_gpu, flags, runs)
					
				times = GF2nStub.getEllapsedTime_ms()
				for time in times:
					PerformanceDataLogger().addPerfResult("parAdd big " + str(num_threads) + " " + str(num_grids), bits, framework, time)
