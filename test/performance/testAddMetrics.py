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
import commands
import re
sys.path.append("../01_Testbench/")
sys.path.append("../01_Testbench/pyGF2n/")
from GF2nTest import *
import GF2nStub
from PerformanceDataLogger import PerformanceDataLogger

def convertUnit( value ):
	pattern = re.compile('([\d\.]+e?[+-]?\d*)(.*)')

	res = float(pattern.search(value).group(1))
	if pattern.search(value).group(1) == 'GB/s':
		res = res * 1000

	return str("%.3f" % res)


class TestAddMetrics( GF2nTest ):

	@SetIterateValue(bits=[8192, 16382, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456])
	@SetIterateValue(func=['parAdd'])
	@UnitTest()
	def testAddMetrics( self, bits, func ):

		runs = 10

		async = 0

		if func == "parAddOwnStream" or func == "parAdd2OwnStream" or func == "parAdd4OwnStream" or func == "parAdd8OwnStream" or func == 'parAddOwnStream1024Threads' or func == 'parAddOwnStream512Threads' or func == 'parAddOwnStream256Threads' or func == 'parAddOwnStream128Threads':
			async = 1


		cmd = "nvprof --normalized-time-unit ms"

		metrics = ["achieved_occupancy", "gld_transactions", "gst_transactions", "inst_per_warp", "gst_throughput", "gld_throughput", "gld_efficiency", "gst_efficiency", "sm_efficiency"]
		events = ["active_warps", "warps_launched", "threads_launched", "gld_request", "gst_request"]

		if metrics:
			cmd = cmd + " --metrics " + ",".join(metrics)
		
		if events:
			cmd = cmd + " --events " + ",".join(events)

		cmd = cmd + " ./runFunction " + func + " " + str(bits) + " " + str(runs) + " " + str(async)

		res = commands.getoutput(cmd).split("\n")

		# store metrics
		for m in metrics:
			line = filter(lambda x: x!="", filter(lambda x: x.find(m)>=0, res)[0].strip().split(" "))
			PerformanceDataLogger().addMetricResult(func, m, bits, runs, convertUnit(line[-3]), convertUnit(line[-2]), convertUnit(line[-1]))

		# store events
		for e in events:
			line = filter(lambda x: x!="", filter(lambda x: x.find(e)>=0, res)[0].strip().split(" "))
			PerformanceDataLogger().addMetricResult(func, e, bits, runs, convertUnit(line[-3]), convertUnit(line[-2]), convertUnit(line[-1]))



	@SetIterateValue(bits=[8192, 16382, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456])
	@SetIterateValue(func=['parAdd'])
	@PerformanceTest(100)
	def testAddTimeDeviation( self, bits, func ):

		runs = 10

		async = 0

		if func == "parAddOwnStream" or func == "parAdd2OwnStream" or func == "parAdd4OwnStream" or func == "parAdd8OwnStream" or func == 'parAddOwnStream1024Threads' or func == 'parAddOwnStream512Threads' or func == 'parAddOwnStream256Threads' or func == 'parAddOwnStream128Threads':
			async = 1


		profiler_cmd = "nvprof --normalized-time-unit ms"

		cmd = profiler_cmd + " ./runFunction " + func + " " + str(bits) + " " + str(runs) + " " + str(async)

		res = commands.getoutput(cmd).split("\n") ## min max avg -> avg min max

		# store netto runtime
		line = filter(lambda x: x!="", filter(lambda x: x.lower().find((func + "kernel").lower())>=0, res)[0].strip().split(" "))
		PerformanceDataLogger().addMetricResult(func, "Kernel runtime", bits, runs, convertUnit(line[4]), convertUnit(line[5]), convertUnit(line[3]))