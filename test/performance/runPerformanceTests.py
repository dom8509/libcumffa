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
import getopt

sys.path.append("../01_Testbench/")
sys.path.append("../01_Testbench/pyGF2n/")

from GF2nTest import GF2nTestSuite
from PerformanceDataLogger import PerformanceDataLogger


if __name__ == '__main__':
	opts, args = getopt.getopt(sys.argv[1:], "hg:", ["help", "testgroups="])

	testgroup_filter = None

	for o, a in opts:
		if o in ["-h", "--help"]: 
			pass
		elif o in ["-g", "--testgroups"]:
			testgroup_filter = a

	suite = GF2nTestSuite(ts_name="libcumffa Performance Tests")
	suite.run(testgroup_filter)			

	PerformanceDataLogger().analyse_and_print_perf_results()
	PerformanceDataLogger().analyse_and_print_metric_results()
