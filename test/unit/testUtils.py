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

import random
import re

BITS_PER_CHUNK = 64

class TestConvertStringToArray( GF2nTest ):

	@SetIterateValue(bits=[10, 100, 1000, 10000])
	@UnitTest()
	def testConvertStringToArray( self, bits ):
	
		value = random.getrandbits(bits)
		res_cpu = GF2nStub.convertStringToArray(value, BITS_PER_CHUNK)
		res_ref = GF2n.convertStringToArray(value, BITS_PER_CHUNK)

		self.assertEqual(res_cpu, res_ref)


	@SetIterateValue(bits=[10, 100, 1000, 10000])
	@UnitTest()
	def testConvertStringToArrayToStringCPU( self, bits ):

		value = random.getrandbits(bits)
		res_array = GF2nStub.convertStringToArray(value, BITS_PER_CHUNK)
		res_string = GF2nStub.convertArrayToString(res_array, BITS_PER_CHUNK)

		self.assertEqual(value, res_string)		



class TestConvertStringToArrayToStringRef( GF2nTest ):

	@SetIterateValue(bits=[10, 100, 1000, 10000])
	@UnitTest()
	def testConvertStringToArrayToStringRef( self, bits ):

		value = random.getrandbits(bits)
		res_array = GF2n.convertStringToArray(value, BITS_PER_CHUNK)
		res_string = GF2n.convertArrayToString(res_array, BITS_PER_CHUNK)

		self.assertEqual(value, res_string)