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

from math import log
import copy
import abc

#############################################################
#
# Classes to define the extension field GF2n
#
#############################################################
class GF2nBase(object):
	__metaclass__ = abc.ABCMeta

	def __init__( self, field_size, irred_poly ):
		self._field_size = field_size
		self._irred_poly = irred_poly


	def __cmp__( self, other ):
		if (self.__class__.__base__ == other.__class__.__base__) and (self._field_size == other._field_size):
			return 0
		else:
			return -1


	def __str__( self ):
		ret_str = ""
		ret_str += "field size: " + str(self._field_size) + "\n"
		ret_str += "irred_poly: " + convertToPolyString("{0:b}".format(self._irred_poly))
		return ret_str


	def order( self ):
		return 2**self._field_size


	def printOrderOfAllElements( self ):
		for i in range(1, self.order() + 1):
			x = self.__call__(i)
			print "Element " + x.prettyPrint() + " has order " + str(x.order())


	@abc.abstractmethod
	def __call__( self, *args, **kwargs ):
		"""Method documentation"""
		return



class GF2nElementBase(object):
	__metaclass__ = abc.ABCMeta

	def __init__( self, value, field ):
		self._value = value
		self._field = field


	def __cmp__( self, other ):
		if (self._field == other._field) & \
			(self._value == other._value):
			return 0
		else:
			return -1


	def getField( self ):
		return self._field


	@abc.abstractmethod
	def __add__( self, other ):
		"""Method documentation"""
		return


	@abc.abstractmethod
	def __sub__( self, other ):
		"""Method documentation"""
		return


	@abc.abstractmethod
	def __mul__( self, other ):
		"""Method documentation"""
		return


	@abc.abstractmethod
	def __div__( self, other ):
		"""Method documentation"""
		return


	def __str__( self ):
		return str(self._value)


	def prettyPrint( self ):
		return convertToPolyString("{0:b}".format(self._value))


	def _checkField( self, other ):
		if(self._field != other._field):
			raise Exception("Fields do not match!")
		return True



def convertToPolyString( bit_str ):
	pretty_str = ""
	if (bit_str is None) or (bit_str == ""):
		pretty_str = ""
	else:
		bit_str_list = list(bit_str)
		bit_str_list_len = len(list(bit_str_list))
		print bit_str_list_len

		indx_list = [i for i,x in enumerate(bit_str_list) if x == "1"]

		for i in range(0, len(indx_list)):
			if pretty_str != "":
				pretty_str += " + "

			if indx_list[i] == (bit_str_list_len - 2):
				pretty_str += "x"
			elif indx_list[i] == (bit_str_list_len - 1):
				pretty_str += "1"
			else: 
				pretty_str += "x^" + str(bit_str_list_len - 1 - indx_list[i])

	return pretty_str
