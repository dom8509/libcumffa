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
from ctypes import *
import ctypes
import copy
import traceback
import sys
import numpy
import binascii
import commands
from GF2nBase import *

# Get shared lib of libcumffa
libcumffa = CDLL("../01_Testbench/c_wrapper/build/libcumffa.so")

# Function definitions
libcumffa.createInstance.restype = ctypes.c_void_p

PyLong_AsByteArray = ctypes.pythonapi._PyLong_AsByteArray
PyLong_AsByteArray.argtypes = [ctypes.py_object,
                               ctypes.c_char_p,
                               ctypes.c_size_t,
                               ctypes.c_int,
                               ctypes.c_int]

lastEllapesTime_ms = 0


# returns the ellapsed time of the last executed operation in ms
def getEllapsedTime_ms():
    global lastEllapesTime_ms
    return lastEllapesTime_ms


#############################################################
#
# Classes to define the extension field GF2n
#
#############################################################
class GF2nStub(GF2nBase):

    def __init__(self, mode, field_size, irred_poly=None):
        super(self.__class__, self).__init__(field_size, irred_poly)
        self._mode = mode
        self._inst = libcumffa.createInstance(c_char_p(self._mode))

        if irred_poly is None:
            libcumffa.setFieldSize(
                c_void_p(self._inst), c_ulong(self._field_size))
        elif irred_poly == -1:
            libcumffa.setDummyParameters(
                c_void_p(self._inst),
                c_ulong(self._field_size), None, 0)
        elif type(irred_poly) == str:
            c_ubyte_irred_poly = (c_ubyte * len(self._irred_poly))
            c_ubyte_irred_poly_value = c_ubyte_irred_poly.from_buffer_copy(
                                        self._irred_poly)

            libcumffa.setDummyParameters(
                c_void_p(self._inst), c_ulong(self._field_size),
                byref(c_ubyte_irred_poly_value),
                c_ulong(len(self._irred_poly)))
        else:
            libcumffa.setDummyParameters(
                c_void_p(self._inst), c_ulong(self._field_size),
                c_char_p(str(self._irred_poly)), 0)

    def __del__(self):
        libcumffa.destroyInstance(c_void_p(self._inst))

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            return _GF2nStubElement(-1, self)
        elif len(args) == 1:
            return _GF2nStubElement(args[0], self)
        else:
            raise Exception(
                "Initialization doesn't support more than one parameter!!!")


class _GF2nStubElement(GF2nElementBase):

    def __init__(self, value, field):
        super(self.__class__, self).__init__(value, field)
        self.__value_int = value
        self.__initializes_value = 0

    def __add__(self, other):
        return run("add", self, other)

    def __sub__(self, other):
        return run("sub", self, other)

    def __mul__(self, other):
        return run("mul", self, other)

    def __div__(self, other):
        assert 0

    @property
    def _value(self):
        if self.__initializes_value == 0:
            num_chunks = ((self._field._field_size - 1) / 8) + 1

            res = (c_ubyte * num_chunks).from_buffer(bytearray(num_chunks))

            libcumffa.getResult(num_chunks, byref(res))

            self.__value_int = int(binascii.hexlify(bytearray(res)), 16)

            self.__initializes_value = 1

        return self.__value_int

    @_value.setter
    def _value(self, value):
        self.__value_int = value


def getMetrics(value_name):
    res = create_string_buffer(libcumffa.getMetricsSize(c_char_p(value_name)))
    libcumffa.getMetrics(c_char_p(value_name), res)

    return dict(
        [x.split('=') for x in
         filter(lambda x: x != '', res.value.split('\n'))])


def setProperty(value_name, property_name, property_value):
    libcumffa.setProperty(
        c_char_p(value_name), c_char_p(property_name),
        c_char_p(property_value))


def run(what, a, b, flags=0, runs=1):
    global lastEllapesTime_ms

    res_time = (c_double * runs)()

    if isinstance(b, _GF2nStubElement):
        a._checkField(b)

        libcumffa.run(
            c_void_p(a._field._inst),
            c_char_p(what),
            c_ulong(0),
            c_ulong(a._field._field_size),
            c_ubyte(flags),
            c_int(runs),
            byref(res_time))
    else:
        flags = flags | 4
        libcumffa.run(
            c_void_p(a._field._inst),
            c_char_p(what),
            c_ulong(b),
            c_ulong(a._field._field_size),
            c_ubyte(flags),
            c_int(runs),
            byref(res_time))

    lastEllapesTime_ms = [res_time[i] for i in range(0, runs)]

    return _GF2nStubElement(-1, a._field)


def getRandomNumber(num_bits, seed):
    num_chunks = ((num_bits - 1) / 8) + 1
    c_ubyte_arr_value = (c_ubyte * num_chunks).from_buffer(
                            bytearray(num_chunks))
    libcumffa.getRandomNumber(num_bits, seed, byref(c_ubyte_arr_value))
    return int(binascii.hexlify(bytearray(c_ubyte_arr_value)), 16)


def getRegisterSize():
    registerSize = 32
    if commands.getoutput("uname -m") == "x86_64":
        registerSize = 64
    return registerSize


def convertStringToArray(a, size_chunk_bits):
    global lastEllapesTime_ms

    num_bits = len("{0:b}".format(a))
    num_chunks = (num_bits-1)/size_chunk_bits+1

    # creates a buffer of num_chunks * size_chunk_bits bytes
    res = create_string_buffer(num_chunks * size_chunk_bits)
    lastEllapesTime_ms = libcumffa.convertStringToArray(
        c_char_p(str(a)),
        c_int(size_chunk_bits),
        c_int(num_chunks),
        cast(res, c_char_p))

    return [int(str(res.raw[i:i+size_chunk_bits]), 2) for
            i in range(0, len(res.raw), size_chunk_bits)]


def convertStringToArray2(a, size_chunk_bits):
    global lastEllapesTime_ms

    num_bits = len("{0:b}".format(a))
    num_chunks = (num_bits-1)/size_chunk_bits+1

    # creates a buffer of num_chunks * size_chunk_bits bytes
    res = create_string_buffer(num_chunks * size_chunk_bits)
    lastEllapesTime_ms = libcumffa.convertStringToArray2(
        c_char_p(str(a)),
        c_int(size_chunk_bits),
        c_int(num_chunks),
        cast(res, c_char_p))

    return [int(str(res.raw[i:i+size_chunk_bits]), 2) for
            i in range(0, len(res.raw), size_chunk_bits)]


def convertArrayToString(array, size_chunk_bits):
    global lastEllapesTime_ms

    num_chunks = len(array)

    # how many digits does one chunk have at max
    max_digits_per_chunk = len(str(2**size_chunk_bits-1))
    max_digits_all = len(str(2**((size_chunk_bits * num_chunks)-1)))

    # creates a buffer of max_digits_all + 1 bytes
    res = create_string_buffer(max_digits_all + 1)

    arr_type = (ctypes.c_char_p * (num_chunks + 1))
    arr = arr_type()
    for i in range(0, num_chunks):
        arr[i] = c_char_p(str(array[i]))
    arr[num_chunks] = None

    lastEllapesTime_ms = libcumffa.convertArrayToString(
        arr,
        c_int(size_chunk_bits),
        c_int(num_chunks),
        cast(res, c_char_p))

    # return the result as number
    return int(res.value)


def packl_ctypes(lnum):
    a = ctypes.create_string_buffer(lnum.bit_length()//8 + 1)
    PyLong_AsByteArray(lnum, a, len(a), 0, 1)
    return a.raw
