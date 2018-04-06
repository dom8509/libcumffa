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
from math import ceil

import copy

from GF2nBase import *
from irred import set_irrep


#############################################################
#
# Classes to define the extension field GF2n
#
#############################################################
class GF2:

    def __add__(self, other):
        return self._value ^ other._value

    def __sub__(self, other):
        return self.__add__(other._value)

    def __mul__(self, other):
        return self._value & other._value

    def __pow__(self, n):
        return self._value


class GF2n(GF2nBase):

    def __init__(self, field_size, irred_poly=None):
        if irred_poly is None:
            irred_poly = set_irrep(field_size)
        super(self.__class__, self).__init__(field_size, irred_poly)
        self._mask = 1 << field_size

    def __call__(self, *args, **kwargs):
        if len(args) != 1:
            raise Exception("Initialization needs exactly one value!!!")

        return _GF2nElement(args[0], self)


class _GF2nElement(GF2nElementBase):

    def __init__(self, value, field):
        super(self.__class__, self).__init__(value, field)

    def __add__(self, other):
        self._checkField(other)
        return _GF2nElement(self._value ^ other._value, self._field)

    def __sub__(self, other):
        self._checkField(other)
        return _GF2nElement(self._value ^ other._value, self._field)

    def __mul__(self, other):

        self._checkField(other)

        a = self._value
        b = other._value
        res = 0

        while(b > 0):
            if (b & 1) > 0:
                res ^= a

            a <<= 1
            if (a & self._field._mask) > 0:
                a ^= self._field._irred_poly
            b >>= 1

        return _GF2nElement(res, self._field)

    def __div__(self, other):
        return _GF2nElement(res.__mul__(other.inverse()), self._field)

    def __pow__(self, value):

        if value == 0:
            res = _GF2nElement(1, self._field)
        else:
            res = _GF2nElement(self._value, self._field)

            if abs(value) > 1:
                for i in range(1, abs(value)):
                    res = self.__mul__(res)

            if value < 0:
                res = res.inverse()

        return res

    def inverse(self):
        r = self._value
        s = self._field._irred_poly
        v = 0
        u = 1

        degR = len(int2bin(r)) - 1
        while degR > 0:
            degS = len(int2bin(s)) - 1
            deltaDeg = degS - degR
            if deltaDeg < 0:
                (s, r) = (r, s)
                (v, u) = (u, v)
                deltaDeg = -deltaDeg

            s ^= r << deltaDeg
            v ^= u << deltaDeg

            degR = len(int2bin(r)) - 1

        return _GF2nElement(u, self._field)

    def inverse2(self):
        res = _GF2nElement(self._value, self._field)
        for i in range(0, self._field._field_size - 2):
            res *= res
            res = self.__mul__(res)
        res *= res

        return _GF2nElement(res, self._field)

    def inverse3(self):
        res = _GF2nElement(self._value, self._field)
        value = (2**self._field._field_size) - 2

        mask = 2**32
        while not (mask & value):
            mask >>= 1

        mask >>= 1
        while mask > 0:
            res *= res
            if mask & value:
                res = self.__mul__(res)
            mask >>= 1

        return _GF2nElement(res, self._field)

    def order(self):
        res = self._field.order()

        for i in range(2, self._field.order() - 1):
            if (self._field.order() - 1) % i == 0:
                test = _GF2nElement(self._value, self._field).__pow__(i)
                if test._value == 1:
                    res = i
                    break

        return res


#############################################################
#
# Calculations over the extension field GF2n
#
#############################################################
def int2bin(x):
    return "{0:b}".format(x)


def schoolbookMul(a, b):
    pass


def montgomeryMul(a, b):
    res = 0

    for i in range(0, len(int2bin(b))):
        if (b & 1) > 0:
            res ^= a

        a <<= 1
        b >>= 1

    return res


def montgomeryMulChunked(a, b, size_chunk):
    ar_buffer = []

    res = 0

    num_cols = int(ceil(len(int2bin(b))/float(size_chunk)))
    num_rows = int(ceil(len(int2bin(a))/float(size_chunk)))

    print "num_cols = ", num_cols
    print "num_rows = ", num_rows

    ar_buffer = [[0 for x in range(num_cols)] for x in range(num_rows)]

    mask = 2**size_chunk - 1

    for idx_row in range(num_rows):
        for idx_col in range(num_cols):
            ca = (a & (mask << idx_row * size_chunk)) >> idx_row * size_chunk
            cb = (b & (mask << idx_col * size_chunk)) >> idx_col * size_chunk
            print "ca = ", ca
            print "cb = ", cb
            ar_buffer[idx_row][idx_col] = montgomeryMul(ca, cb)
            print ar_buffer[idx_row][idx_col]

    res = ar_buffer[0][0]

    # sum up the array
    for idx_deg in range(1, (num_cols-1) + (num_rows-1)):
        print "idx_deg = ", idx_deg
        m = idx_deg
        n = 0
        while m >= 0:
            v = (ar_buffer[m][n] << (idx_deg * size_chunk))
            print "adding ", v
            res = res + v
            m = m - 1
            n = n + 1

    res = 0
    res = ar_buffer[0][0]
    res = res ^ (ar_buffer[1][0] << 4)
    res = res ^ (ar_buffer[0][1] << 4)
    res = res ^ (ar_buffer[1][1] << 8)

    return res


def karatsubaMul(a, b):
    """ Recursive multiplication using karatsuba
    a = 2^n/2 * c + d
    b = 2^n/2 * e + f
    x * y = 2^n * ce + 2^(n/2) (cf+de) + df
    where (cf+de) = (c+d)(e+f) - ce - df
    """

    n = max(len(int2bin(a)), len(int2bin(b)))

    if n <= 1:
        return a & b

    n = n if n % 2 == 0 else n + 1
    n_2 = n / 2

    bf_a, bf_b = int2bin(a).zfill(n), int2bin(b).zfill(n)

    c, d = int(bf_a[:n_2], 2), int(bf_a[n_2:], 2)
    e, f = int(bf_b[:n_2], 2), int(bf_b[n_2:], 2)

    ce = karatsubaMul(c, e)
    df = karatsubaMul(d, f)
    cf_de = karatsubaMul((c ^ d), (e ^ f)) ^ ce ^ df

    return (ce << n) ^ (cf_de << n_2) ^ df


def ext_binary_gcd(a, b):
    """Extended binary GCD.
    Given input a, b the function returns
    d, s, t such that gcd(a,b) = d = as + bt."""
    u, v, s, t, r = 1, 0, 0, 1, 0

    while (a % 2 == 0) and (b % 2 == 0):
        a, b, r = a//2, b//2, r+1

    alpha, beta = a, b

    #
    # from here on we maintain a = u * alpha + v * beta
    # and b = s * alpha + t * beta #
    while (a % 2 == 0):
        a = a//2
        if (u % 2 == 0) and (v % 2 == 0):
            u, v = u//2, v//2
        else:
            u, v = (u + beta)//2, (v - alpha)//2

    while a != b:
        if (b % 2 == 0):
            b = b//2

            #
            # Commentary: note that here, since b is even,
            # (i) if s, t are both odd then so are alpha, beta
            # (ii) if s is odd and t even then alpha must be even, so beta is odd
            # (iii) if t is odd and s even then beta must be even, so alpha is odd
            # so for each of (i), (ii) and (iii) s + beta and t - alpha are even #
            if (s % 2 == 0) and (t % 2 == 0):
                s, t = s//2, t//2
            else:
                s, t = (s + beta)//2, (t - alpha)//2
        elif b < a:
            a, b, u, v, s, t = b, a, s, t, u, v
        else:
            b, s, t = b - a, s - u, t - v

    return (2 ** r) * a, s, t


def ext_binary_gcd_env(a, b):
    """Extended binary GCD.
    Given input a, b the function returns
    d, s, t such that gcd(a,b) = d = as + bt."""
    u, v, s, t, r = 1, 0, 0, 1, 0

    while (a & 1 == 0) and (b & 1 == 0):
        a, b, r = a >> 1, b >> 1, r + 1

    alpha, beta = a, b

    #
    # from here on we maintain a = u * alpha + v * beta
    # and b = s * alpha + t * beta #
    while (a & 1 == 0):
        a = a >> 1
        if (u & 1 == 0) and (v & 1 == 0):
            u, v = u >> 1, v >> 1
        else:
            u, v = (u + beta) >> 1, (v - alpha) >> 1

    while a != b:
        if (b & 1 == 0):
            b = b >> 1

            #
            # Commentary: note that here, since b is even,
            # (i) if s, t are both odd then so are alpha, beta
            # (ii) if s is odd and t even then alpha must be even, so beta is odd
            # (iii) if t is odd and s even then beta must be even, so alpha is odd
            # so for each of (i), (ii) and (iii) s + beta and t - alpha are even #
            if (s & 1 == 0) and (t & 1 == 0):
                s, t = s >> 1, t >> 1
            else:
                s, t = (s + beta) >> 1, (t - alpha) >> 1
        elif b < a:
            a, b, u, v, s, t = b, a, s, t, u, v
        else:
            b, s, t = b - a, s - u, t - v

    return a << r, s << r, t << r


def countBinarySize(x):
    res = 0
    while x != 0:
        x = x >> 1
        res = res + 1
    return res


def invert(x, p):
    resultList = []
    bezoutIdentity = []
    r, n, d = x, p, x

    temp = 0

    while r != 0:
        sn = countBinarySize(n)
        sd = countBinarySize(d)
        temp_n = n

        res = 0
        while sn >= sd:
            res = res ^ (1 << (sn - sd))
            temp_n = temp_n ^ (d << (sn - sd))

            sn = countBinarySize(temp_n)

            print "temp_n = " + str(temp_n)
            print "d = " + str(d)

        r = temp_n

        if (r == 0) and (d != 1):
            raise("Cannot compute inverse!")

        resultList.append(n)
        resultList.append(d)
        resultList.append(res)

        n = d
        d = r

        print "r = " + str(r)
        print "n = " + str(n)
        print "d = " + str(d)

    print resultList
    print bezoutIdentity

    if len(resultList) > 3:
        [resultList.pop() for _ in range(0, 3)]

    bezoutIdentity.append(1)
    e3, e2, e1 = resultList.pop(), resultList.pop(), resultList.pop()
    bezoutIdentity.append(e1)
    bezoutIdentity.append(e2)
    bezoutIdentity.append(e3)

    while len(resultList) != 3:
        [resultList.pop() for _ in range(0, 3)]
        e3, e2, e1 = resultList.pop(), resultList.pop(), resultList.pop()

        temp = bezoutIdentity[0]
        bezoutIdentity[0] = bezoutIdentity[3]
        bezoutIdentity[1] = e1
        bezoutIdentity[2] = e2
        bezoutIdentity[3] = temp ^ karatsubaMul(bezoutIdentity[3], e3)

    print "finished"
    print resultList
    print bezoutIdentity


def barrettRed(a, m):
    pass


def parallelExpandVec(x, n):
    return [x] * n


def parallelPrefProdReduce(leaves):

    leavesOut = copy.deepcopy(leaves)
    offset = 1

    i = len(leavesOut) >> 1
    while(i > 0):
        for j in range(0, i):
            leavesOut[(offset*(2*j + 1) - offset)] = \
                leavesOut[(offset * (2 * j + 1) - offset)] \
                * leavesOut[(offset * (2 * j + 2) - offset)]

        i >>= 1
        offset <<= 1

    return leavesOut


def parallelPrefProdDownSweep(leaves):

    leavesOut = copy.deepcopy(leaves)
    leavesOut[0] = (leavesOut[0].getField())(1)

    offset = len(leavesOut)
    i = 1
    while(i < len(leavesOut)):
        offset >>= 1

        for j in range(0, i):
            tmp = leavesOut[(offset*(2*j + 2) - offset)]
            leavesOut[(offset*(2*j + 2) - offset)] = leavesOut[(offset*(2*j + 1) - offset)]
            leavesOut[(offset*(2*j + 1) - offset)] = leavesOut[(offset*(2*j + 1) - offset)] * tmp

        i *= 2

    return leavesOut


def parallelPrefProdMultiply(leaves, coeffs):
    if(len(leaves) != len(coeffs)):
        raise Exception("Wow, there must be the same amount of coeffs as \
                         leaves! What did you think I'll do with that?")
    return [(leaves[i] * coeffs[i]) for i in range(0, len(leaves))]


def parallelPrefProdSum(leaves):
    res = (leaves[0].getField())(0)

    for x in leaves:
        res = res + x

    return res


def convertStringToArray(value, size_chunk_bits):
    # convert string in binary string
    # example:
    #     input = "256"
    #     converted to integer 256
    #     converted to binary string "100000000"
    num_as_bin_str = "{0:b}".format(int(value))

    # get the number of chunks needed for the binary string
    num_chunks = (len(num_as_bin_str) - 1) / size_chunk_bits + 1

    # fill the left side of the string with zeros to fill up the first chunk
    num_as_bin_str = num_as_bin_str.zfill(num_chunks * size_chunk_bits)

    # split the binary string to separate chunks
    chunks = [int(str(num_as_bin_str[i * size_chunk_bits:(i + 1) *
              size_chunk_bits]), 2) for i in range(0, num_chunks)]

    # return the chunks
    return chunks


def convertArrayToString(array, size_chunk_bits):
    # convert string in binary string
    # example:
    #     input = "256"
    #     converted to integer 256
    #     converted to binary string "100000000"

    # return the chunks
    return int(str(''.join(["{0:b}".format(array[i]).zfill(size_chunk_bits) for
                            i in range(0, len(array))])), 2)


#############################################################
#
# Tests for the GF2n Module
#
#############################################################
def checkPackage():
    f = GF2n(3, 11)
    assert [str(x) for x in parallelExpandVec(f(6), 4)] == ['6', '6', '6', '6']
    assert [str(x) for x in parallelPrefProdReduce(parallelExpandVec(f(6), 4))] == ['4', '6', '2', '6']
    assert [str(x) for x in parallelPrefProdDownSweep(parallelPrefProdReduce(parallelExpandVec(f(6), 4)))] == ['7', '2', '6', '1']
    assert str(parallelPrefProdSum([f(4), f(3)])) == '7'


checkPackage()
