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

from __future__ import print_function
import os
import imp
import timeit
import sys

testgroups = set()
registry = {}


class MissingArgumentException( Exception ):
	def __init__( self, fun_name, arg_name ):
		self.__fun_name = fun_name
		self.__arg_name = arg_name

	def __str__( self ):
		return "Exception: The function " + self.__fun_name + " expects the argument " + self.__arg_name + ", but it was not passed!"


class GF2nTestType( type ):
	def __init__( cls, name, bases, attrs ):
		super(GF2nTestType, cls).__init__(name, bases, attrs)
		testgroups.add(cls)


class GF2nTestReportGenerator( object ):
	def update_initTestSuite( self, data ): pass
	def update_finishTestSuite( self, data ): pass
	def update_initTestGroup( self, tg_name, data ): pass
	def update_finishTestGroup( self, tg_name, data ): pass
	def update_initTestCase( self, tg_name, tc_name, data ): pass
	def update_finishTestCase( self, tg_name, tc_name, data ): pass


class GF2nTestReportGeneratorText( GF2nTestReportGenerator ):

	def update_initTestSuite( self, data ): 
		print("")
		print("------------------------------------------------------------")
		print("#")
		print("#\tExecuting Testsuite ", end="")
		if data["ts_name"]:
			print(data["ts_name"])
		else:
			print("")
		print("#")
		print("------------------------------------------------------------")
		print("")

	def update_finishTestSuite( self, data ): 
		print("")
		print("------------------------------------------------------------")
		print("Testcases run: " + str(data["all"]["run"]))
		print("Testcases passed: " + str(data["all"]["passed"]))
		print("Testcases failed: " , end="")
		if data["all"]["failed"] != 0:
			print("\033[91m", end="")
		print(str(data["all"]["failed"]) + "\033[0m")
		print("Execution time: " + str(data["exec_time"]))
		print("Teststatus: ", end="")
		if data["result"] == "passed":
			print("\033[92m" + "PASSED\n" + "\033[0m", end="")	
		else:
			print("\033[91m" + "FAILED\n" + "\033[0m", end="")
		print("")

	def update_initTestGroup( self, tg_name, data ): pass
	def update_finishTestGroup( self, tg_name, data ): pass

	def update_initTestCase( self, tg_name, tc_name, data ): 
		print("Running testcase " + tg_name + "." + tc_name + "...", end="")

	def update_finishTestCase( self, tg_name, tc_name, data) : 
		if data[tc_name]["result"] == "passed":
			print("\033[92m" + "passed\n" + "\033[0m", end="")	
		else:
			print("\033[91m" + "FAILED!\n" + "\033[0m", end="")
			print("\t\033[91m" + data[tc_name]["msg"] + "\n" + "\033[0m", end="")


class GF2nTest( object ):

	__metaclass__ = GF2nTestType

	def __init__( self, name, report_generator ):

		# name of the testgroup
		self._name = name

		# store a ref to the report generator
		self._report_generator = report_generator

		# initialize stats for this testgroup
		self._stat = {"all" : {"passed" : 0, "failed" : 0, "run" : 0}}

		# get all testcases of this testgroup
		self.__testcases = list(set(dir(self.__class__)) - set(dir(self.__class__.__base__)))

		# protected properties for all derived Testcase Classes
		self._currentTestGroup = self.__class__.__name__
		self._currentTestCase = None
		self._currentTestCaseBase = None

	def initTestGroup( self ): pass
	def finishTestGroup( self ): pass	
	def initTestCase( self ): pass
	def finishTestCase( self ): pass
		
	def initTestGroupWrapper( self ):
		self.initTestGroup()
		self._report_generator.update_initTestGroup(self._name, self._stat)

	def finishTestGroupWrapper( self ):
		self.finishTestGroup()
		self._report_generator.update_finishTestGroup(self._name, self._stat)

	def end( self ):
		more_test_available = True
		if len(self.__testcases) > 0:
			more_test_available = (self._currentTestCaseBase == self.__testcases[-1]) | (len(self.__testcases) == 0)
		return more_test_available

	def next( self ):
		if self._currentTestCase == None:
			self.__testcase_idx = 0
		elif self.end() == False:
			self.__testcase_idx += 1

		self._currentTestCaseBase = self.__testcases[self.__testcase_idx]
		self._currentTestCase = self._currentTestCaseBase

	def run( self ):
		func = getattr(self, self._currentTestCaseBase)
		func()

	def addStatToCurrentTestCase( self, key, data ):
		self._stat[self._currentTestCase][key] = data

	def getStats( self ):
		return self._stat

	def assertEqual( self, value1, value2 ):
		if value1 != value2:
			self._stat[self._currentTestCase]["result"] = "failed"
			self._stat[self._currentTestCase]["msg"] = "Assertion Failed: " + str(value1) + " != " + str(value2)



class GF2nTestSuite( object ):
	def __init__( self, ts_name = None, root_path = ".", report_generator = GF2nTestReportGeneratorText() ):
		# get all python file recursive starting at the root_path
		files = self.__getTestFiles(root_path)

		# load all the found files
		self.__loadTestFiles(files)

		self.__stat = {"ts_name" : ts_name, 
					   "result" : "passed", 
					   "exec_time" : 0,
					   "all" : {"passed" : 0, "failed" : 0, "run" : 0}}
		self.__report_generator = report_generator

	def initTestSuite( self ): pass
	def finishTestSuite( self ): pass

	def run( self, testgroup_filter = None ):
		# capture start time
		start = timeit.default_timer()

		# init the test suite
		self.initTestSuite()
		self.__report_generator.update_initTestSuite(self.__stat)

		global testgroups
		# iterate over all thestgroups registered in the
		# global variable testgropus. The variable should
		# contain all classes that derive from the the 
		# base class GF2nTest
		for testgroup in list(testgroups):
			# iterate over all base classes to find out if
			# the base class is GF2nTest. 
			# I don't think that step is really necessary
			for base in testgroup.__bases__:
				# check if the class is a GF2nTest class
				# normaly that should be the case because else
				# it wouldn't be registered in the global 
				# variable testgroups
				if base.__name__ == "GF2nTest" and (testgroup_filter == None or testgroup.__name__ in testgroup_filter):
					# create a new testgroup
					inst = testgroup(testgroup.__name__, self.__report_generator)

					# init the test group
					inst.initTestGroupWrapper()

					while inst.end() == False:
						# if testgroup has more testcases, goto next testcase
						inst.next()

						# run the test case
						inst.run()

					# finish the test group
					inst.finishTestGroupWrapper()

					# get statistics of test run
					self.__stat[testgroup.__name__] = inst.getStats()

					# test if the global test status changed
					if self.__stat[testgroup.__name__]["all"]["failed"] > 0:
						self.__stat["result"] = "failed"

					# update global stats
					self.__stat["all"]["passed"] = self.__stat["all"]["passed"] + self.__stat[testgroup.__name__]["all"]["passed"]
					self.__stat["all"]["failed"] = self.__stat["all"]["failed"] + self.__stat[testgroup.__name__]["all"]["failed"]
					self.__stat["all"]["run"] = self.__stat["all"]["run"] + self.__stat[testgroup.__name__]["all"]["run"]

		# capture stop time and store it to stat
		stop = timeit.default_timer()
		self.__stat["exec_time"] = stop - start

		# finish the test suite
		self.finishTestSuite()
		self.__report_generator.update_finishTestSuite(self.__stat)

	def getResult( self ):
		return self.__stat

	def __getTestFiles( self, root_path ):
		return [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_path) 
			for f in filenames if os.path.splitext(f)[1] == '.py']

	def __loadTestFiles( self, files ):
		[imp.load_source(os.path.splitext(os.path.basename(f))[0], f) for f in files]


# decorators for test cases
def UnitTest( *args, **kwargs ):
	args_unit_test = kwargs
	def UnitTestDecorator( testcase_fun ):
		def UnitTestWrapper( self, *args, **kwargs ):

			all_args = args_unit_test.copy()
			all_args.update(kwargs)

			# parse arguments
			num_args = testcase_fun.func_code.co_argcount - 1
			varnames = testcase_fun.func_code.co_varnames[1:num_args+1]

			fcn_args = {}
			for var in varnames:
				if not all_args.has_key(var):
					raise MissingArgumentException(self._currentTestCase, var)
				else:
					fcn_args[var] = all_args[var]

			if kwargs.has_key("__meta_testcase_name"):
				self._currentTestCase = kwargs["__meta_testcase_name"]

			# init the test case
			self.initTestCase()
			self._report_generator.update_initTestCase(self._name, self._currentTestCase, self._stat)

			self._stat[self._currentTestCase] = {}
			self._stat[self._currentTestCase]["result"] = "passed"
			self._stat[self._currentTestCase]["msg"] = ""

			# run the test case
			testcase_fun( self, **fcn_args )

			self._stat["all"]["run"] = self._stat["all"]["run"] + 1
			if self._stat[self._currentTestCase]["result"] == "passed":
				self._stat["all"]["passed"] = self._stat["all"]["passed"] + 1
			else:
				self._stat["all"]["failed"] = self._stat["all"]["failed"] + 1

			# finish the test case
			self.finishTestCase()
			self._report_generator.update_finishTestCase(self._name, self._currentTestCase, self._stat)

		return UnitTestWrapper
	return UnitTestDecorator


def PerformanceTest( count ):
	def PerformanceTestDecorator( testcase_fun ):
		def PerformanceTestWrapper( self, *args, **kwargs ):
			count_calc = count

			# parse arguments
			num_args = testcase_fun.func_code.co_argcount - 1
			varnames = testcase_fun.func_code.co_varnames[1:num_args+1]

			# parse arguments
			num_args = testcase_fun.func_code.co_argcount - 1
			varnames = testcase_fun.func_code.co_varnames[1:num_args+1]

			fcn_args = {}
			for var in varnames:
				if not kwargs.has_key(var):
					raise MissingArgumentException(self._currentTestCase, var)
				else:
					fcn_args[var] = kwargs[var]

			if kwargs.has_key("__meta_testcase_name"):
				self._currentTestCase = kwargs["__meta_testcase_name"]

			# init the test case
			self.initTestCase()
			self._report_generator.update_initTestCase(self._name, self._currentTestCase, self._stat)

			self._stat[self._currentTestCase] = {}
			self._stat[self._currentTestCase]["result"] = "passed"
			self._stat[self._currentTestCase]["msg"] = ""

			i = 0
			while i < count_calc:
				# run the test case
				testcase_fun( self, **fcn_args )
				i = i + 1

			self._stat["all"]["run"] = self._stat["all"]["run"] + 1
			if self._stat[self._currentTestCase]["result"] == "passed":
				self._stat["all"]["passed"] = self._stat["all"]["passed"] + 1
			else:
				self._stat["all"]["failed"] = self._stat["all"]["failed"] + 1

			# finish the test case
			self.finishTestCase()
			self._report_generator.update_finishTestCase(self._name, self._currentTestCase, self._stat)
		return PerformanceTestWrapper
	return PerformanceTestDecorator	


def SetIterateValue( *args, **kwargs ):
	args_iterate_value = kwargs
	def SetIterateValueDecorator( testcase_func ):
		def SetIterateValueWrapper( self, *args, **kwargs ):
			
			if kwargs.has_key("__meta_testcase_name"):
				currentTestCase = kwargs["__meta_testcase_name"]
			else:
				currentTestCase = self._currentTestCaseBase

			range_calc = args_iterate_value[args_iterate_value.keys()[0]]

			for i in range_calc:
				fcn_args = {args_iterate_value.keys()[0] : i}
				fcn_args.update(kwargs)
				fcn_args["__meta_testcase_name"] = currentTestCase + str(i)
				testcase_func(self, **fcn_args)
		return SetIterateValueWrapper
	return SetIterateValueDecorator