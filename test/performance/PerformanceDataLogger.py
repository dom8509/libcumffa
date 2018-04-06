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

from prettytable import PrettyTable
import re
import pickle
import os
import time
import datetime
import sqlite3 as lite
import sys

log_data = dict()


class PerformanceDataLogger( object ):

	_instance = None

	def __new__( cls, *args, **kwargs ):

		if not cls._instance:

			cls._instance = object().__new__(cls, *args, **kwargs)

			cls.__con = None

			# create performance test result table if not exists
			try:
			    cls.__con = lite.connect('result_data.db')
			    
			    cur = cls.__con.cursor()    

			    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='RES_PERF_TEST';")

			    data = cur.fetchone()

			    if not data: 
			    	# table does not exist
			    	cur.execute('''CREATE TABLE RES_PERF_TEST(
			    		ID 			INTEGER PRIMARY KEY,
					   	TEST_TIME   DATETIME    	NOT NULL,
					   	FUNC_NAME 	CHAR(50)		NOT NULL,
   						FRAMEWORK 	CHAR(50) 		NOT NULL,
   						BITS	    INT	 			NOT NULL,
   						DURATION    REAL 			NOT NULL
						);''')

			    	cls.__con.commit()

			    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='RES_METRIC_TEST';")

			    data = cur.fetchone()

			    if not data: 
			    	# table does not exist
			    	cur.execute('''CREATE TABLE RES_METRIC_TEST(
			    		ID 			INTEGER PRIMARY KEY,
					   	TEST_TIME   DATETIME    	NOT NULL,
					   	FUNC_NAME 	CHAR(50)		NOT NULL,
   						METRIC_NAME CHAR(50) 		NOT NULL,
   						BITS	    INT	 			NOT NULL,
   						RUNS 		INT 			NOT NULL,
   						MIN 	    REAL 			NOT NULL,
   						MAX 	    REAL 			NOT NULL,
   						AVG 	    REAL 			NOT NULL
						);''')

			    	cls.__con.commit()			    	
			    
			except lite.Error, e:
			    
			    print "Error %s:" % e.args[0]
			    sys.exit(1)

		return cls._instance



	def __del__( self ):
	    if self.__con:
	        self.__con.close()



	def addPerfResult( self, function, bits, framework, time ):

		cur = self.__con.cursor()

		cur.execute("INSERT INTO RES_PERF_TEST(TEST_TIME, FUNC_NAME, FRAMEWORK, BITS, DURATION) VALUES (CURRENT_TIMESTAMP, \'" + function + "\', \'" + framework + "\', " + str(bits) + ", " + str(time) + ")")

		self.__con.commit()



	def addMetricResult( self, function, metric, bits, runs, mmin, mmax, mavg ):

		cur = self.__con.cursor()

		cur.execute("INSERT INTO RES_METRIC_TEST(TEST_TIME, FUNC_NAME, METRIC_NAME, BITS, RUNS, MIN, MAX, AVG) VALUES (CURRENT_TIMESTAMP, \'" + function + "\', \'" + metric + "\', " + str(bits) + ", " + str(runs) + ", " + str(mmin) + ", " + str(mmax) + ", " + str(mavg) + ")")

		self.__con.commit()



	def analyse_and_print_perf_results( self ):
		global log_data

		print "Performance Summary:"

		self.__con.row_factory = lambda cursor, row: row[0]
		self.__con.text_factory = str
		cur = self.__con.cursor()

		frameworks = cur.execute("select FRAMEWORK from RES_PERF_TEST GROUP BY FRAMEWORK;").fetchall()

		header = ["Function", "Bits"]
		for framework in frameworks:
			header.extend([framework + " invocations", framework + " avg. time (ms)", framework + " variance"])
		
		table = PrettyTable(header)

		functions = cur.execute("select FUNC_NAME from RES_PERF_TEST GROUP BY FUNC_NAME;").fetchall()

		for function in functions:

			bits = cur.execute("select BITS from RES_PERF_TEST where FUNC_NAME=\'" + function + "\' GROUP BY BITS;").fetchall()

			for bit in bits:
			
				row = [function, bit]

				curr_frameworks = cur.execute("select FRAMEWORK from RES_PERF_TEST where FUNC_NAME=\'" + function + "\' and BITS=" + str(bit) +  " GROUP BY FRAMEWORK;").fetchall()

				for framework in frameworks:

					if framework in curr_frameworks:

						curr_data = cur.execute("select DURATION from RES_PERF_TEST where FUNC_NAME=\'" + function + "\' and BITS=" + str(bit) +  " and FRAMEWORK=\'" + framework + "\';").fetchall()

						num_values = len(curr_data)

						# add num runs for this framework
						row.extend([str(num_values)])

						# add mean for this framework
						value_sum = sum(curr_data)
						
						mean = value_sum / num_values
						row.extend([str(mean)])

						# add variance for this framework
						variance = sum( [(mean - value) ** 2 for value in curr_data] )
						
						row.extend([str(variance / num_values)])

					else:

						row.extend(str(0))
						row.extend(str(0))
						row.extend(str(0))
				
				table.add_row(row)

		print table



	def analyse_and_print_metric_results( self ):
		global log_data

		print "Metric Summary:"

		self.__con.row_factory = lambda cursor, row: row[0]
		self.__con.text_factory = str
		cur = self.__con.cursor()

		header = ["Function", "Metric", "Bits", "Invocations", "Min", "Max", "Avg"]
		
		table = PrettyTable(header)

		functions = cur.execute("select FUNC_NAME from RES_METRIC_TEST GROUP BY FUNC_NAME;").fetchall()

		for function in functions:

			metrics = cur.execute("select METRIC_NAME from RES_METRIC_TEST where FUNC_NAME=\'" + function + "\' GROUP BY METRIC_NAME;").fetchall()

			for metric in metrics:

				bits = cur.execute("select BITS from RES_METRIC_TEST where FUNC_NAME=\'" + function + "\' and METRIC_NAME=\'" + metric +  "\' GROUP BY BITS;").fetchall()

				for bit in bits:

					invocations = cur.execute("select RUNS from RES_METRIC_TEST where FUNC_NAME=\'" + function + "\' and METRIC_NAME=\'" + metric +  "\' and BITS=" + str(bit) + ";").fetchall()
					mins = cur.execute("select MIN from RES_METRIC_TEST where FUNC_NAME=\'" + function + "\' and METRIC_NAME=\'" + metric +  "\' and BITS=" + str(bit) + ";").fetchall()
					maxs = cur.execute("select MAX from RES_METRIC_TEST where FUNC_NAME=\'" + function + "\' and METRIC_NAME=\'" + metric +  "\' and BITS=" + str(bit) + ";").fetchall()
					avgs = cur.execute("select AVG from RES_METRIC_TEST where FUNC_NAME=\'" + function + "\' and METRIC_NAME=\'" + metric +  "\' and BITS=" + str(bit) + ";").fetchall()

					table.add_row([function, metric, bit, sum(invocations), sum(mins)/len(mins), sum(maxs)/len(maxs), sum(avgs)/len(avgs)])

		print table		