DEBUG=-g
OPTIMISE=-O0

INCDIRS=-I$(HOME)/opt/include -I../../include
LIBDIRS=-L$(HOME)/opt/lib
LIBS=-lntl -lgf2x -lgmp

CXX=g++
CXXFLAGS=-std=c++11 -DLINUXINTEL32

all: gen_irreps gen_irreps_time gen_irreps_array

gen_irreps: gen_irreps.cc
	$(CXX) $(CXXFLAGS) $(INCDIRS) $(LIBDIRS) $^ $(LIBS) -o $@ 

gen_irreps_time: gen_irreps_time.cc
	$(CXX) $(CXXFLAGS) $(INCDIRS) $(LIBDIRS) $^ $(LIBS) -o $@

gen_irreps_array: gen_irreps_array.cc
	$(CXX) $(CXXFLAGS) $(INCDIRS) $(LIBDIRS) $^ $(LIBS) -o $@

clean:
	@rm -f gen_irreps
	@rm -f gen_irreps_time
	@rm -f gen_irreps_array
