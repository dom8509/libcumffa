# include global Makefile options
include ../../../Makefile.options

# the output directory of the compiler
OUTDIR=build
SRCDIR=src

# use static libcumffa
INCDIRS+=-I../../../src/lib/include -I../c_wrapper/include
LIBDIRS+=-L../../../src/lib/build
LIBS+=-L/opt/cuda-6.0/lib64 -lcudart -lcudadevrt

# used libs
INCDIRS+=-I$(HOME)/opt/include
LIBDIRS+=-L$(HOME)/opt/lib
LIBS+=-lcrypto -lgmp -lgmpxx

all: runFunction runFunction2047

runFunction: src/runFunction.cc
	@echo "Creating runFunction ..."
	@$(CXX) $(CXXFLAGS) $(INCDIRS) $(LIBDIRS) -o $@ $(SRCDIR)/runFunction.cc -lcumffa ../../../src/lib/build/gpuCode.o -lcumffa_cuda $(LIBS)

runFunction2047: src/runFunction2047.cc
	@echo "Creating runFunction2047 ..."
	@$(CXX) $(CXXFLAGS) $(INCDIRS) $(LIBDIRS) -o $@ $(SRCDIR)/runFunction2047.cc -lcumffa ../../../src/lib/build/gpuCode.o -lcumffa_cuda $(LIBS)

clean:
	@rm -f runFunction
	@rm -f runFunction2047