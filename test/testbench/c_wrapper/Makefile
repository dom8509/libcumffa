# include global Makefile options
include ../../../Makefile.options

# the name of the generated lib
TARGET=libcumffa.so

# the output directory of the compiler
OUTDIR=build
SRCDIR=src

# include directories of the libcumffa
INCDIRS=-Iinclude

SRC:=$(shell find $(SRCDIR) -name \*.cc)
SRC_FILES:=$(foreach file, $(SRC), $(shell echo $(file) | sed 's/.*\///'))

# all obj files that are generated
OBJS:=$(foreach file, $(SRC_FILES), $(addprefix $(OUTDIR)/,${file:%.cc=%.o}))

# use static libcumffa
INCDIRS+=-I../../../src/lib/include
LIBDIRS+=-L../../../src/lib/build
LIBS+=-L/opt/cuda-6.0/lib64 -lcudart -lcudadevrt 

# used libs
INCDIRS+=-I$(HOME)/opt/include
LIBDIRS+=-L$(HOME)/opt/lib
LIBS+=-lcrypto -lgmp -lgmpxx

all: $(OUTDIR)/$(TARGET) 
	@echo "Compiling $(TARGET) finished!"

$(OUTDIR)/$(TARGET): $(OBJS) 
	@echo "Creating dynamic librariy" $@ "..."
	@$(CXX) $(CXXFLAGS) $(LIBDIRS) -shared -o $@ $^ -lcumffa ../../../src/lib/build/gpuCode.o -lcumffa_cuda $(LIBS)

# pull in dependency info for *existing* .o files
-include $(OBJS:.o=.d)

$(OUTDIR)/%.o: $(SRCDIR)/%.cc | $(OUTDIR)
	@echo "Compiling" $< "..."
	@$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@
	@$(CXX) $(CXXFLAGS) $(INCDIRS) -MM $< > $(OUTDIR)/$*.d
	@mv $(OUTDIR)/$*.d $(OUTDIR)/$*.d.tmp
	@sed 's|.*:|$@:|' < $(OUTDIR)/$*.d.tmp > $(OUTDIR)/$*.d
	@rm -f $(OUTDIR)/$*.d.tmp

$(OUTDIR):
	@mkdir $(OUTDIR)

clean:
	@rm -fR build