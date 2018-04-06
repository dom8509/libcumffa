# include global Makefile options
include ../../Makefile.options

# the name of the generated lib
TARGET=libcumffa.a

# the output directory of the compiler
OUTDIR=build
SRCDIR=src

MODULES:=cpu_code gpu_code

# include directories of the libcumffa
INCDIRS_MOD:=$(foreach module, $(MODULES), -I$(module)/include)
INCDIRS+=$(INCDIRS_MOD) -Iinclude

SRC:=$(shell find $(SRCDIR) -name \*.cc)
SRC_FILES:=$(foreach file, $(SRC), $(shell echo $(file) | sed 's/.*\///'))

# all obj files that are generated
OBJS:=$(foreach file, $(SRC_FILES), $(addprefix $(OUTDIR)/,${file:%.cc=%.o}))
OBJS_MODULES=$(shell find $(MODULES)/$(OUTDIR) -name \*.o)
OBJS_KERNEL=gpu_code/kernels/buils/gpuCode.o

# used libs
INCDIRS+=-I$(HOME)/opt/include
LIBDIRS+=-L$(HOME)/opt/lib
LIBS+=-lcrypto -lgmp

all: lib
	$(info Compiling $(TARGET) finished!)

lib: $(OUTDIR)/$(TARGET)

$(OUTDIR)/$(TARGET): $(OBJS) COPY_KERNEL_STUFF | $(OUTDIR)
	$(info Creating static librariy $@ ...)
	@$(AR) rcs $@ $(OBJS) $(OBJS_MODULES)
	@ranlib $@

COPY_KERNEL_STUFF: $(MODULES)
	$(info Copy additional kernel stuff ...)
	@$(shell cp -r gpu_code/kernels/build/*.* $(OUTDIR)/)

$(MODULES): force_make
	@$(MAKE) -C $@

# pull in dependency info for *existing* .o files
-include $(OBJS:.o=.d)

$(OUTDIR)/%.o: $(SRCDIR)/%.cc
	$(info Compiling $< ...)
	@$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@
	@$(CXX) $(CXXFLAGS) $(INCDIRS) -MM $< > $(OUTDIR)/$*.d
	@mv $(OUTDIR)/$*.d $(OUTDIR)/$*.d.tmp
	@sed 's|.*:|$@:|' < $(OUTDIR)/$*.d.tmp > $(OUTDIR)/$*.d
	@rm -f $(OUTDIR)/$*.d.tmp

$(OUTDIR):
	@mkdir $(OUTDIR)

force_make:
	@true

test: force_make
	$(CXX) -ldl $(CXXFLAGS) $(INCDIRS) $(LIBDIRS) src/test.cc -o $(OUTDIR)/test -Lbuild -lcumffa gpu_code/kernels/build/gpuCode.o -lcumffa_cuda $(LIBS) -lcudart -lcudadevrt $(LDFLAGS)

clean:
	@rm -fR build
	@$(foreach module, $(MODULES), $(MAKE) -C $(module) $@ &&) true