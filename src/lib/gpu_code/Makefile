# include global Makefile options
include ../../../Makefile.options

# the output directory of the compiler
OUTDIR=build
KERNEL_DIR=kernels
SRCDIR=src

# include directories
INCDIRS+=-I../include -Iinclude -Ikernels

SRC:=$(shell find src -name \*.cc)
SRC_FILES:=$(foreach file, $(SRC), $(shell echo $(file) | sed 's/.*\///'))

# all obj files that are generatoed
OBJS:=$(foreach file, $(SRC_FILES), $(addprefix $(OUTDIR)/,${file:%.cc=%.o}))

# used libs
INCDIRS+=-I$(HOME)/opt/include
LIBDIRS+=-L$(HOME)/opt/lib
LIBS+=-lcrypto

all: KERNEL_OBJS $(OBJS)
	$(info Compiling gpu_code finished!)

KERNEL_OBJS: force_make
	@$(MAKE) -C kernels

# pull in dependency info for *existing* .o files
-include $(OBJS:.o=.d)

$(OUTDIR)/%.o: $(SRCDIR)/%.cc | $(OUTDIR)
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

clean:
	@$(MAKE) -C kernels $@
	@rm -fR build