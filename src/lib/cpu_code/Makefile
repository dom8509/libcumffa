# include global Makefile options
include ../../../Makefile.options

# the output directory of the compiler
OUTDIR=build
SRCDIR=src

# include directories
INCDIRS+=-I../include -Iinclude

SRC:=$(shell find src -name \*.cc)
SRC_FILES:=$(foreach file, $(SRC), $(shell echo $(file) | sed 's/.*\///'))

# all obj files that are generatoed
OBJS:=$(foreach file, $(SRC_FILES), $(addprefix $(OUTDIR)/,${file:%.cc=%.o}))

# used libs
INCDIRS+=-I$(HOME)/opt/include
LIBDIRS+=-L$(HOME)/opt/lib
LIBS+=-lcrypto

all: $(OBJS)
	@echo "Compiling cpu_code finished!"

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