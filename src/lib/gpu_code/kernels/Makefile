# include global Makefile options
include ../../../../Makefile.options

# the name of the generated lib
TARGET 	= libcumffa_cuda.a

# the output directory of the compiler
OUTDIR 	= build
SRCDIR 	= .

# include directories
INCDIRS 	+= -I../../include -I../include

SRC 		:= $(shell find . -name \*.cu)
SRC_FILES 	:= $(foreach file, $(SRC), $(shell echo $(file) | sed 's/.*\///'))

# all obj files that are generatoed
OBJS 	:= $(foreach file, $(SRC_FILES), $(addprefix $(OUTDIR)/,${file:%.cu=%.o}))

# used libs
INCDIRS += -I$(HOME)/opt/include
LIBDIRS	+= -L$(HOME)/opt/lib
LIBS 	+=

all: $(OUTDIR)/gpuCode.o
	$(info Compiling cuda kernel code finished!)

$(OUTDIR)/gpuCode.o: $(OUTDIR)/$(TARGET)
	@$(NVCC) $(OUTDIR)/$(TARGET) $(NVCCLDFLAGS) -o $@

$(OUTDIR)/$(TARGET): $(OBJS)
	@$(AR) -rcs $(OUTDIR)/$(TARGET) $(OBJS)
	@rm $(OBJS)

$(OUTDIR)/%.o: $(SRCDIR)/%.cu | $(OUTDIR)
	$(info Compiling $< ...) 
	@$(NVCC) $(NVCCFLAGS) $(INCDIRS) -c $< -o $@

$(OUTDIR):
	@mkdir $(OUTDIR)

clean:
	@rm -fR build
