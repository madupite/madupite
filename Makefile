# -*- mode: makefile -*-

#  Adapted from the PETSc sample GNU makefile $PETSC_DIR/share/petsc/Makefile.user
#  You must set the environmental variable(s) PETSC_DIR (and PETSC_ARCH if PETSc was not configured with the --prefix option)

#  The following variable must either be a path to petsc.pc or just "petsc" if petsc.pc
#  has been installed to a system location or can be found in PKG_CONFIG_PATH.
petsc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/petsc.pc

# Additional libraries that support pkg-config can be added to the list of PACKAGES below.
PACKAGES := $(petsc.pc)

# Set the compiler and linker flags from the pkg-config output for PETSc
CC := $(shell pkg-config --variable=ccompiler $(PACKAGES))
CXX := $(shell pkg-config --variable=cxxcompiler $(PACKAGES))
CFLAGS := $(shell pkg-config --variable=cflags_extra $(PACKAGES)) $(CFLAGS_OTHER)
CXXFLAGS := $(shell pkg-config --variable=cxxflags_extra $(PACKAGES)) $(CFLAGS_OTHER)
CPPFLAGS := $(shell pkg-config --cflags-only-I $(PACKAGES))
LDFLAGS := $(shell pkg-config --libs-only-L --libs-only-other $(PACKAGES))
LDFLAGS += $(patsubst -L%, $(shell pkg-config --variable=ldflag_rpath $(PACKAGES))%, $(shell pkg-config --libs-only-L $(PACKAGES)))
LDLIBS := $(shell pkg-config --libs-only-l $(PACKAGES)) -lm
# CUDAC := $(shell pkg-config --variable=cudacompiler $(PACKAGES))
# CUDAC_FLAGS := $(shell pkg-config --variable=cudaflags_extra $(PACKAGES))
# CUDA_LIB := $(shell pkg-config --variable=cudalib $(PACKAGES))
# CUDA_INCLUDE := $(shell pkg-config --variable=cudainclude $(PACKAGES))

MADUPITE_LDLIBS = -lmadupite
MADUPITE_LDFLAGS = -L${CURDIR}/lib -Wl,-rpath,${CURDIR}/lib
MADUPITE_INCLUDE = -I${CURDIR}/include
MADUPITE_LIB := $(CURDIR)/lib/libmadupite.so
MADUPITE_BIN := $(CURDIR)/bin/example

CPPFLAGS += $(MADUPITE_INCLUDE)

# Workaround for the -fvisibility=hidden flag that is added by PETSc
# - we currently need to have symbols exported by default -
# and also add -MMD -MP to generate header dependencies
CFLAGS := $(subst  -fvisibility=hidden,,$(CFLAGS)) -MMD -MP
CXXFLAGS := $(subst  -fvisibility=hidden,,$(CXXFLAGS)) -MMD -MP

# We remove compiler and preprocessor flags from the link command for conciseness
LINK.cc := $(subst $(CXXFLAGS),,$(LINK.cc))
LINK.cc := $(subst $(CPPFLAGS),,$(LINK.cc))

# Define source and build directories
SRC_DIR := src
BUILD_DIR := build

# Find all .cpp files in the src directory (including subdirectories)
SRCS := $(shell find $(SRC_DIR) -name '*.cpp')

# Generate corresponding .o files in the build directory
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Include the .d files (header dependencies produced by the -MMD -MP flags in CFLAGS and CXXFLAGS)
DEPS = $(OBJS:.o=.d)
-include $(DEPS)

# Default target (first target in the file)
all: example

# Explicit rules
#   (see https://www.gnu.org/software/make/manual/html_node/Catalogue-of-Rules.html#Catalogue-of-Rules for implicit rules)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

$(BUILD_DIR)/example.o: example/example.cpp
	@echo "Building example"
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

$(MADUPITE_LIB): $(OBJS)
	mkdir -p lib
	$(LINK.cc) -shared -o $@ $^ $(LDLIBS)

$(MADUPITE_BIN): $(BUILD_DIR)/example.o $(MADUPITE_LIB)
	mkdir -p bin
	$(LINK.cc) $(MADUPITE_LDFLAGS) -o $@ $< $(MADUPITE_LDLIBS) $(LDLIBS)

# Phony targets
.PHONY: all lib example run_example clean print help

lib: $(MADUPITE_LIB)

example: $(MADUPITE_BIN)

run_example: example
	$(MADUPITE_BIN)

clean:
	rm -f $(OBJS) $(BUILD_DIR)/example.o $(DEPS) $(MADUPITE_LIB) $(MADUPITE_BIN)

print:
	@echo MADUPITE_INCLUDE=$(MADUPITE_INCLUDE)
	@echo MADUPITE_LDLIBS=$(MADUPITE_LDLIBS)
	@echo MADUPITE_LDFLAGS=$(MADUPITE_LDFLAGS)
	@echo MADUPITE_LIB=$(MADUPITE_LIB)
	@echo MADUPITE_BIN=$(MADUPITE_BIN)
	@echo \#
	@echo CC=$(CC)
	@echo CXX=$(CXX)
	@echo CFLAGS=$(CFLAGS)
	@echo CXXFLAGS=$(CXXFLAGS)
	@echo CPPFLAGS=$(CPPFLAGS)
	@echo LDFLAGS=$(LDFLAGS)
	@echo LDLIBS=$(LDLIBS)
	@echo \#
	@echo COMPILE.cc=$(COMPILE.cc)
	@echo LINK.cc=$(LINK.cc)
	@echo \#
	@echo SRCS=$(SRCS)
	@echo OBJS=$(OBJS)
	@echo DEPS=$(DEPS)

help:
	@echo "make [all|lib|example|run_example|clean|print|help]"
