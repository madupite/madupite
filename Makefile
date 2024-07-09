include makefile_variables

SRC_DIR := src
EXAMPLES_DIR := examples

# Recursive dispatch
# $(1) - source directory
# $(2) - target
dispatch = @for dir in $(wildcard $(1)/*); do $(MAKE) -C $$dir $(2); done

# Default target (first target in the file)
# Build the shared library and all examples
all: lib build_examples

# Build all examples
build_examples: lib
	+$(call dispatch,$(EXAMPLES_DIR),build)

# Run all examples
run_examples: build_examples
	$(call dispatch,$(EXAMPLES_DIR),run)
	$(call dispatch,$(EXAMPLES_DIR),run2)

# Alias
test: run_examples

# Build the shared library
lib:
	+$(call dispatch,$(SRC_DIR),build)

# Clean up everything
clean:
	$(call dispatch,$(SRC_DIR),clean)
	$(call dispatch,$(EXAMPLES_DIR),clean)

# Format all source files
format:
	$(call dispatch,$(SRC_DIR),format)
	$(call dispatch,$(EXAMPLES_DIR),format)

# Recursive debug print
print:
	@echo SRC_DIR=$(SRC_DIR)
	@echo EXAMPLES_DIR=$(EXAMPLES_DIR)
	@echo \##### Dispatch from $(SRC_DIR) \#####
	$(call dispatch,$(SRC_DIR),print)
	@echo \##### Dispatch from $(EXAMPLES_DIR) \#####
	$(call dispatch,$(EXAMPLES_DIR),print)

# Help and phony targets should list the same stuff
help:
	@echo "make [build|build_examples|clean|format|help|lib|print|run_examples]"

.PHONY: build build_examples clean format help lib print run_examples
