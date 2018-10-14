# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = icc 
OPT =
CFLAGS = -Wall -DGETTIMEOFDAY -std=c99 $(OPT)
LDFLAGS = -Wall
# mkl is needed for blas implementation
LDLIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

BUILD_DIR = .build
OBJDIR = $(BUILD_DIR)/obj
BINDIR = $(BUILD_DIR)/bin
MKDIR_P = mkdir -p

SERVER = thuhpc
SERVER_DIR = ~/workspace/lab1/dgemm
REMOTE = $(SERVER):$(SERVER_DIR)

targets = $(BINDIR)/benchmark-naive \
          $(BINDIR)/benchmark-blocked \
		  $(BINDIR)/benchmark-blas

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

$(BINDIR)/benchmark-naive : $(OBJDIR)/benchmark.o $(OBJDIR)/dgemm-naive.o
	@$(MKDIR_P) $(dir $@)
	@$(CC) -o $@ $^ $(LDLIBS)

$(BINDIR)/benchmark-blocked : $(OBJDIR)/benchmark.o $(OBJDIR)/dgemm-blocked.o
	@$(MKDIR_P) $(dir $@)
	@$(CC) -o $@ $^ $(LDLIBS)

$(BINDIR)/benchmark-blas : $(OBJDIR)/benchmark.o $(OBJDIR)/dgemm-blas.o
	@$(MKDIR_P) $(dir $@)
	@$(CC) -o $@ $^ $(LDLIBS)

$(OBJDIR)/%.o : %.c
	@$(MKDIR_P) $(dir $@)
	@$(CC) $(CFLAGS) -c $< -o $@

.PHONY : clean
clean:
	@rm -rf $(BUILD_DIR)
	@rm -rf job/output*

.PHONY : run
run: $(targets)
	sbatch --wait job/submit

.PHONY : hpc
hpc:
	@rsync -aC . $(REMOTE)
	@echo "cd $(SERVER_DIR) && make && make run" | ssh $(SERVER)
	@mv job/output job/output.prev 2>/dev/null; true
	@rsync $(REMOTE)/job/output job/output
