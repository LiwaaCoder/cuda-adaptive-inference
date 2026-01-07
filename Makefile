NVCC = nvcc
NVCC_FLAGS = -arch=sm_75 -std=c++17 -I./src

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = .

SRCS = $(SRC_DIR)/main.cpp $(SRC_DIR)/kernels.cu
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/kernels.o
TARGET = $(BIN_DIR)/adaptive_engine

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/engine.h
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/kernels.o: $(SRC_DIR)/kernels.cu $(SRC_DIR)/engine.h
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
