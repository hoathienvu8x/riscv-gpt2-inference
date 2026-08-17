TARGET = gpt2

CC = gcc

CFLAGS = -std=c99 -Wall -Wextra

LDFLAGS = -lm

# make BUILD=debug / make BUILD=release
BUILD ?= release

ifeq ($(BUILD),debug)
    CFLAGS += -g -O0 -DDEBUG
    BUILD_DIR = build/debug
    $(info [Config] Building in DEBUG mode)
else
    CFLAGS += -O3 -DNDEBUG
    BUILD_DIR = build/release
    $(info [Config] Building in PRODUCTION mode)
endif

SRC = main.c
OBJ = $(BUILD_DIR)/main.o

all: directories $(BUILD_DIR)/$(TARGET)

directories:
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)
	@echo "Build complete: $@ ($(BUILD))"

clean:
	rm -rf build
	@echo "Cleaned build directory."

run: all
	./$(BUILD_DIR)/$(TARGET)

.PHONY: all directories clean run
