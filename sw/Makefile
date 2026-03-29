# Air Hockey Robot - Motor Control Software

# sFoundation paths
SF_ROOT := third_party/sFoundation
SF_LIB_DIR := $(SF_ROOT)/sFoundation
SF_INC := $(SF_ROOT)/inc/inc-pub
SF_SO := $(SF_LIB_DIR)/libsFoundation20.so

CXX := g++
CXXFLAGS := -std=c++11 -O2 -Wall -I$(SF_INC) -Ilib
LDFLAGS := -L$(SF_LIB_DIR) -Wl,-rpath=$(CURDIR)/$(SF_LIB_DIR) -lsFoundation20 -lpthread

BIN_DIR := bin
LIB_DIR := lib
BUILD_DIR := build

# Library objects
LIB_SRCS := $(wildcard $(LIB_DIR)/*.cpp)
LIB_OBJS := $(patsubst $(LIB_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(LIB_SRCS))

# Programs that link against the library
PROGRAMS := $(BIN_DIR)/cdpr_test

# Standalone programs (no library dependency)
STANDALONE := $(BIN_DIR)/test_motor $(BIN_DIR)/scan_motors $(BIN_DIR)/retract_test

.PHONY: all clean sfoundation

all: sfoundation $(STANDALONE) $(PROGRAMS)

sfoundation: $(SF_SO)

$(SF_SO):
	$(MAKE) -C $(SF_LIB_DIR) -j$$(nproc)

$(BUILD_DIR)/%.o: $(LIB_DIR)/%.cpp $(wildcard $(LIB_DIR)/*.h) $(SF_SO) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $@

# Standalone programs (single .cpp, no library)
$(BIN_DIR)/test_motor $(BIN_DIR)/scan_motors $(BIN_DIR)/retract_test: $(BIN_DIR)/%: $(BIN_DIR)/%.cpp $(SF_SO)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Programs that link against the CDPR library
$(PROGRAMS): $(BIN_DIR)/%: $(BIN_DIR)/%.cpp $(LIB_OBJS) $(SF_SO)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_OBJS) $(LDFLAGS)

clean:
	rm -f $(STANDALONE) $(PROGRAMS)
	rm -rf $(BUILD_DIR)
	$(MAKE) -C $(SF_LIB_DIR) clean
