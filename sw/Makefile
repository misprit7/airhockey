# Air Hockey Robot - Motor Control Software

# sFoundation paths
SF_ROOT := third_party/sFoundation
SF_LIB_DIR := $(SF_ROOT)/sFoundation
SF_INC := $(SF_ROOT)/inc/inc-pub
SF_SO := $(SF_LIB_DIR)/libsFoundation20.so

CXX := g++
CXXFLAGS := -std=c++11 -O2 -Wall -I$(SF_INC)
LDFLAGS := -L$(SF_LIB_DIR) -Wl,-rpath=$(CURDIR)/$(SF_LIB_DIR) -lsFoundation20 -lpthread

BIN_DIR := bin

.PHONY: all clean sfoundation

all: sfoundation $(BIN_DIR)/test_motor $(BIN_DIR)/scan_motors

sfoundation: $(SF_SO)

$(SF_SO):
	$(MAKE) -C $(SF_LIB_DIR) -j$$(nproc)

$(BIN_DIR)/%: $(BIN_DIR)/%.cpp $(SF_SO)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BIN_DIR)/test_motor
	$(MAKE) -C $(SF_LIB_DIR) clean
