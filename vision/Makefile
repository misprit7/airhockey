# Air Hockey Robot - Vision tools (FLIR Blackfly S via Spinnaker SDK)

SPIN := /opt/spinnaker

CXX := g++
CXXFLAGS := -std=c++14 -O2 -Wall -I$(SPIN)/include
LDFLAGS := -L$(SPIN)/lib -lSpinnaker -Wl,-rpath=$(SPIN)/lib

BUILD_DIR := build

.PHONY: all clean

all: $(BUILD_DIR)/snap

$(BUILD_DIR)/snap: bin/snap.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
