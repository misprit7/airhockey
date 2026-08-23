# Host build of the firmware's motion profile as a shared library.
#
# Separate from fw/test (which builds an executable that asserts) and from the
# PlatformIO build (which targets the Teensy). This target exists so the
# SIMULATOR can call the real control law instead of carrying a second copy of
# it in NumPy.
#
# -O3 and -ffast-math: this is the sim's inner loop, run millions of times per
# training run, and it is not the code that drives the machine -- the Teensy
# build sets its own flags. -ffast-math is acceptable HERE for that reason,
# and the divergence test in ai/tests is what bounds the consequence.

CXX      ?= g++
CXXFLAGS ?= -std=c++11 -O3 -ffast-math -fPIC -Wall -Wextra -I../include -I../../shared
BUILD    := build
LIB      := $(BUILD)/libmotion.so

.PHONY: all clean
all: $(LIB)

$(LIB): motion_batch.cpp ../include/motion_profile.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -shared -o $@ $< -lm

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
