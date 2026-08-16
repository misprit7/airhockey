# Host tests for the parts of the firmware that are pure math.
#
# Deliberately separate from the PlatformIO build: this compiles for the
# machine you are sitting at, with no Arduino core, so the motion law can be
# exercised a few million ticks at a time before it ever reaches a motor.

CXX      ?= g++
CXXFLAGS ?= -std=c++11 -O2 -Wall -Wextra -I../include -I../../shared
BUILD    := build

.PHONY: all test clean
all: test

# Depend on the headers too. Without this the test silently keeps validating
# against a stale copy of the geometry: change the workspace bounds, rerun,
# and it cheerfully reports "All checks passed" for the workspace you no
# longer have.
HEADERS := ../include/motion_profile.h ../../shared/cdpr_geometry.h

$(BUILD)/test_motion_profile: test_motion_profile.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

$(BUILD):
	mkdir -p $(BUILD)

test: $(BUILD)/test_motion_profile
	./$(BUILD)/test_motion_profile

clean:
	rm -rf $(BUILD)
