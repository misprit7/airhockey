#include "serial_lines.h"
#include <cassert>
#include <string>
#include <vector>

int main() {
    const std::string stream = "OK ACCEL\r\nS 1 2 3 4 5 6 7 8\r\nOK CMD\r\nS 9 8 7 6 5 4 3 2\r\n";
    const std::vector<std::string> expected = {"OK ACCEL", "S 1 2 3 4 5 6 7 8", "OK CMD", "S 9 8 7 6 5 4 3 2"};
    // Every possible read split, including an OK followed by a partial S.
    for (std::size_t split = 0; split <= stream.size(); ++split) {
        SerialLines lines;
        std::vector<std::string> got;
        auto collect = [&](const char *line) { got.emplace_back(line); };
        lines.feed(stream.data(), split, collect);
        lines.feed(stream.data() + split, stream.size() - split, collect);
        assert(got == expected);
    }
    SerialLines lines;
    std::vector<std::string> got;
    for (char c : stream) lines.feed(&c, 1, [&](const char *line) { got.emplace_back(line); });
    assert(got == expected);
    // A corrupt oversized line must recover at its newline, not its middle.
    std::string corrupt(5000, 'x'); corrupt += "\nS recovered\n";
    got.clear();
    lines.feed(corrupt.data(), corrupt.size(), [&](const char *line) { got.emplace_back(line); });
    assert(got == std::vector<std::string>{"S recovered"});
}
