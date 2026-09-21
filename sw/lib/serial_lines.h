#pragma once
#include <cstddef>
#include <string>

// One persistent framer per serial connection. Command acknowledgements and
// unsolicited status share the same byte stream: returning at an OK must
// never discard a trailing status or a partial line from that same read.
class SerialLines {
public:
    template<class Consumer>
    void feed(const char *bytes, std::size_t length, Consumer consume) {
        for (std::size_t i = 0; i < length; ++i) {
            const char c = bytes[i];
            if (c == '\n') {
                if (!dropping_ && !pending_.empty()) consume(pending_.c_str());
                pending_.clear();
                dropping_ = false;
            } else if (!dropping_) {
                if (pending_.size() >= 4096) {
                    pending_.clear();
                    dropping_ = true;
                } else if (c != '\r') {
                    pending_ += c;
                }
            }
        }
    }
private:
    std::string pending_;
    bool dropping_ = false;
};
