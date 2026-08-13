// Free-running blob tracker: streams BLOB COORDINATES, not frames.
//
// `snap --stream` hands over whole frames one request at a time, which is
// right for calibration and hopeless for tracking a puck. At 200 Hz a
// 1440x1080 Mono8 frame is 311 MB/s down a pipe, and the request/response
// round trip puts Python in the hot loop. Here the thresholding and
// centroiding happen in C++ on the frame the SDK already handed us, and what
// crosses the pipe is a few dozen bytes per frame.
//
// Deliberately dumb about WHAT it found. It reports every bright blob; which
// one is the puck, which is the paddle and which are the bolted-down markers
// is a question about calibration, and calibration lives in Python. That also
// means this binary does not go stale when the marker layout changes.
//
// Output on stdout, one line per frame, ASCII (a few dozen bytes at 200 Hz is
// nothing, and being able to `head` the stream while debugging is worth more
// than the bytes):
//
//     F <seq> <t_us> <n>  <x> <y> <area>  <x> <y> <area> ...
//
// t_us is the camera's own timestamp where available, so latency in this
// process cannot corrupt a velocity estimate. Blobs are brightest-first.
//
// Usage:
//   vision/build/blobtrack --fps 200 --exposure 300 --threshold 90
//   vision/build/blobtrack --probe        # report achievable rate and exit

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <vector>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;

static volatile sig_atomic_t g_stop = 0;
static void onStop(int) { g_stop = 1; }

struct Blob {
    double x, y;      // intensity-weighted centre, pixels
    int area;         // pixels above threshold
    long sum;         // total (I - threshold), used only to rank
};

// One pass of 8-connected labelling over the thresholded image, accumulating
// the intensity-weighted centroid of each region as we go.
//
// Flood fill with an explicit stack rather than recursion: a saturated blob
// on a bright frame can be thousands of pixels and blowing the stack inside
// the acquisition loop would be a very confusing failure.
static void findBlobs(const unsigned char *img, int W, int H, int thr,
                      int minArea, int maxArea, size_t maxBlobs,
                      std::vector<Blob> &out, std::vector<int> &label,
                      std::vector<int> &stack) {
    out.clear();
    std::fill(label.begin(), label.end(), 0);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const int i0 = y * W + x;
            if (label[i0] || img[i0] <= thr) continue;
            // new region
            stack.clear();
            stack.push_back(i0);
            label[i0] = 1;
            double sx = 0, sy = 0;
            long sw = 0;
            int area = 0;
            while (!stack.empty()) {
                const int i = stack.back();
                stack.pop_back();
                const int px = i % W, py = i / W;
                const long w = (long)img[i] - thr;
                sx += (double)px * (double)w;
                sy += (double)py * (double)w;
                sw += w;
                area++;
                for (int dy = -1; dy <= 1; dy++) {
                    const int ny = py + dy;
                    if (ny < 0 || ny >= H) continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        const int nx = px + dx;
                        if (nx < 0 || nx >= W) continue;
                        const int j = ny * W + nx;
                        if (label[j] || img[j] <= thr) continue;
                        label[j] = 1;
                        stack.push_back(j);
                    }
                }
            }
            if (area < minArea || area > maxArea || sw <= 0) continue;
            Blob b;
            b.x = sx / (double)sw;
            b.y = sy / (double)sw;
            b.area = area;
            b.sum = sw;
            out.push_back(b);
        }
    }
    std::sort(out.begin(), out.end(),
              [](const Blob &a, const Blob &b) { return a.sum > b.sum; });
    if (out.size() > maxBlobs) out.resize(maxBlobs);
}

int main(int argc, char **argv) {
    double fps = 200.0, exposure = 300.0, gain = 0.0;
    int thr = 90, minArea = 4, maxArea = 4000;
    size_t maxBlobs = 24;
    bool probe = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](double d) { return (i + 1 < argc) ? atof(argv[++i]) : d; };
        if (a == "--fps") fps = next(fps);
        else if (a == "--exposure") exposure = next(exposure);
        else if (a == "--gain") gain = next(gain);
        else if (a == "--threshold") thr = (int)next(thr);
        else if (a == "--min-area") minArea = (int)next(minArea);
        else if (a == "--max-area") maxArea = (int)next(maxArea);
        else if (a == "--max-blobs") maxBlobs = (size_t)next((double)maxBlobs);
        else if (a == "--probe") probe = true;
        else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }

    SystemPtr system = System::GetInstance();
    CameraList cams = system->GetCameras();
    if (cams.GetSize() == 0) {
        fprintf(stderr, "ERROR: no Spinnaker cameras found\n");
        cams.Clear();
        system->ReleaseInstance();
        return 1;
    }
    CameraPtr cam = cams.GetByIndex(0);
    int ret = 0;
    try {
        cam->Init();
        fprintf(stderr, "camera: %s\n",
                cam->TLDevice.DeviceModelName.ToString().c_str());

        // Drop frames rather than queue them. A tracker wants the newest
        // frame; a stale one is worse than none, because it produces a
        // confident velocity estimate for where the puck used to be.
        INodeMap &sMap = cam->GetTLStreamNodeMap();
        CEnumerationPtr handling = sMap.GetNode("StreamBufferHandlingMode");
        if (IsWritable(handling)) handling->FromString("NewestOnly");

        cam->ExposureAuto.SetValue(ExposureAuto_Off);
        double elo = cam->ExposureTime.GetMin(), ehi = cam->ExposureTime.GetMax();
        double us = exposure < elo ? elo : (exposure > ehi ? ehi : exposure);
        cam->ExposureTime.SetValue(us);
        cam->GainAuto.SetValue(GainAuto_Off);
        cam->Gain.SetValue(gain);

        // Exposure caps the frame rate: you cannot run 200 Hz on a 6 ms
        // exposure. Set exposure FIRST, then ask for the rate, then report
        // what the camera actually accepted rather than what we asked for.
        CBooleanPtr en = cam->GetNodeMap().GetNode("AcquisitionFrameRateEnable");
        if (IsWritable(en)) en->SetValue(true);
        CFloatPtr rate = cam->GetNodeMap().GetNode("AcquisitionFrameRate");
        double rlo = 0, rhi = 0, rset = 0;
        if (IsWritable(rate)) {
            rlo = rate->GetMin();
            rhi = rate->GetMax();
            rset = fps < rlo ? rlo : (fps > rhi ? rhi : fps);
            rate->SetValue(rset);
        }
        fprintf(stderr, "exposure %.0f us, gain %.1f dB, rate %.1f Hz "
                        "(camera allows %.1f-%.1f)\n",
                us, gain, rset, rlo, rhi);

        cam->AcquisitionMode.SetValue(AcquisitionMode_Continuous);
        cam->BeginAcquisition();

        struct sigaction sa = {};
        sa.sa_handler = onStop;
        sigaction(SIGINT, &sa, NULL);
        sigaction(SIGTERM, &sa, NULL);

        std::vector<unsigned char> buf;
        std::vector<Blob> blobs;
        std::vector<int> label, stack;
        int W = 0, H = 0;
        unsigned long seq = 0, incomplete = 0;
        uint64_t t0 = 0, tlast = 0;
        double worstMs = 0;

        const unsigned long probeFrames = 400;
        while (!g_stop) {
            ImagePtr img;
            try {
                img = cam->GetNextImage(1000);
            } catch (Spinnaker::Exception &) {
                continue;
            }
            if (img->IsIncomplete()) { incomplete++; img->Release(); continue; }

            if (W == 0) {
                W = (int)img->GetWidth();
                H = (int)img->GetHeight();
                buf.resize((size_t)W * H);
                label.resize((size_t)W * H);
                stack.reserve(8192);
                fprintf(stderr, "frame %dx%d\n", W, H);
                if (!probe) { printf("# %d %d\n", W, H); fflush(stdout); }
            }
            const unsigned char *src = (const unsigned char *)img->GetData();
            const size_t stride = img->GetStride();
            for (int y = 0; y < H; y++)
                memcpy(&buf[(size_t)y * W], src + (size_t)y * stride, W);
            const uint64_t ts = img->GetTimeStamp();
            img->Release();

            const uint64_t tick0 = ts;
            findBlobs(buf.data(), W, H, thr, minArea, maxArea, maxBlobs,
                      blobs, label, stack);

            if (t0 == 0) t0 = ts;
            if (tlast) {
                const double dtms = (double)(ts - tlast) / 1e6;
                if (dtms > worstMs) worstMs = dtms;
            }
            tlast = ts;
            (void)tick0;

            if (!probe) {
                printf("F %lu %llu %zu", seq,
                       (unsigned long long)((ts - t0) / 1000), blobs.size());
                for (size_t i = 0; i < blobs.size(); i++)
                    printf(" %.2f %.2f %d", blobs[i].x, blobs[i].y, blobs[i].area);
                printf("\n");
                fflush(stdout);
            }
            seq++;
            if (probe && seq >= probeFrames) break;
        }

        if (probe) {
            const double secs = (double)(tlast - t0) / 1e9;
            fprintf(stderr,
                    "\nPROBE: %lu frames in %.3f s = %.1f Hz "
                    "(worst gap %.2f ms, %lu incomplete)\n",
                    seq, secs, secs > 0 ? (seq - 1) / secs : 0.0,
                    worstMs, incomplete);
            fprintf(stderr, "last frame: %zu blobs\n", blobs.size());
            for (size_t i = 0; i < blobs.size() && i < 20; i++)
                fprintf(stderr, "   (%7.2f, %7.2f) area %d\n",
                        blobs[i].x, blobs[i].y, blobs[i].area);
        }

        cam->EndAcquisition();
        cam->DeInit();
    } catch (Spinnaker::Exception &e) {
        fprintf(stderr, "Spinnaker error: %s\n", e.what());
        ret = 1;
    }
    cam = nullptr;
    cams.Clear();
    system->ReleaseInstance();
    return ret;
}
