#include "Spinnaker.h"
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

using namespace Spinnaker;

// ============================================================================
// snap — capture still frames from the FLIR Blackfly S (Spinnaker SDK).
//
// Usage: snap [outdir] [count] [interval_s] [--exposure us|auto|keep]
//                                           [--gain db|auto|keep]
//   outdir      output directory (default "shots")
//   count       number of frames (default 20)
//   interval_s  seconds between frames (default 2.0);
//               0 = wait for Enter before each frame
//   --exposure  microseconds, "auto" (default), or "keep" (don't touch)
//   --gain      dB, "auto" (default), or "keep"
//   --focus     focus-assist mode: no files, just a live sharpness score
//               (gradient energy, center half of frame). Lay a high-contrast
//               target flat on the table, turn the focus ring to MAXIMIZE
//               the score, lock the ring, verify the score holds. Ctrl+C ends.
//   --stream    serve frames over stdout for a Python consumer (there is no
//               PySpin here, so the camera must be driven from C++ while
//               detection happens in Python). Request/response rather than
//               free-running, so the consumer never falls behind and never
//               sees a stale frame:
//                 stdout: "SNAPSTRM" + u32 width + u32 height (little endian),
//                         then per request: u8 status (1 ok, 0 failed)
//                         followed, if ok, by width*height Mono8 bytes
//                 stdin:  'g' = send the newest frame, 'q' = quit
//               In this mode every human-readable message goes to stderr so
//               it cannot corrupt the binary stream.
//
// Frames are saved as Mono8 PNGs (shot_000.png ...) suitable for
// vision/bin/calibrate_intrinsics.py --images 'outdir/*.png'.
// Each save line reports the exposure/gain actually used for that frame.
// ============================================================================

static volatile sig_atomic_t g_stop = 0;
static void onStop(int) { g_stop = 1; }

// Normalized gradient energy over the center half of a Mono8 frame.
// Brightness-normalized so auto-exposure drift doesn't masquerade as focus.
static double sharpness(const unsigned char *p, size_t w, size_t h,
                        size_t stride) {
    size_t x0 = w / 4, x1 = 3 * w / 4, y0 = h / 4, y1 = 3 * h / 4;
    double g = 0, mean = 0;
    size_t n = 0;
    for (size_t y = y0; y < y1; y += 2) {
        const unsigned char *row = p + y * stride;
        for (size_t x = x0; x < x1; x += 2) {
            double dx = (double)row[x + 1] - row[x - 1];
            double dy = (double)p[(y + 1) * stride + x] - p[(y - 1) * stride + x];
            g += dx * dx + dy * dy;
            mean += row[x];
            n++;
        }
    }
    mean /= n;
    return g / n / (mean * mean + 1e-9) * 1e4;
}

int main(int argc, char **argv) {
    std::string outDir = "shots";
    int count = 20;
    double interval = 2.0;
    std::string expArg = "auto", gainArg = "auto";
    bool focusMode = false, streamMode = false;
    int pos = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--exposure" && i + 1 < argc) {
            expArg = argv[++i];
        } else if (a == "--gain" && i + 1 < argc) {
            gainArg = argv[++i];
        } else if (a == "--focus") {
            focusMode = true;
        } else if (a == "--stream") {
            streamMode = true;
        } else if (pos == 0) {
            outDir = a; pos++;
        } else if (pos == 1) {
            count = atoi(a.c_str()); pos++;
        } else {
            interval = atof(a.c_str()); pos++;
        }
    }

    // In --stream mode stdout carries binary frames, so chatter goes to stderr.
    FILE *msg = streamMode ? stderr : stdout;

    if (!streamMode) mkdir(outDir.c_str(), 0755);

    SystemPtr system = System::GetInstance();
    CameraList cams = system->GetCameras();
    if (cams.GetSize() == 0) {
        fprintf(msg, "ERROR: no Spinnaker cameras found (check USB + udev rules)\n");
        cams.Clear();
        system->ReleaseInstance();
        return 1;
    }
    CameraPtr cam = cams.GetByIndex(0);
    int ret = 0;
    try {
        fprintf(msg, "camera: %s (SN %s)\n",
                cam->TLDevice.DeviceModelName.ToString().c_str(),
                cam->TLDevice.DeviceSerialNumber.ToString().c_str());
        cam->Init();

        // Always hand us the freshest frame — we capture slowly and don't
        // want seconds-stale buffered images.
        GenApi::INodeMap &sMap = cam->GetTLStreamNodeMap();
        GenApi::CEnumerationPtr handling = sMap.GetNode("StreamBufferHandlingMode");
        if (GenApi::IsWritable(handling))
            handling->FromString("NewestOnly");

        // Exposure / gain configuration.
        if (expArg == "auto") {
            cam->ExposureAuto.SetValue(ExposureAuto_Continuous);
            fprintf(msg, "exposure: auto\n");
        } else if (expArg != "keep") {
            double us = atof(expArg.c_str());
            cam->ExposureAuto.SetValue(ExposureAuto_Off);
            double lo = cam->ExposureTime.GetMin();
            double hi = cam->ExposureTime.GetMax();
            us = us < lo ? lo : (us > hi ? hi : us);
            cam->ExposureTime.SetValue(us);
            fprintf(msg, "exposure: %.0f us (valid %.0f-%.0f)\n", us, lo, hi);
        } else {
            fprintf(msg, "exposure: unchanged\n");
        }
        if (gainArg == "auto") {
            cam->GainAuto.SetValue(GainAuto_Continuous);
            fprintf(msg, "gain: auto\n");
        } else if (gainArg != "keep") {
            double db = atof(gainArg.c_str());
            cam->GainAuto.SetValue(GainAuto_Off);
            double lo = cam->Gain.GetMin();
            double hi = cam->Gain.GetMax();
            db = db < lo ? lo : (db > hi ? hi : db);
            cam->Gain.SetValue(db);
            fprintf(msg, "gain: %.1f dB (valid %.1f-%.1f)\n", db, lo, hi);
        } else {
            fprintf(msg, "gain: unchanged\n");
        }

        cam->AcquisitionMode.SetValue(AcquisitionMode_Continuous);
        cam->BeginAcquisition();

        ImageProcessor proc;

        if (streamMode) {
            struct sigaction sa = {};
            sa.sa_handler = onStop;
            sigaction(SIGINT, &sa, NULL);
            sigaction(SIGTERM, &sa, NULL);

            // Prime once to learn the frame geometry before announcing it.
            size_t W = 0, H = 0;
            for (int t = 0; t < 20 && W == 0; t++) {
                try {
                    ImagePtr img = cam->GetNextImage(2000);
                    if (!img->IsIncomplete()) {
                        ImagePtr m = proc.Convert(img, PixelFormat_Mono8);
                        W = m->GetWidth();
                        H = m->GetHeight();
                    }
                    img->Release();
                } catch (Spinnaker::Exception &) {
                }
            }
            if (W == 0) {
                fprintf(msg, "ERROR: no complete frame from the camera\n");
                cam->EndAcquisition();
                cam->DeInit();
                cam = nullptr;
                cams.Clear();
                system->ReleaseInstance();
                return 1;
            }

            unsigned char hdr[16];
            memcpy(hdr, "SNAPSTRM", 8);
            uint32_t w32 = (uint32_t)W, h32 = (uint32_t)H;
            memcpy(hdr + 8, &w32, 4);
            memcpy(hdr + 12, &h32, 4);
            fwrite(hdr, 1, sizeof(hdr), stdout);
            fflush(stdout);
            fprintf(msg, "stream: %zux%zu Mono8 ready\n", W, H);

            std::vector<unsigned char> buf(W * H);
            while (!g_stop) {
                int c = getchar();
                if (c == EOF || c == 'q') break;
                if (c != 'g') continue;
                bool ok = false;
                try {
                    ImagePtr img = cam->GetNextImage(2000);
                    if (!img->IsIncomplete()) {
                        ImagePtr m = proc.Convert(img, PixelFormat_Mono8);
                        const unsigned char *src =
                            (const unsigned char *)m->GetData();
                        size_t stride = m->GetStride();
                        // Copy row by row: stride may exceed the row width.
                        for (size_t y = 0; y < H; y++)
                            memcpy(&buf[y * W], src + y * stride, W);
                        ok = true;
                    }
                    img->Release();
                } catch (Spinnaker::Exception &) {
                    ok = false;
                }
                unsigned char st = ok ? 1 : 0;
                fwrite(&st, 1, 1, stdout);
                if (ok) fwrite(buf.data(), 1, buf.size(), stdout);
                fflush(stdout);
            }
            cam->EndAcquisition();
            cam->DeInit();
            cam = nullptr;
            cams.Clear();
            system->ReleaseInstance();
            return 0;
        }

        if (focusMode) {
            struct sigaction sa = {};
            sa.sa_handler = onStop;
            sigaction(SIGINT, &sa, NULL);
            sigaction(SIGTERM, &sa, NULL);
            printf("FOCUS ASSIST — lay a high-contrast target flat on the "
                   "table.\nTurn the focus ring to MAXIMIZE the score; lock "
                   "the ring; verify the score holds. Ctrl+C to exit.\n\n");
            double best = 0;
            auto lastPrint = std::chrono::steady_clock::now();
            while (!g_stop) {
                ImagePtr img;
                try {
                    img = cam->GetNextImage(2000);
                } catch (Spinnaker::Exception &) {
                    // Ctrl+C interrupts the buffer wait mid-call; anything
                    // else transient just retries.
                    continue;
                }
                if (img->IsIncomplete()) {
                    img->Release();
                    continue;
                }
                ImagePtr mono = proc.Convert(img, PixelFormat_Mono8);
                double s = sharpness((const unsigned char *)mono->GetData(),
                                     mono->GetWidth(), mono->GetHeight(),
                                     mono->GetStride());
                img->Release();
                if (s > best) best = s;
                auto now = std::chrono::steady_clock::now();
                if (now - lastPrint > std::chrono::milliseconds(150)) {
                    lastPrint = now;
                    int bars = (int)(40.0 * s / (best + 1e-9));
                    printf("\rsharpness %7.2f  best %7.2f  [%-40.*s]",
                           s, best, bars,
                           "||||||||||||||||||||||||||||||||||||||||");
                    fflush(stdout);
                }
            }
            printf("\n");
            cam->EndAcquisition();
            cam->DeInit();
            cam = nullptr;
            cams.Clear();
            system->ReleaseInstance();
            return 0;
        }

        for (int i = 0; i < count; i++) {
            if (interval <= 0) {
                printf("shot %d/%d — position the board, press Enter...",
                       i + 1, count);
                fflush(stdout);
                int c;
                do { c = getchar(); } while (c != '\n' && c != EOF);
                if (c == EOF) break;
            } else {
                for (int t = (int)interval; t > 0; t--) {
                    printf("\rshot %d/%d in %d...   ", i + 1, count, t);
                    fflush(stdout);
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
            ImagePtr img = cam->GetNextImage(2000);
            if (img->IsIncomplete()) {
                printf("\rincomplete image, retrying\n");
                img->Release();
                i--;
                continue;
            }
            char name[512];
            snprintf(name, sizeof(name), "%s/shot_%03d.png", outDir.c_str(), i);
            ImagePtr mono = proc.Convert(img, PixelFormat_Mono8);
            mono->Save(name);
            printf("\rsaved %s (%zux%zu, %.0f us, %.1f dB)          \n",
                   name, img->GetWidth(), img->GetHeight(),
                   cam->ExposureTime.GetValue(), cam->Gain.GetValue());
            img->Release();
        }
        cam->EndAcquisition();
        cam->DeInit();
    } catch (Spinnaker::Exception &e) {
        printf("ERROR: %s\n", e.what());
        ret = 1;
    }
    cam = nullptr;
    cams.Clear();
    system->ReleaseInstance();
    return ret;
}
