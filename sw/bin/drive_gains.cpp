// Read every ClearPath's tuning and save it; change one gain; put it back.
//
//   sw/build/drive_gains <new-output-dir>
//   sw/build/drive_gains scale <backup-dir> <KV|KP|KI|KFA> <factor> [--apply]
//   sw/build/drive_gains restore <backup-dir> [--apply]
//
// `scale` and `restore` print what they would do and change nothing without
// --apply. New values are always computed from the backup, never from the
// drive's current value, so repeated runs cannot compound. They refuse to
// touch an enabled drive or a drive whose serial is not the one backed up,
// read every write back, and on any failure put that gain back on all four.
//
// The first form is READ-ONLY: it never enables a drive and never writes a
// parameter. It writes two kinds of file into a directory it creates itself,
// and refuses a directory that already exists so a backup can never be
// overwritten:
//
//   gains.json   every tuning parameter per drive at full precision -- the
//                direct way back, one parameter at a time
//   node<N>.mtr  the drive's whole configuration via ConfigSave, loadable in
//                ClearView -- the last resort
//
// The SDK marks these parameters non-volatile (PT_CFG_T in cpmAPI.cpp), so a
// write to one is still there after a power cycle. Run this first.
//
// Needs the SC-Hub port to itself: stop cdpr_master and activate.

#include "clearpath.h"
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <string>
#include <sys/stat.h>

namespace {

struct Param { const char *name; int index; };

const Param PARAMS[] = {
    // The compensator: what a tuning sweep would change.
    {"KV", CPM_P_DRV_KV}, {"KP", CPM_P_DRV_KP}, {"KI", CPM_P_DRV_KI},
    {"KFV", CPM_P_DRV_KFV}, {"KFA", CPM_P_DRV_KFA}, {"KFJ", CPM_P_DRV_KFJ},
    {"KNV", CPM_P_DRV_KNV}, {"KR", CPM_P_DRV_KR},
    {"KZERO", CPM_P_DRV_KZERO}, {"IZERO", CPM_P_DRV_IZERO},
    {"KP_FACTOR", CPM_P_DRV_KP_FACTOR}, {"KPZ_FACTOR", CPM_P_DRV_KPZ_FACTOR},
    {"KI_FACTOR", CPM_P_DRV_KI_FACTOR}, {"FINE_TUNE", CPM_P_DRV_FINE_TUNE},
    {"TUNE_CONFIG_REG", CPM_P_DRV_TUNE_CONFIG_REG},
    // Current loop and the rest of the tuning group.
    {"KIP", CPM_P_DRV_KIP}, {"KII", CPM_P_DRV_KII},
    {"KIP_ADJ", CPM_P_DRV_KIP_ADJ}, {"KII_ADJ", CPM_P_DRV_KII_ADJ},
    {"TRQ_BIAS", CPM_P_DRV_TRQ_BIAS},
    {"AH_FILT_TC", CPM_P_DRV_AH_FILT_TC}, {"AH_HOLDOFF", CPM_P_AH_HOLDOFF},
    {"FUZZY_APERTURE", CPM_P_DRV_FUZZY_APERTURE},
    {"FUZZY_HYST", CPM_P_DRV_FUZZY_HYST},
    {"TARGET_WIN", CPM_P_DRV_TARGET_WIN}, {"STAB_WIN", CPM_P_DRV_STAB_WIN},
    // Command smoothing. Slot 63 is the RAS control register on these drives
    // (cpmAPI.cpp) although the header calls it JERK_LIM.
    {"RAS_CONTROL_63", CPM_P_JERK_LIM}, {"RAS_DELAY", CPM_P_APP_RAS_DELAY},
    // Context for reading the rest; not part of the tuning group.
    {"MTR_INERTIA_343", 343}, {"CONFIG_CHANGED", CPM_P_DRV_CONFIG_CHANGED},
    {"TORQUE_LIMIT_A", CPM_P_DRV_TRQ_LIM},
};
const int N_PARAMS = sizeof(PARAMS) / sizeof(PARAMS[0]);

// The only gains this tool will write.
const Param WRITABLE[] = {
    {"KV", CPM_P_DRV_KV}, {"KP", CPM_P_DRV_KP},
    {"KI", CPM_P_DRV_KI}, {"KFA", CPM_P_DRV_KFA},
};
const int N_WRITABLE = sizeof(WRITABLE) / sizeof(WRITABLE[0]);
const double MIN_FACTOR = 0.25, MAX_FACTOR = 2.0;

struct Backup {
  double serial[4];
  std::map<std::string, double> gains[4];
};

bool numberAfter(const std::string &line, const std::string &key, double &out) {
  const std::size_t at = line.find("\"" + key + "\": ");
  if (at == std::string::npos) return false;
  out = strtod(line.c_str() + at + key.size() + 4, nullptr);
  return true;
}

bool loadBackup(const std::string &dir, Backup &b) {
  std::ifstream in(dir + "/gains.json");
  std::string line;
  bool seen[4] = {false, false, false, false};
  while (std::getline(in, line)) {
    double node_d, serial;
    if (!numberAfter(line, "node", node_d) || !numberAfter(line, "serial", serial)) continue;
    const int node = static_cast<int>(node_d);
    if (node < 0 || node > 3) return false;
    b.serial[node] = serial;
    for (const Param &p : WRITABLE) {
      double v;
      if (!numberAfter(line, p.name, v)) return false;
      b.gains[node][p.name] = v;
    }
    seen[node] = true;
  }
  return seen[0] && seen[1] && seen[2] && seen[3];
}

const Param *writable(const std::string &name) {
  for (const Param &p : WRITABLE)
    if (name == p.name) return &p;
  return nullptr;
}

void history(const std::string &text) {
  FILE *f = fopen("logs/drive_gains.log", "a");
  if (!f) return;
  char when[32];
  const time_t now = time(nullptr);
  strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", localtime(&now));
  fprintf(f, "%s %s\n", when, text.c_str());
  fclose(f);
}

// Write one value and confirm the drive holds it.
bool writeVerified(ClearPath &robot, unsigned node, const Param &p, double value) {
  std::string err;
  if (!robot.writeParameter(node, p.index, value, &err)) {
    fprintf(stderr, "node %u %s write failed: %s\n", node, p.name, err.c_str());
    return false;
  }
  double back = 0;
  if (!robot.readParameter(node, p.index, back, &err)) {
    fprintf(stderr, "node %u %s read-back failed: %s\n", node, p.name, err.c_str());
    return false;
  }
  if (std::fabs(back - value) > 0.5) {
    fprintf(stderr, "node %u %s wrote %.0f, drive holds %.0f\n", node, p.name, value, back);
    return false;
  }
  return true;
}

// Every drive must be the one backed up, and must be off.
bool safeToWrite(ClearPath &robot, const Backup &b) {
  bool ok = true;
  for (unsigned node = 0; node < 4; node++) {
    const LoadValue serial = robot.readLoad(node, LoadField::Serial);
    const LoadValue status = robot.readLoad(node, LoadField::Status);
    if (!serial.valid || serial.value != b.serial[node]) {
      fprintf(stderr, "node %u serial %.0f is not the backed-up %.0f\n", node, serial.value,
              b.serial[node]);
      ok = false;
    }
    if (!status.valid || status.bits[0] != 0) {
      fprintf(stderr, "node %u is enabled or unreadable; refusing\n", node);
      ok = false;
    }
  }
  return ok;
}

int changeGains(int argc, char **argv) {
  const std::string mode = argv[1];
  const bool scale = mode == "scale";
  const int args = scale ? 5 : 3;
  const bool apply = argc == args + 1 && !strcmp(argv[args], "--apply");
  if ((argc != args && !apply) || (!scale && mode != "restore")) {
    fprintf(stderr, "usage: drive_gains scale <backup-dir> <KV|KP|KI|KFA> <factor> [--apply]\n"
                    "       drive_gains restore <backup-dir> [--apply]\n");
    return 2;
  }
  const std::string dir = argv[2];
  Backup b;
  if (!loadBackup(dir, b)) {
    fprintf(stderr, "cannot read a complete backup from %s/gains.json\n", dir.c_str());
    return 1;
  }
  const Param *only = nullptr;
  double factor = 1.0;
  if (scale) {
    only = writable(argv[3]);
    factor = strtod(argv[4], nullptr);
    if (!only) {
      fprintf(stderr, "%s is not a gain this tool writes (KV, KP, KI, KFA)\n", argv[3]);
      return 2;
    }
    if (!(factor >= MIN_FACTOR && factor <= MAX_FACTOR)) {
      fprintf(stderr, "factor must be %.2f to %.2f\n", MIN_FACTOR, MAX_FACTOR);
      return 2;
    }
  }

  ClearPath robot;
  if (!robot.connect()) {
    fprintf(stderr, "cannot open the SC-Hub -- is cdpr_master or activate still running?\n");
    return 1;
  }
  if (!safeToWrite(robot, b)) return 1;

  bool ok = true;
  printf("%-5s %-5s %12s %12s %12s\n", "node", "gain", "backup", "now", "target");
  for (unsigned node = 0; node < 4; node++) {
    for (const Param &p : WRITABLE) {
      if (only && strcmp(only->name, p.name)) continue;
      double now = 0;
      std::string err;
      if (!robot.readParameter(node, p.index, now, &err)) ok = false;
      const double target = std::round(b.gains[node][p.name] * factor);
      printf("%-5u %-5s %12.0f %12.0f %12.0f\n", node, p.name, b.gains[node][p.name], now, target);
    }
  }
  if (!ok) {
    fprintf(stderr, "could not read the current gains; nothing changed\n");
    return 1;
  }
  if (!apply) {
    printf("dry run: nothing changed. Add --apply to write.\n");
    return 0;
  }

  for (const Param &p : WRITABLE) {
    if (only && strcmp(only->name, p.name)) continue;
    for (unsigned node = 0; node < 4 && ok; node++)
      ok = writeVerified(robot, node, p, std::round(b.gains[node][p.name] * factor));
    if (!ok) {
      fprintf(stderr, "FAILED: putting %s back to the backup on all four drives\n", p.name);
      bool back = true;
      for (unsigned node = 0; node < 4; node++)
        back &= writeVerified(robot, node, p, b.gains[node][p.name]);
      fprintf(stderr, "%s\n", back ? "restored" : "RESTORE FAILED -- do not enable; load the .mtr files");
      history(mode + " FAILED " + p.name + (back ? " restored" : " RESTORE FAILED"));
      return 1;
    }
  }
  const std::string what = scale ? std::string("scale ") + only->name + " x" + argv[4] + " from " + dir
                                 : "restore " + dir;
  history(what);
  printf("applied and read back: %s\n", what.c_str());
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc >= 2 && (!strcmp(argv[1], "scale") || !strcmp(argv[1], "restore")))
    return changeGains(argc, argv);
  if (argc != 2 || argv[1][0] == '-') {
    fprintf(stderr, "usage: drive_gains <new-output-dir>\n"
                    "       drive_gains scale <backup-dir> <KV|KP|KI|KFA> <factor> [--apply]\n"
                    "       drive_gains restore <backup-dir> [--apply]\n");
    return 2;
  }
  const std::string dir = argv[1];
  if (mkdir(dir.c_str(), 0755) != 0) {
    fprintf(stderr, "cannot create %s: %s (a backup is never overwritten; "
            "the parent must exist)\n", dir.c_str(), strerror(errno));
    return 1;
  }

  ClearPath robot;
  if (!robot.connect()) {
    fprintf(stderr, "cannot open the SC-Hub -- is cdpr_master or activate "
            "still running?\n");
    return 1;
  }
  robot.validateHardware();

  double values[4][N_PARAMS];
  bool ok[4][N_PARAMS];
  bool all_ok = true;
  double serial[4] = {0, 0, 0, 0}, firmware[4] = {0, 0, 0, 0};
  bool full_access[4] = {false, false, false, false};

  for (unsigned node = 0; node < 4; node++) {
    serial[node] = robot.readLoad(node, LoadField::Serial).value;
    firmware[node] = robot.readLoad(node, LoadField::Firmware).value;
    full_access[node] = robot.hasFullAccess(node);
    for (int p = 0; p < N_PARAMS; p++) {
      std::string err;
      ok[node][p] = robot.readParameter(node, PARAMS[p].index, values[node][p], &err);
      if (!ok[node][p]) {
        fprintf(stderr, "node %u %s unreadable: %s\n", node, PARAMS[p].name, err.c_str());
        all_ok = false;
      }
    }
    const std::string path = dir + "/node" + std::to_string(node) + ".mtr";
    std::string err;
    if (!robot.saveConfig(node, path.c_str(), &err)) {
      fprintf(stderr, "node %u ConfigSave failed: %s\n", node, err.c_str());
      all_ok = false;
    }
  }

  printf("\n%-16s %16s %16s %16s %16s\n", "parameter", "node 0", "node 1", "node 2", "node 3");
  for (int p = 0; p < N_PARAMS; p++) {
    printf("%-16s", PARAMS[p].name);
    bool same = true;
    for (unsigned node = 0; node < 4; node++) {
      if (ok[node][p]) printf(" %16.6g", values[node][p]);
      else printf(" %16s", "--");
      same &= ok[node][p] && ok[0][p] && values[node][p] == values[0][p];
    }
    printf("%s\n", same ? "" : "  <- differs");
  }
  printf("\nfull access:");
  for (unsigned node = 0; node < 4; node++) printf(" %d", full_access[node]);
  printf("\n");

  const std::string json = dir + "/gains.json";
  FILE *f = fopen(json.c_str(), "w");
  if (!f) {
    fprintf(stderr, "cannot write %s: %s\n", json.c_str(), strerror(errno));
    return 1;
  }
  fprintf(f, "{\n  \"nodes\": [\n");
  for (unsigned node = 0; node < 4; node++) {
    fprintf(f, "    {\"node\": %u, \"serial\": %.0f, \"firmware\": %.0f, \"params\": {", node,
            serial[node], firmware[node]);
    bool first = true;
    for (int p = 0; p < N_PARAMS; p++) {
      if (!ok[node][p]) continue;
      fprintf(f, "%s\"%s\": %.17g", first ? "" : ", ", PARAMS[p].name, values[node][p]);
      first = false;
    }
    fprintf(f, "}}%s\n", node < 3 ? "," : "");
  }
  fprintf(f, "  ]\n}\n");
  fclose(f);

  robot.disconnect();
  printf("%s %s\n", all_ok ? "saved to" : "INCOMPLETE backup in", dir.c_str());
  return all_ok ? 0 : 1;
}
