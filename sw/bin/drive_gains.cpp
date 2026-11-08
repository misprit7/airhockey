// Read every ClearPath's tuning and save it, before anything changes it.
//
//   sw/build/drive_gains <new-output-dir>
//
// READ-ONLY: never enables a drive and never writes a parameter. Writes two
// kinds of file into a directory it creates itself, and refuses a directory
// that already exists so a backup can never be overwritten:
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
#include <cstdio>
#include <cstring>
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

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2 || argv[1][0] == '-') {
    fprintf(stderr, "usage: drive_gains <new-output-dir>\n");
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
