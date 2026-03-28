---
name: sFoundation serial initialization sequence
description: Detailed analysis of sFoundation serial port init and node discovery protocol - critical for CDC ACM vs Exar driver compatibility
type: reference
---

## Serial Port Configuration

**Initial Parameters (SerialLinux.cpp:219):**
- Baud: 9600 (hardcoded, MN_NET_BAUD_RATE = MN_BAUD_1X)
- Data: 8 bits
- Parity: None
- Stop: 1 bit
- Flow control: None (no CTS/RTS or XON/XOFF)
- DTR/RTS: Both cleared after 100ms

**Setup Flags:**
- BRKINT enabled in c_iflag (break generates SIGINT)
- VMIN=1, VTIME=0 (non-blocking read, wait for 1 byte)

## Node Discovery Handshake (Critical Path)

```
Port Open @ 9600 → Setup() → Break-based Discovery → Node Addressing
```

**Key Call Stack:**
- mnInitializeSystem() → mnInitializeProc() → infcStartController() → infcResetNetRate()

**infcResetNetRate() sequence (lnkAccessCommon.cpp:2774):**
1. Set port to 9600, 8N1
2. **Loop 5 times:** Send 40ms BREAK signal via tcsendbreak()
3. Wait 48ms (1.2 × COMM_EVT_BRK_DLY_MS)
4. Check ErrorReportGet() for BREAKcnt counter
5. If BREAKcnt == 0, continue loop (node didn't echo break)
6. If BREAKcnt > 0, success - set SERIALMODE_MERIDIAN_7BIT_PACKET mode and exit

**Node Addressing (netSetAddress in netCmdAPI.cpp:1425):**
- Send MN_PKT_TYPE_SET_ADDR packet
- Retry 3x if no response
- Extract node count from response

## Critical Compatibility Issue: Break Signal Handling

**Why this matters for CDC ACM vs Exar:**

The sFoundation driver REQUIRES proper tcsendbreak() support and break detection error reporting:
- Sends serial BREAK to reset nodes to defaults
- **Depends on driver reporting break was received** via ErrorReportGet()
- CDC ACM may not properly detect/report breaks
- Exar driver likely has better break signal implementation

**Specific Failure Signature:**
- infcResetNetRate loops 5x waiting for BREAKcnt > 0
- Returns MN_ERR_NO_NET_CONNECTIVITY at line 2857
- Initialization timeout

## Debug Points

If CDC ACM fails:
1. Check if tcsendbreak() succeeds (no ioctl errors)
2. Check if ErrorReportGet() returns BREAKcnt > 0
3. Use strace to monitor break signal generation
4. Packet sniffer to see if nodes respond to breaks

## Supported Baud Rates

Standard rates only: 9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600, 1036800

No custom baud rates used - shouldn't be a CDC ACM compatibility issue.

## Key Files

- `/sw/third_party/sFoundation/sFoundation/src-linux/SerialLinux.cpp` - Port open/setup
- `/sw/third_party/sFoundation/sFoundation/src/lnkAccessCommon.cpp` - Controller start/reset
- `/sw/third_party/sFoundation/sFoundation/src/netCmdAPI.cpp` - Node discovery
- `/sw/third_party/sFoundation/sFoundation/src/meridianNet.cpp` - Initialization flow
- `/sw/third_party/sFoundation/inc/inc-pub/pubMnNetDef.h` - Constants
