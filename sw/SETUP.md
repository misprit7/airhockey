# SC4-Hub Linux Setup Guide

## Overview

The Teknic SC4-Hub uses an Exar USB-to-serial chip internally. Linux's generic `cdc_acm` driver auto-claims it and creates `/dev/ttyACM0`, but `cdc_acm` cannot properly control the Exar chip's internal registers — baud rate changes and break signals don't actually reach the hardware. The sFoundation library relies on serial BREAK signals for motor discovery, so communication silently fails with "no nodes found."

The fix is to use the Exar-specific kernel driver (`xr_usb_serial_common`), which knows how to configure the chip via vendor-specific USB control transfers.

## Issues Encountered (Arch Linux, kernel 6.19)

### 1. Kernel/module version mismatch
**Problem:** Running kernel didn't match installed modules (6.18 vs 6.19), so `cdc_acm` wasn't even available as a module.
**Fix:** Reboot into the current kernel.

### 2. `cdc_acm` binds but can't communicate
**Problem:** `cdc_acm` creates `/dev/ttyACM0` and the port opens, but sFoundation's init fails with `MN_ERR_INIT_FAILED_PORTS_0` ("Port opened OK but no nodes found"). The BREAK signals used for node discovery never reach the motors because `cdc_acm` can't set the Exar chip's baud rate or transmit breaks correctly.
**Fix:** Use the Exar kernel driver instead.

### 3. In-tree `xr_serial` driver doesn't work
**Problem:** The kernel's built-in `xr_serial` module only recognizes Exar's VID (0x04E2), not Teknic's (0x2890). Adding the Teknic VID/PID via `new_id` creates `/dev/ttyUSB0` but the driver doesn't know which chip variant it is, so register configuration fails.
**Fix:** Use Teknic's out-of-tree Exar driver which has the Teknic VID/PID hardcoded.

### 4. Teknic's Exar driver doesn't compile on kernel 6.x
**Problem:** The driver (shipped ~2021, tested up to kernel 5.14) has multiple incompatibilities with kernel 6.19:
- `asm/unaligned.h` moved to `linux/unaligned.h`
- GPIO API (`gpiochip_add`/`gpiochip_remove`, `struct gpio_chip`) changed significantly
- `tty_operations.write` signature changed from `int(…, const unsigned char*, int)` to `ssize_t(…, const u8*, size_t)`
- `tty_operations.set_termios` parameter changed to `const struct ktermios*`
- `struct usb_cdc_country_functional_desc` field typo `wCountyCode0` fixed to `wCountryCode0`

**Fix:** Patched driver source in `third_party/ExarKernelDriver/`. Key changes:
- Header already had `linux/unaligned.h` (previously patched)
- Disabled GPIO support entirely (`#undef CONFIG_GPIOLIB`, removed struct field) — not needed for serial communication
- Updated `xr_usb_serial_tty_write` return type to `ssize_t` and params to `const u8*, size_t`
- Updated `xr_usb_serial_tty_set_termios` to take `const struct ktermios*`
- Fixed `wCountyCode0` → `wCountryCode0`

### 5. `cdc_acm` races the Exar driver on hotplug
**Problem:** Even with the Exar driver loaded, `cdc_acm` auto-binds first because the SC4-Hub advertises standard CDC ACM descriptors.
**Fix:** Blacklist `cdc_acm` via modprobe config. The blacklist prevents auto-loading; the Exar driver then claims the device.

### 6. Missing end-of-loop jumper on SC4-Hub
**Problem:** Even with the correct driver, no nodes were found. The SC4-Hub requires a jumper on the header corresponding to the last populated motor port (J0 for 1 motor, J1 for 2, etc.).
**Fix:** Place jumper on J3 (for all 4 ports populated). Motors must be connected starting from CP0 with no gaps.

## Fresh Machine Setup

### Prerequisites
- Linux with kernel headers installed (`linux-headers` on Arch)
- `g++`, `make`
- User in the `uucp` group (Arch) or `dialout` group (Debian/Ubuntu) for serial port access

### Step 1: Add user to serial port group

```bash
# Arch Linux
sudo usermod -aG uucp $USER

# Debian/Ubuntu
sudo usermod -aG dialout $USER

# Log out and back in for group change to take effect
```

### Step 2: Build the Exar kernel driver

```bash
cd sw/third_party/ExarKernelDriver
make
```

If the build fails, you likely need kernel headers:
```bash
# Arch
sudo pacman -S linux-headers

# Debian/Ubuntu
sudo apt install linux-headers-$(uname -r)
```

### Step 3: Install the kernel driver

```bash
sudo mkdir -p /lib/modules/$(uname -r)/extra
sudo cp sw/third_party/ExarKernelDriver/xr_usb_serial_common.ko /lib/modules/$(uname -r)/extra/
sudo depmod
```

### Step 4: Blacklist `cdc_acm` and install udev rules

```bash
echo "blacklist cdc_acm" | sudo tee /etc/modprobe.d/sc4hub-blacklist.conf
sudo cp sw/99-sc4hub.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

### Step 5: Load the driver

```bash
# Unload cdc_acm if currently loaded
sudo modprobe -r cdc_acm

# Load Exar driver
sudo modprobe xr_usb_serial_common
```

### Step 6: Plug in the SC4-Hub

The device should appear as `/dev/ttyXRUSB0`. Verify:
```bash
ls /dev/ttyXRUSB*
```

### Step 7: Build and test

```bash
cd sw
bash setup.sh   # fetch and build sFoundation SDK (first time only)
make
bin/scan_motors  # should list all connected motors
```

### Step 8: Hardware checklist

- SC4-Hub powered with 15-30V DC
- Motors powered with 24-75V DC
- Motors connected starting from CP0, no gaps
- End-of-loop jumper on the correct header:
  - 1 motor: J0
  - 2 motors: J1
  - 3 motors: J2
  - 4 motors: J3
- Motor status LED solid green = powered, disabled (normal)

## After Kernel Updates

The `.ko` module must be rebuilt for each new kernel version:

```bash
cd sw/third_party/ExarKernelDriver
make clean && make
sudo cp xr_usb_serial_common.ko /lib/modules/$(uname -r)/extra/
sudo depmod
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No `/dev/ttyXRUSB*` | `cdc_acm` claimed device | `sudo modprobe -r cdc_acm`, re-plug USB |
| `/dev/ttyACM0` appears | Exar driver not loaded | `sudo modprobe xr_usb_serial_common` |
| "No nodes found" | Missing jumper or motor not connected | Check jumper position and motor cables |
| "No nodes found" (driver ok) | Wrong driver (cdc_acm) | Check `readlink /sys/bus/usb/devices/1-5.1:1.0/driver` — should say `cdc_xr_usb_serial` |
| Permission denied | Not in uucp/dialout group | `sudo usermod -aG uucp $USER`, log out/in |
| Build fails | Missing kernel headers | Install `linux-headers` package |
| Build fails after kernel update | Stale `.ko` | Rebuild: `cd ExarKernelDriver && make clean && make` |
