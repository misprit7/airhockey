// Quick test: does tcsendbreak() work on /dev/ttyACM0?
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <errno.h>
#include <string.h>

int main(int argc, char *argv[]) {
    const char *port = argc > 1 ? argv[1] : "/dev/ttyACM0";

    printf("Opening %s...\n", port);
    int fd = open(port, O_RDWR | O_NOCTTY);
    if (fd < 0) {
        printf("ERROR: open failed: %s\n", strerror(errno));
        return 1;
    }
    printf("Opened fd=%d\n", fd);

    // Configure 9600 8N1 like sFoundation does
    struct termios tio;
    tcgetattr(fd, &tio);
    cfmakeraw(&tio);
    cfsetispeed(&tio, B9600);
    cfsetospeed(&tio, B9600);
    tio.c_cflag &= ~(PARENB | CSTOPB);
    tio.c_cflag |= CS8;
    tio.c_iflag |= BRKINT;  // sFoundation enables this
    tcsetattr(fd, TCSANOW, &tio);
    tcflush(fd, TCIOFLUSH);

    // Clear DTR and RTS like sFoundation does
    int modem_bits;
    ioctl(fd, TIOCMGET, &modem_bits);
    printf("Modem bits before: 0x%04x (DTR=%d RTS=%d)\n",
           modem_bits, !!(modem_bits & TIOCM_DTR), !!(modem_bits & TIOCM_RTS));
    modem_bits &= ~(TIOCM_DTR | TIOCM_RTS);
    ioctl(fd, TIOCMSET, &modem_bits);

    // Send 5 breaks like sFoundation does
    for (int i = 0; i < 5; i++) {
        printf("Sending break %d... ", i + 1);
        int ret = tcsendbreak(fd, 40);  // 40ms break
        printf("ret=%d (errno=%s)\n", ret, ret ? strerror(errno) : "none");
        usleep(48000);  // 48ms between breaks
    }

    // Try to read any response
    printf("\nWaiting for response (1 second)...\n");
    struct timeval tv = {1, 0};
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);
    int sel = select(fd + 1, &rfds, NULL, NULL, &tv);
    if (sel > 0) {
        unsigned char buf[256];
        int n = read(fd, buf, sizeof(buf));
        printf("Got %d bytes:", n);
        for (int i = 0; i < n; i++) printf(" %02x", buf[i]);
        printf("\n");
    } else {
        printf("No response (timeout)\n");
    }

    close(fd);
    return 0;
}
