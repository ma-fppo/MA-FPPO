"""Process-wide SC2 port leases outside Linux's outgoing ephemeral range.

Used only by explicitly protected entry points. No global machine settings or
installed packages are modified, and ports already bound by any process are skipped.
"""
import atexit
import errno
import fcntl
import os
from pathlib import Path
import socket
import threading


class PortLeases:
    def __init__(self, directory=None, first=10000, last=30000):
        self.directory = Path(directory or '/tmp/macflow_ppo_sc2_ports_%d' % os.getuid())
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.stat().st_uid != os.getuid():
            raise PermissionError('Port lease directory is owned by another user')
        ephemeral = Path('/proc/sys/net/ipv4/ip_local_port_range')
        if ephemeral.exists():
            lo, hi = map(int, ephemeral.read_text().split())
            if first <= hi and last >= lo:
                raise ValueError('SC2 lease range overlaps outgoing ephemeral ports')
        self.first, self.last = first, last
        self.held = {}
        self.mutex = threading.Lock()

    def pick(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError('SC2 lease allocator expects no portpicker arguments')
        with self.mutex:
            for port in range(self.first, self.last + 1):
                if port in self.held:
                    continue
                fd = os.open(str(self.directory / str(port)), os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                        probe.bind(('127.0.0.1', port))
                except OSError as exc:
                    os.close(fd)
                    if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EADDRINUSE):
                        continue
                    raise
                self.held[port] = fd
                return port
        raise RuntimeError('No free SC2 port lease remains')

    def release(self, port):
        with self.mutex:
            fd = self.held.pop(port, None)
            if fd is None:
                return False
            os.close(fd)  # releases flock; the file must remain to avoid inode races
            return True

    def close(self):
        for port in list(self.held):
            self.release(port)


def install():
    import portpicker
    if getattr(portpicker, '_macflow_sc2_leases', None) is not None:
        return portpicker._macflow_sc2_leases
    leases = PortLeases()
    original_return = portpicker.return_port

    def return_port(port):
        if not leases.release(port):
            original_return(port)

    portpicker.pick_unused_port = leases.pick
    portpicker.return_port = return_port
    portpicker._macflow_sc2_leases = leases
    atexit.register(leases.close)
    return leases
