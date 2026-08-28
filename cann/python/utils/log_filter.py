import os
import threading


class StdoutFilter:
    def __init__(self, filter_func):
        self.filter_func = filter_func
        self._orig_out_fd = os.dup(1)
        self._orig_err_fd = os.dup(2)
        self._read_fd, self._write_fd = os.pipe()
        os.dup2(self._write_fd, 1)
        os.dup2(self._write_fd, 2)
        os.close(self._write_fd)

        self._buffer = b""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                data = os.read(self._read_fd, 4096)
                if not data:
                    break
                self._buffer += data
                while b"\n" in self._buffer:
                    line, self._buffer = self._buffer.split(b"\n", 1)
                    decoded = line.decode('utf-8', errors='replace')
                    if not self.filter_func(decoded):
                        os.write(self._orig_out_fd, line + b"\n")
            except OSError:
                break

    def _flush_remaining(self):
        try:
            import fcntl
            fcntl.fcntl(self._read_fd, fcntl.F_SETFL, os.O_NONBLOCK)
        except Exception:
            pass
        try:
            while True:
                data = os.read(self._read_fd, 4096)
                if not data:
                    break
                self._buffer += data
        except OSError:
            pass

        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            decoded = line.decode('utf-8', errors='replace')
            if not self.filter_func(decoded):
                os.write(self._orig_out_fd, line + b"\n")
        if self._buffer:
            decoded = self._buffer.decode('utf-8', errors='replace')
            if not self.filter_func(decoded):
                os.write(self._orig_out_fd, self._buffer)

    def stop(self):
        self._running = False
        try:
            os.dup2(self._orig_out_fd, 1)
            os.dup2(self._orig_err_fd, 2)
        except OSError:
            pass
        self._flush_remaining()
        try:
            os.close(self._read_fd)
        except OSError:
            pass
        try:
            os.close(self._orig_out_fd)
        except OSError:
            pass
        try:
            os.close(self._orig_err_fd)
        except OSError:
            pass


_default_filter = None


def start_filter():
    global _default_filter
    if _default_filter is None:
        _default_filter = StdoutFilter(
            lambda line: line.strip().startswith("[INFO]")
            and "[MobileNetV1]" not in line)
    return _default_filter


def stop_filter():
    global _default_filter
    if _default_filter is not None:
        try:
            _default_filter.stop()
        except Exception:
            pass
        _default_filter = None
