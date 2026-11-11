#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Record the machine that produced a timing file.

The same block is appended for Python, Spark, and MapReduce so two devices
can be compared without mixing their wall-clock seconds.
"""
from __future__ import print_function

import os
import platform
import socket
import subprocess
import sys


def _text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = value.decode("utf-8", "replace")
        except Exception:
            value = str(value)
    return " ".join(value.split())


def _run(args):
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT)
    except Exception:
        return ""
    return _text(out)


def _cpu_model():
    if sys.platform == "darwin":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            return brand
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return _text(line.split(":", 1)[1])
        except Exception:
            pass
    if sys.platform.startswith("win"):
        brand = _run(["wmic", "cpu", "get", "Name"])
        parts = [p for p in brand.split(" ") if p and p.lower() != "name"]
        if parts:
            return " ".join(parts)
        brand = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ]
        )
        if brand:
            return brand
    return _text(platform.processor()) or "unknown"


def _memory_bytes():
    """Return (total_bytes, available_bytes). Either may be ''."""
    if sys.platform == "darwin":
        total = _run(["sysctl", "-n", "hw.memsize"])
        page = _run(["sysctl", "-n", "hw.pagesize"])
        vm = _run(["vm_stat"])
        avail_pages = 0
        have = False
        for label in ("Pages free", "Pages inactive", "Pages speculative"):
            marker = label + ":"
            start = vm.find(marker)
            if start < 0:
                continue
            rest = vm[start + len(marker):]
            number = ""
            for ch in rest:
                if ch.isdigit():
                    number += ch
                elif number:
                    break
            if number:
                avail_pages += int(number)
                have = True
        avail = ""
        if have and page.isdigit():
            avail = str(avail_pages * int(page))
        if total.isdigit():
            return total, avail
    if os.path.exists("/proc/meminfo"):
        info = {}
        try:
            with open("/proc/meminfo") as handle:
                for line in handle:
                    key, _, rest = line.partition(":")
                    number = rest.strip().split()
                    if number and number[0].isdigit():
                        info[key] = int(number[0]) * 1024
        except Exception:
            info = {}
        total = info.get("MemTotal", "")
        avail = info.get("MemAvailable", info.get("MemFree", ""))
        return (str(total) if total != "" else ""), (str(avail) if avail != "" else "")
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return str(int(stat.ullTotalPhys)), str(int(stat.ullAvailPhys))
        except Exception:
            return "", ""
    return "", ""


def peak_rss_pair():
    """Peak resident memory of this process and of finished child processes."""
    try:
        import resource
    except ImportError:
        return "", ""
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    return str(int(own) * scale), str(int(kids) * scale)


def machine_rows():
    total, avail = _memory_bytes()
    total_gb = ""
    if total.isdigit() and int(total) > 0:
        total_gb = "%.3f" % (float(total) / float(1024 ** 3))
    try:
        import multiprocessing
        cores = multiprocessing.cpu_count()
    except Exception:
        cores = None
    return [
        ("host", _text(socket.gethostname()) or "unknown"),
        ("os", _text(platform.platform()) or "unknown"),
        ("cpu_model", _cpu_model()),
        ("cpu_cores", "" if cores is None else str(int(cores))),
        ("mem_total_bytes", total),
        ("mem_total_gb", total_gb),
        ("mem_available_bytes", avail),
    ]


def format_block(rows):
    return "".join("%s=%s\n" % (key, value) for key, value in rows)


def write_result(path, rows):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write(format_block(list(rows) + machine_rows()))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--append":
        with open(sys.argv[2], "a") as handle:
            handle.write(format_block(machine_rows()))
        return
    sys.stdout.write(format_block(machine_rows()))


if __name__ == "__main__":
    main()
