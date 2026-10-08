"""Read whole-device VRAM usage through the installed AMD SMI library.

The operational Card 04 launcher invokes this in the registered ROCm environment.
It deliberately makes one reading per process and exits nonzero on any identity or
measurement failure, so its caller can stop the owned training process group.
"""

from __future__ import annotations

import argparse


def read_vram_bytes(expected_uuid: str) -> int:
    import amdsmi

    amdsmi.amdsmi_init()
    try:
        devices = amdsmi.amdsmi_get_processor_handles()
        if len(devices) != 1:
            raise RuntimeError(f"expected one GPU, found {len(devices)}")
        device = devices[0]
        uuid = amdsmi.amdsmi_get_gpu_device_uuid(device)
        if uuid != expected_uuid:
            raise RuntimeError(f"unexpected GPU UUID: {uuid}")
        used = amdsmi.amdsmi_get_gpu_memory_usage(device, amdsmi.AmdSmiMemoryType.VRAM)
        total = amdsmi.amdsmi_get_gpu_memory_total(device, amdsmi.AmdSmiMemoryType.VRAM)
        if (
            type(used) is not int
            or type(total) is not int
            or total <= 0
            or not 0 <= used <= total
        ):
            raise RuntimeError("invalid VRAM reading")
        return used
    finally:
        amdsmi.amdsmi_shut_down()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-uuid", required=True)
    args = parser.parse_args(argv)
    print(read_vram_bytes(args.expected_uuid))


if __name__ == "__main__":
    main()
