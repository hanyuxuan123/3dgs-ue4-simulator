from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def try_add_carla_egg() -> None:
    carla_root = os.environ.get("CARLA_ROOT")
    if not carla_root:
        return

    dist = Path(carla_root) / "PythonAPI" / "carla" / "dist"
    if not dist.exists():
        return

    version = f"py{sys.version_info.major}.{sys.version_info.minor}"
    eggs = sorted(dist.glob(f"carla-*-{version}-*.egg"))
    if eggs:
        sys.path.append(str(eggs[-1]))


def check_import(module: str, package_name: str | None = None) -> bool:
    label = package_name or module
    try:
        imported = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {label}: {exc}")
        return False

    version = getattr(imported, "__version__", "unknown")
    print(f"[ OK ] {label}: {version}")
    return True


def main() -> int:
    try_add_carla_egg()

    ok = True
    ok &= check_import("gs_carla")
    ok &= check_import("numpy")
    ok &= check_import("PIL", "pillow")
    ok &= check_import("scipy")
    ok &= check_import("torch")
    ok &= check_import("gsplat")
    ok &= check_import("carla")

    try:
        import torch

        print(f"[INFO] torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] gsplat rendering needs a CUDA-visible PyTorch session.")
    except Exception:
        pass

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
