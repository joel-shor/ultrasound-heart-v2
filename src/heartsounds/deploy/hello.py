"""1-GPU Beaker smoke probe verifying CUDA, bf16 support, and `/data` integrity."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import pyarrow.parquet as pq
import torch

from heartsounds.storage.gcs import sha256_file


def run_hello_probe(
    data_dir: str | Path = "/data",
    results_dir: str | Path | None = None,
) -> dict[str, object]:
    """Inspect GPU runtime and optional mounted `/data` dataset, writing `hello.json`."""
    out_root = Path(results_dir or os.environ.get("RESULTS_DIR", "/results"))
    out_root.mkdir(parents=True, exist_ok=True)

    cuda_ok = bool(torch.cuda.is_available())
    if not cuda_ok:
        raise RuntimeError("CUDA is not available inside Beaker GPU probe!")

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    bf16_ok = bool(torch.cuda.is_bf16_supported())
    vram_gib = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

    # Quick matmul sanity check on GPU
    x = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16 if bf16_ok else torch.float32)
    _ = (x @ x.T).sum().item()

    data_root = Path(data_dir)
    data_info: dict[str, object] = {"mounted": data_root.is_dir(), "files": {}}
    if data_root.is_dir():
        files_summary: dict[str, object] = {}
        for pq_name in ("manifest.parquet", "train_val_folds.parquet", "test.parquet"):
            fpath = data_root / pq_name
            if fpath.is_file():
                pf = pq.ParquetFile(fpath)
                files_summary[pq_name] = {
                    "num_rows": pf.metadata.num_rows,
                    "sha256": sha256_file(fpath),
                }
        data_info["files"] = files_summary

    report: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda_ok,
        "device_count": device_count,
        "device_name": device_name,
        "vram_gib": vram_gib,
        "bf16_supported": bf16_ok,
        "data": data_info,
    }
    out_file = out_root / "hello.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_hello_probe()
