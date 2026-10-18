# Repository cleanup report

Date: 18 October 2026  
Branch: `codex/modularize-bdd100k`  
Cleanup base commit: `1abc71ed2eec60bfba8703a71ec683f9323517c3`

## Result

Cleanup option A+B completed successfully. Only the Python bytecode files in the approved allowlist were deleted.

| Set | Deleted files | Content bytes | Allocated bytes recovered |
| --- | ---: | ---: | ---: |
| A — project caches | 42 | 408,972 | 520,192 |
| B — virtual-environment caches | 8,852 | 180,320,794 | 200,343,552 |
| **Total** | **8,894** | **180,729,766** | **200,863,744 (191.56 MiB)** |

The filesystem free-space increase measured immediately after the verified backup and immediately after deletion was exactly 200,863,744 bytes. No cache directory was deleted.

The recovery archive occupies 63,422,464 allocated bytes, so keeping the rollback copy reduces the current net free-space benefit to 137,441,280 bytes (131.07 MiB). The archive can be removed later only after its retention value is reviewed.

## Safety controls

Before deletion, every candidate was checked against the audit allowlist for:

- a relative path inside one of the three approved project `__pycache__` directories or inside `.venv/**/__pycache__`;
- a `.pyc` suffix and a regular, non-symlink target;
- no symlink in any target ancestor;
- a single hard link;
- the recorded size, allocated size and SHA-256;
- an existing regular, non-symlink Python source file with the recorded SHA-256.

All 8,894 candidates passed. Each target's type, inode, size and SHA-256 was checked again immediately before unlinking. There were no rejected, skipped, changed or remaining approved targets.

The exact deleted-file allowlist, source hashes and execution manifest remain outside the repository under:

`/home/merlin/.codex/visualizations/2026/10/18/01a14ef1-d181-76a2-9967-48a859051b8b/cleanup-audit/`

The recoverable archive is:

`python-bytecode-A-B-backup-20261018T162703Z.tar.gz`

Archive SHA-256: `e9e0c9375be8bc8a3a86d2ac5f977b527cadf480d75b229928c1fe15fcc94607`

No source code, installed package, package metadata, dependency, virtual environment, model, dataset, evaluation result, benchmark report, verification record, symlink or cache directory was deleted. The dashboard HTML backup remains unchanged with SHA-256 `210f650d0a1fec1f452b8401374cfb0bb9c777a5d786e8850d0cbca1b2810581`. The 16 import bindings marked REVIEW were not edited.

## Verification

- Python: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -v` — **58 tests passed** in 41.055 seconds.
- JavaScript: all six `node test_*.js` suites — **passed**.
- Dependency consistency: `.venv/bin/python -m pip check` — **no broken requirements**.
- CLI compatibility: all seven supported `--help` outputs matched the saved post-refactor baseline byte for byte.
- Runtime coverage: the Python suite exercised real CPU/OpenVINO model loading, image/video MVP inference, API and static routes, startup checks, compatibility imports and cleanup regressions.
- Artifact integrity: all 38 recorded model and runtime-export files matched their preserved sizes and SHA-256 values.
- Links: no broken repository symlink was found.
- Bytecode state after verification: zero approved project or `.venv` `.pyc` files remained. Tests used `PYTHONDONTWRITEBYTECODE=1` so verification did not recreate them.

Git was clean before cleanup and remained clean after bytecode deletion and verification. The only repository change produced by this task is this requested report. No unexpected tracked deletion or modification was found.

## Limits and rollback

Python will recreate bytecode during normal execution, so some recovered space is temporary and the first uncached imports can be slower. Full 10,000-image evaluation, GPU/CUDA/TensorRT/WebGPU execution and a new live-browser run were not repeated because the approved change removed only interpreter-generated caches; the existing automated tests cover model loading, representative inference, API behavior and frontend logic.

To roll back the cache deletion, extract the recovery archive at the repository root after verifying its SHA-256. Restoring bytecode is optional because Python can regenerate it from the retained sources.
