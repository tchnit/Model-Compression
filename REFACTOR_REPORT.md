# Repository refactor — 18 October 2026

The approved flat `bdd100k_eval/` package now owns the implementations. Existing root commands and import names remain compatibility entry points. No dependency or framework was added. Model identifiers, artifact paths, dataset mappings, prediction logic, quantization settings, evaluation formulas and benchmark definitions are preserved.

Branch: `codex/modularize-bdd100k`. Original HEAD: `4b6616a`.

## Safety fixes and original changes

The initial worktree contained changes to `README.txt`, `demo_data/mvp.js`, `mvp_demo.py`, `test_mvp_demo.py` and `test_mvp_overlay.js`. These were preserved in checkpoint commit `cdbcd0e`, with the original binary patch and status saved before editing.

The two requested fixes precede all structural commits:

| Commit | Fix | Regression coverage |
| --- | --- | --- |
| `9b79a04` | Move upload-directory creation inside the inference-slot cleanup boundary. | Simulated directory-creation failure must release the slot. |
| `21ffb89` | Use `TemporaryDirectory` instead of `/tmp/test_report.json` in the comparison self-test. | Preserve an existing report; remove temporary fixtures after success and parser failure. |

Both regressions failed before their respective fixes and passed afterward. Neither changes inference or report calculations.

## Migration and file ownership

| Stage / commit | Files and changes |
| --- | --- |
| 1 / `4c0d8b8` | `bdd100k_eval/config.py` owns project-root resolution, the existing model registry and server limits. Callers use the same root paths. |
| 2 / `03311cb` | `artifacts.py` shares SHA-256, artifact metadata and directory-size helpers. `reporting.py` shares the identical JSON/CSV/table implementations. Different report formats remain separate. |
| 3 / `1fd8c3f` | `dataset.py` contains the existing BDD100K names, aliases, frame parsing and dataset preparation. `image_ops.py` contains unchanged Hardware image decoding, preprocessing and postprocessing. |
| 4 / `1ee2b37` | `inference_runtime.py` contains unchanged CPU/precision helpers. `models.py` owns the original lazy model cache and loading functions; offline imports no longer initialize the HTTP/MVP service. |
| 5 / `418a928` | `evaluation.py` owns the evaluator and worker implementation. Seven offline CLI implementations move to `tools/`. Root scripts retain commands, imports and virtual-environment bootstraps where previously present. Worker subprocesses still invoke root `evaluate.py`. |
| 6 / `0611903` | `api.py` owns HTTP handlers and routes; `server.py` owns runtime checks/startup; `mvp_demo.py` owns sessions and media processing. MVP evaluation moves into `evaluation.py`; its existing service methods remain compatibility adapters. |
| 7 / `54d0d0e`, `f6b1e10` | Extract Studio/dashboard inline source into five assets under `demo_data/`. `overlays.js` holds the existing shared palette and renderer. Root HTML URLs and Hardware worker URLs remain unchanged. Dashboard CSP permits same-origin scripts/styles, removes inline-script permission and retains its other restrictions. |
| 8 / final documentation commit | Document package ownership, deployment files, unchanged commands and test commands. Extend `verify_project` source hashes to package implementations, maintained JS/CSS, documentation and existing manifests. Add inventory/import-boundary coverage. Finish the remaining direct package import in `refresh_demo_data`. |

Root compatibility files:

- `run_demo.py` → `bdd100k_eval/api.py`, with startup delegated to `server.py`.
- `evaluate.py` → `bdd100k_eval/evaluation.py`.
- `mvp_demo.py`, `hardware_benchmark.py`, `inference_runtime.py` → their matching package modules.
- `compare_benchmarks.py`, `compare_ptq_vs_qat.py`, `compare_yolo26m_vs_bdd100k.py`, `prepare_runtime_models.py`, `refresh_demo_data.py`, `verify_project.py`, `verify_hardware.py` → matching modules in `bdd100k_eval/tools/`.

New frontend files: `demo_data/studio.js`, `studio.css`, `dashboard.js`, `dashboard.css`, `overlays.js`. The original page markup remains in `demo.html` and `model-compression-yolo26-dashboard.html`. Dashboard benchmark JSON stays embedded, so existing refresh logic continues to update it. Vendored runtime files, `hardware_benchmark.js`, `mvp.js`, browser model manifests, generated benchmark data and images retain their paths and contents.

Changed tests:

- `test_mvp_demo.py`, `test_compare_ptq_vs_qat.py`: requested safety regressions.
- `test_project_layout.py`: legacy/canonical module identity, shared mutable cache and offline-import boundaries.
- `test_verify_project.py`: package/UI source inventory includes implementations and excludes models/history/vendor internals.
- `test_run_demo.py`: package source stays private; extracted assets are served and dashboard CSP permits them. Shared registry/cache fixtures now patch their contents.
- `test_compare_benchmarks.py`: patch the explicit project root instead of the implementation file location.
- `test_hardware_benchmark.py`: wait up to one second for cleanup after receiving an HTTP response; response delivery can precede the handler's `finally` block.
- `test_demo_overlap.js`, `test_demo_selection.js`, `test_demo_exports.js`, `test_dashboard_state.js`, `test_hardware_browser.js`, `test_mvp_overlay.js`: load extracted implementations instead of slicing scripts from HTML. The Hardware label-width expectation is corrected as described below.

Only duplicate helper bodies, relocated implementation bodies and one unused local size calculation were removed. No historical file, dataset, trained artifact or uncertain functionality was deleted. Cohesive evaluation and backend functions were retained rather than split into additional abstractions.

## Baseline and reproducibility

Local evidence directory: `verification/refactor_20261018T124548Z/` (ignored by Git; preserved on this machine).

Before moving files, this directory recorded:

- `initial-status.txt`, `user-changes.patch`, original tracked-file SHA-256/symlink inventory and source copies under `originals/`.
- Python 3.12.3, Node v18.19.1, Linux x86_64, CPU affinity and all installed package versions in `environment.json`. Relevant versions: PyTorch 2.14.1, Ultralytics 8.4.175, OpenVINO 2026.4.1, ONNX Runtime GPU 1.30.0, NumPy 2.5.4, OpenCV 5.0.0.93, Pillow 12.3.0.
- SHA-256/size for 38 model/runtime-export files in `artifacts-before.json`.
- Metadata for 280,790 historical files/links, including 140,106 symlinks. JSON/CSV/YAML/Markdown also have content hashes.
- 53 passing Python tests after the safety fixes. Five JavaScript suites passed; `test_hardware_browser.js` had a pre-existing label-geometry expectation failure: expected x=1170/width=110, while the unchanged renderer produces x=1166/width=114. The final test uses the existing calculation.
- Seven complete CLI help outputs, real predictions/API schema keys, native runtime outputs, MVP image/video results, latency samples and RSS. `capture.py` reruns the same inputs, including a saved eight-frame WebM fixture.

Evidence scripts use new output locations. The existing `verification/` reports, benchmark snapshots and evaluation outputs were not overwritten.

## Verification results

- Final Python suite: **58 tests passed** (`.venv/bin/python -m unittest discover -v`).
- All six JavaScript suites pass: overlap, model selection, exports, dashboard state, Hardware browser helpers and MVP overlays.
- Seven CLI `--help` outputs match the baseline byte for byte. The root/package Python files compile; `pip check` reports no broken requirements. `run_demo.py` and `verify_hardware.py` do not implement a help parser and were not invoked with `--help`.
- Real API inference: two images × all ten model identifiers. Serialized detection boxes, runtime settings and response key sets match exactly; no numerical tolerance was needed.
- Native Hardware inference: YOLO26s PyTorch CPU FP32, ONNX Runtime CPU FP32 and OpenVINO CPU INT8 produce exactly matching boxes, runtime records and response key sets.
- MVP image and eight-frame video jobs: settings, runtime, counts, frame detections and report schema match exactly. Volatile latency fields are excluded from comparisons.
- Original saved evaluator versus new root CLI on the same two-image subset: dataset manifests, aggregate accuracy and every per-class metric match exactly. Smoke mAP50–95 is 0.4740602912633024; this subset is not a replacement for the historical 10,000-image benchmark.
- AST comparisons confirm unchanged runtime helpers, model-loading functions, prediction handlers, evaluation metric functions and image/data processing algorithms. Intentional differences are root/import resolution, safety fixes, module ownership and compatibility adapters. Extracted CSS/JS, palette, renderer and embedded benchmark JSON match their original source.
- Browser verification: real MVP image/video processing, playable video, interactive SVG boxes, table/box keyboard focus, downloads and dashboard initialization work. WASM FP32, FP16 and INT8 each return 13 detections on the selected scene with one thread. Worker/model loading uses the existing URLs and cross-origin isolation headers.
- Threaded WASM FP16 at two threads succeeds on the refactored page at a fresh origin, with exactly the same 13 boxes as the original page. A reused test origin stalled on both the refactored page and the saved original HTML; reloading did not resolve it. Fresh origins succeeded for both. This browser/origin-state issue remains unresolved; worker source, vendored runtime and model files are unchanged. See `browser-thread-investigation.json`.
- Preservation verification: 38 model/runtime files retain SHA-256 and sizes; historical metadata/content checks pass; original tracked assets outside the stated source changes remain identical. `index.html → demo.html` and the model-directory alias retain their targets.
- `verify_project.py --requirements requirements-hardware.txt --output <new evidence directory>` verifies all ten configurations, standalone OpenVINO loading and all 10,000 validation images. The new image manifest matches every original hash, size and dimension. This environment uses the existing NVIDIA dependency profile: its 14 pins pass. The CPU profile expects the separate `onnxruntime` distribution; this environment already contains `onnxruntime-gpu` instead. No package was installed or replaced.

Intermediate failures were investigated: the old Hardware JS assertion was corrected without changing the renderer; the HTTP cleanup test now synchronizes with the existing cleanup boundary; browser testing identified and corrected the dashboard asset CSP before completion. Final tests contain no new failures.

## Latency and memory investigation

The first before/after sample used 12 warm requests per configuration after loading all ten models:

| Measurement | Before | After |
| --- | ---: | ---: |
| YOLO26s PTQ INT8 median | 43.5 ms | 54.7 ms |
| `yolo26m_fp32` median (existing key; actual CPU FP16) | 2697.7 ms | 2783.35 ms |
| End-of-probe process RSS | 2168.69 MiB | 2133.75 MiB |

The initial PTQ increase was material and triggered a separate comparison. Two original-server runs alternated with two refactored-server runs, each with five warmups and 40 measured HTTP requests. Combined medians were **58.1 ms original / 53.45 ms refactored** (80 samples per implementation); means were 66.36/57.20 ms. This did not reproduce a refactor slowdown. Host-load snapshots and every sample are in `latency-investigation.json` and the accompanying logs. The baseline/after process RSS fell by 34.93 MiB (1.61%); this is resident memory at the same probe endpoint, not a claim about peak model memory. Shared-host measurements do not establish a deployment performance guarantee.

## Evidence and rollback

The evidence directory contains per-stage logs, `before.json`, `after.json`, `inference-equivalence.json`, `evaluation-equivalence.json`, `frontend-extraction.json`, `function-equivalence.json`, `latency-investigation.json`, `preservation.json`, CLI help snapshots, browser results and screenshots. The final Git change inventory records every changed path.

The last pre-structure checkpoint is **`21ffb89`**, which includes the original user changes and both safety fixes. It is the rollback point for the entire migration. Each later migration commit is separate, so it can be reverted in reverse order after preserving any subsequent work. No reset, artifact deletion or historical-output rewrite is required. The original `main` branch reference remains unchanged.

## Remaining limits

Full 10,000-image inference for every model, GPU/CUDA/TensorRT/WebGPU execution, webcam capture, all video codecs and a clean-machine reinstall were not performed. This browser exposes no WebGPU adapter. Threaded WASM can stall at a reused browser origin on both original and refactored pages; its underlying browser-state cause was not resolved. Native CPU and representative browser/media checks establish the refactor comparisons described above; they do not certify untested hardware or new model accuracy. Existing historical FP32 results and pending FP16 benchmark labels remain as they were.

Deploy the whole package and extracted frontend assets with the root wrappers. Copying only the old root Python/HTML files is no longer sufficient. No benchmark refresh, model export, training, recalibration or framework rewrite was performed.
