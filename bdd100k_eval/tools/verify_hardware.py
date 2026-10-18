#!/usr/bin/env python3
"""Smoke every available native hardware combination; not an accuracy/performance claim."""
import json
import time
from pathlib import Path
from bdd100k_eval.config import ROOT
from bdd100k_eval import hardware_benchmark as bench


def main():
    root = ROOT
    data = json.loads((root / 'demo_data/detections.json').read_text())
    scene = data['scenes'][0]
    image, digest = bench.decode_image(scene['data_uri'])
    caps = bench.capabilities()
    results, failures = [], []
    for config in caps['configs']:
        if not config['available'] or config['engine'] in ['webgpu','wasm']:
            continue
        choice = {k:config[k] for k in ['model','engine','device','precision']}
        print('RUN',choice,flush=True)
        try:
            record = bench.run_benchmark(bench.validate({**choice,'warmup':1,'repeats':3}),image,digest)
            assert record['detections_count'] > 0, 'Expected nonempty detections on traffic fixture'
            assert all(t > 0 for t in record['samples_ms']), 'Invalid timings'
            assert record['p50_ms'] <= record['p95_ms'], 'Invalid percentiles'
            results.append(record)
            print('OK',record['detections_count'],round(record['p50_ms'],2),flush=True)
        except Exception as error:
            failures.append(dict(**choice,error=f'{type(error).__name__}: {error}'))
            print('FAIL',error,flush=True)
    output = dict(timestamp=time.time(),purpose='functional smoke; shared hardware, not a benchmark ranking',
                  scene=scene['file'],capabilities=caps,results=results,failures=failures)
    (root / 'verification/hardware_smoke.json').write_text(json.dumps(output,indent=2)+'\n')
    print(f'Passed {len(results)}; failed {len(failures)}',flush=True)
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
