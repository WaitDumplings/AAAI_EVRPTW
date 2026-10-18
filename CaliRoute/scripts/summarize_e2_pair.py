#!/usr/bin/env python3
"""Compare two accepted, protocol-matched E2 per-instance test files."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e2.protocol import paired_results, require_matched_protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-protocol',required=True,type=Path)
    p.add_argument('--variant-protocol',required=True,type=Path)
    p.add_argument('--base-results',required=True,type=Path)
    p.add_argument('--variant-results',required=True,type=Path)
    p.add_argument('--panel',choices=('A','B'),required=True)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    base=json.loads(a.base_protocol.read_text());variant=json.loads(a.variant_protocol.read_text())
    require_matched_protocol(base,variant,a.panel)
    for cfg in (base,variant):
        if cfg.get('independent_test_accepted') is not True:
            raise ValueError('Independent test acceptance required')
    def read(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result=paired_results(read(a.base_results),read(a.variant_results),training_seed=base['training_seed'],
                          k=base['evaluation_protocol']['K'],expected_count=base['evaluation_protocol']['test_count'])
    result.update(panel=a.panel,base_protocol=str(a.base_protocol.resolve()),variant_protocol=str(a.variant_protocol.resolve()),
                  base_results=str(a.base_results.resolve()),variant_results=str(a.variant_results.resolve()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(a.output)
    print(json.dumps({k:v for k,v in result.items() if k!='common_instance_ids'},indent=2))

if __name__=='__main__':main()
