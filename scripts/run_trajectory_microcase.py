#!/usr/bin/env python3
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from trajectory_microcase import execute
if __name__=='__main__':
    s=execute(ROOT/'configs'/'trajectory_microcase.json', ROOT/'results'/'trajectory_microcase')
    print(json.dumps(s['results'],indent=2,sort_keys=True))
