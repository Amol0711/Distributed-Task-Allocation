#!/usr/bin/env python3
import csv, math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; src=ROOT/'results'/'trajectory_microcase'/'episode_records.csv'; outdir=ROOT/'results'/'trajectory_microcase'/'figure_data'; outdir.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader(src.open())); by=defaultdict(list)
for r in rows: by[int(r['seed'])].append(r)
for rs in by.values(): rs.sort(key=lambda r:int(r['episode']))
out=[]
for k in range(1,241):
    ret=[]; cert=[]; raw=[]; zd=[]; p=[]; tr=[]; eta=[]
    for rs in by.values():
        pre=rs[:k]; ret.append(sum(float(x['true_value']) for x in pre)/sum(float(x['optimal_value']) for x in pre)); cert.append(float(pre[-1]['cumulative_universal_utilization'])); raw.append(2*sum(float(x['raw_certificate_increment']) for x in pre)/(k*1.2)); ex=[x for x in pre if int(x['exploration'])==0]; zd.append(sum(int(x['zero_scale_difference']) for x in ex)/len(ex) if ex else 0); p.append(float(pre[-1]['parameter_confidence_ratio'])); tr.append(float(pre[-1]['tracking_tube_utilization'])); eta.append(float(pre[-1]['trajectory_deviation_utilization']))
    def ms(a): a=np.asarray(a); return float(a.mean()),float(a.std(ddof=1)/math.sqrt(len(a)))
    rm,rse=ms(ret); cm,cse=ms(cert); wm,wse=ms(raw); zm,zse=ms(zd)
    out.append({'episode':k,'cumulative_retention_mean':rm,'cumulative_retention_se':rse,'cumulative_certificate_mean':cm,'cumulative_certificate_se':cse,'cumulative_raw_certificate_mean':wm,'cumulative_raw_certificate_se':wse,'zero_scale_difference_mean':zm,'zero_scale_difference_se':zse,'parameter_confidence_max':max(p),'tracking_utilization_max':max(tr),'return_deviation_utilization_max':max(eta)})
with (outdir/'trajectory_running.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(out[0]),lineterminator='\n'); w.writeheader(); w.writerows(out)
