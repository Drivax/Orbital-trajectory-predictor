"""Run the controlled orbital benchmark without downloads or external services."""
import argparse
import csv
import json
import pickle
import platform
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from .config import Config
from .data.synthetic import generate
from .dynamics.propagator import propagate
from .models.residual_model import train, predict
from .evaluation.metrics import position_metrics, summarize
from .evaluation.report import markdown_results
from .visualization.plots import make_figures

ROOT = Path(__file__).resolve().parents[1]

def write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def run(config, output, model_dir, update_readme=False):
    started=time.perf_counter()
    output,model_dir=Path(output),Path(model_dir)
    output.mkdir(parents=True,exist_ok=True)
    model_dir.mkdir(parents=True,exist_ok=True)
    groups, manifest=generate(config)
    (output/"split_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    np.savez_compressed(output/"dataset.npz", **{f"{split}_{key}":value for split,g in groups.items() for key,value in g.items()})
    records, predictions, training = [], {}, []
    for split in ("test","ood"):
        group=groups[split]
        predictions[(split,"Constant Velocity",-1)] = (group["state"][:,None,:3]+
            group["state"][:,None,3:]*np.asarray(config.horizons_s)[None,:,None])
        predictions[(split,"Two Body",-1)]=group["two_body"]
        predictions[(split,"Two Body + J2",-1)]=group["j2"]
    with threadpool_limits(limits=1):
        for seed in config.seeds:
            for kind,name in (("pure","Pure ML"),("hybrid","Physics + ML residual")):
                print(f"Training {name}, seed {seed}",flush=True)
                model=train(groups["train"],groups["val"],kind,seed,config)
                with (model_dir/f"{kind}_{seed}.pkl").open("wb") as stream:
                    pickle.dump(model,stream)
                training.append({k:v for k,v in model.items() if k in ("kind","seed","best_epoch","history")})
                for split in ("test","ood"):
                    predictions[(split,name,seed)]=predict(model,groups[split])
    raw_errors=[]
    arrays={}
    for (split,name,seed),prediction in predictions.items():
        group=groups[split]
        arrays[f"{split}__{name.replace(' ','_')}__{seed}"]=prediction
        for i,horizon in enumerate(config.horizons_s):
            records.append(dict(split=split,model=name,seed=seed,horizon_min=horizon/60,
                                **position_metrics(prediction[:,i],group["target"][:,i])))
            distances=np.linalg.norm(prediction[:,i]-group["target"][:,i,:3],axis=1)
            for satellite,origin,error in zip(group["satellite"],group["origin_s"],distances):
                raw_errors.append(dict(split=split,model=name,seed=seed,satellite=satellite,
                                       origin_s=int(origin),horizon_min=horizon/60,error_km=float(error)))
    summary=summarize(records)
    write_csv(output/"metrics_by_seed.csv",records)
    write_csv(output/"metrics_summary.csv",summary)
    write_csv(output/"errors.csv",raw_errors)
    np.savez_compressed(output/"predictions.npz",**arrays)
    (output/"metrics.json").write_text(json.dumps({"by_seed":records,"summary":summary},indent=2),encoding="utf-8")
    (output/"training.json").write_text(json.dumps(training,indent=2),encoding="utf-8")
    convergence=[]
    for split in ("test","ood"):
        # Sample a full window at each evaluated distribution to check solver sensitivity.
        state=groups[split]["state"][0]
        nominal=propagate(state,config.horizons_s,"reference",config.rtol,config.atol,config.max_step_s)
        tight=propagate(state,config.horizons_s,"reference",1e-12,1e-14,30)
        convergence.append(dict(split=split,max_position_difference_km=float(np.max(np.linalg.norm(nominal[:,:3]-tight[:,:3],axis=1)))))
    metadata=dict(config=config.as_dict(),python=platform.python_version(),
                  dependencies={name:version(name) for name in ("numpy","scipy","scikit-learn","matplotlib")},
                  convergence=convergence, sample_counts={k:len(v["state"]) for k,v in groups.items()},
                  duration_seconds=time.perf_counter()-started,
                  reference="two-body + J2 + fixed exponential drag; synthetic, not operational truth")
    (output/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    make_figures(output/"figures",groups,predictions,summary,config)
    report=markdown_results(summary,config)
    (output/"results.md").write_text(report,encoding="utf-8")
    if update_readme:
        path=ROOT/"README.md"
        text=path.read_text(encoding="utf-8")
        begin,end="<!-- RESULTS:START -->","<!-- RESULTS:END -->"
        prefix=text.split(begin)[0]
        suffix=text.split(end)[1]
        path.write_text(prefix+begin+"\n"+report+end+suffix,encoding="utf-8")
    print(f"Saved outputs in {output}",flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true",help="Smoke run; does not update the README")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--models",type=Path)
    args=parser.parse_args()
    config=Config()
    if args.quick:
        config=replace(config,train_satellites=8,val_satellites=3,test_satellites=3,ood_satellites=3,
                       origins_per_satellite=2,epochs=15,seeds=(11,))
    output=args.output or ROOT/("results/smoke" if args.quick else "results")
    models=args.models or ROOT/("models/smoke" if args.quick else "models")
    run(config,output,models,update_readme=not args.quick and output.resolve()==(ROOT/"results").resolve())

if __name__=="__main__":
    main()
