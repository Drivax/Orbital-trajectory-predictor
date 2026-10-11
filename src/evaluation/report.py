from ..visualization.plots import MODELS

def markdown_results(summary, config):
    def row(split,model,horizon):
        return next(r for r in summary if r["split"]==split and r["model"]==model and r["horizon_min"]==horizon)
    horizons = [s/60 for s in config.horizons_s]
    lines = ["Mean 3D error in **km** on held-out ID satellites; OOD column is at the longest horizon.",
             "ML entries are mean ± sample std across training seeds, not confidence intervals.",
             "", "| Model | "+" | ".join(f"{h:g} min" for h in horizons)+" | OOD |",
             "|---|"+"---:|"*(len(horizons)+1)]
    for model in MODELS:
        cells=[]
        for split,h in [("test",h) for h in horizons]+[("ood",horizons[-1])]:
            item=row(split,model,h)
            text=f'{item["mae_3d_km_mean"]:.3g}'
            if item["seed_count"]>1:
                text+=f' ± {item["mae_3d_km_std"]:.2g}'
            cells.append(text)
        lines.append("| "+model+" | "+" | ".join(cells)+" |")
    lines += ["", "### Key findings", ""]
    for split,label in (("test","ID"),("ood","OOD")):
        h = horizons[-1]
        before=row(split,"Two Body",h)["mae_3d_km_mean"]
        after=row(split,"Physics + ML residual",h)["mae_3d_km_mean"]
        change=100*(after/before-1)
        lines.append(f"- At {h:g} min on {label}, the hybrid changes two-body error by {change:+.1f}% ({before:.3g} → {after:.3g} km).")
    best=min(MODELS,key=lambda m:row("test",m,horizons[-1])["mae_3d_km_mean"])
    lines.append(f"- The lowest ID mean error at {horizons[-1]:g} min is obtained by **{best}**.")
    pure=row("test","Pure ML",horizons[-1])["mae_3d_km_mean"]
    lines.append(f"- Pure ML has {pure:.3g} km ID mean error at {horizons[-1]:g} min; no superiority over physics is assumed.")
    return "\n".join(lines)+"\n"
