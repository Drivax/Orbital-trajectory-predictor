import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from ..dynamics.propagator import propagate

MODELS = ("Constant Velocity", "Two Body", "Two Body + J2", "Pure ML", "Physics + ML residual")
COLORS = dict(zip(MODELS, ("#777777", "#d97706", "#059669", "#be185d", "#2563eb")))

def make_figures(output, groups, predictions, summary, config):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "figure.dpi":120, "savefig.dpi":220})
    minutes = np.asarray(config.horizons_s)/60
    def curve(split, model, metric):
        rows = sorted((r for r in summary if r["split"]==split and r["model"]==model),
                      key=lambda r:r["horizon_min"])
        return np.array([r[metric+"_mean"] for r in rows]), np.array([r[metric+"_std"] for r in rows])
    def save(fig, name):
        fig.savefig(output/name, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4.6))
    for model in MODELS:
        mean, std = curve("test",model,"mae_3d_km")
        ax.plot(minutes,mean,"o-",color=COLORS[model],label=model)
        ax.fill_between(minutes,np.maximum(mean-std,1e-8),mean+std,color=COLORS[model],alpha=.15)
    ax.set(xscale="log", yscale="log", xlabel="Prediction horizon (min)",
           ylabel="Mean 3D position error (km)", title="Held-out ID satellites: error vs horizon")
    ax.set_xticks(minutes, [str(int(m)) for m in minutes])
    ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2)
    save(fig,"01_error_vs_horizon.png")

    group = groups["test"]
    seed = config.seeds[0]
    times = np.linspace(0,max(config.horizons_s),500)
    reference = propagate(group["state"][0],times)
    physics = propagate(group["state"][0],times,"two_body")
    fig = plt.figure(figsize=(10,5.4))
    ax = fig.add_subplot(121,projection="3d")
    ax.plot(*reference[:,:3].T, color="black",lw=1.5,label="Reference (dense)")
    ax.plot(*physics[:,:3].T,color=COLORS["Two Body"],ls="--",label="Two Body (dense)")
    for name,marker in (("Pure ML","x"),("Physics + ML residual","o")):
        points = predictions[("test",name,seed)][0]
        ax.scatter(*points.T,color=COLORS[name],marker=marker,s=35,label=name+" (5 horizons)")
    ax.set(xlabel="ECI x (km)",ylabel="ECI y (km)",zlabel="ECI z (km)",title="First held-out ID satellite")
    extent = 1.05 * max(np.abs(reference[:,:3]).max(), np.abs(physics[:,:3]).max(),
                        np.abs(predictions[("test","Pure ML",seed)][0]).max(),
                        np.abs(predictions[("test","Physics + ML residual",seed)][0]).max())
    ax.set(xlim=(-extent,extent), ylim=(-extent,extent), zlim=(-extent,extent))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax.set_zlabel("ECI z (km)", labelpad=14)
    ax.set_box_aspect((1,1,1)); fig.legend(*ax.get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=2)
    error_ax = fig.add_subplot(122)
    for name in ("Two Body","Pure ML","Physics + ML residual"):
        key_seed = seed if "ML" in name else -1
        distances = np.linalg.norm(predictions[("test",name,key_seed)][0]-group["target"][0,:,:3],axis=1)
        error_ax.plot(minutes,distances,"o-",label=name,color=COLORS[name])
    error_ax.set(xscale="log",yscale="log",xlabel="Horizon (min)",ylabel="3D error (km)",
                 title=f"Same forecast; ML seed {seed}")
    error_ax.grid(alpha=.2)
    fig.tight_layout(rect=(0,0.13,1,1))
    save(fig,"02_example_trajectory.png")

    fig, ax = plt.subplots(figsize=(8,4.8))
    distributions = []
    for name in MODELS:
        key_seed = seed if "ML" in name else -1
        distributions.append(np.linalg.norm(predictions[("test",name,key_seed)][:,-1]-group["target"][:,-1,:3],axis=1))
    ax.boxplot(distributions,tick_labels=["CV","Two Body","+ J2","Pure ML","Hybrid"],showfliers=True)
    p95 = [np.percentile(d,95) for d in distributions]
    ax.scatter(range(1,6),p95,color="red",marker="D",label="P95",zorder=4)
    ax.set(yscale="log",ylabel="3D position error (km)",
           title=f"ID error distribution at {minutes[-1]:g} min; ML seed {seed}")
    ax.legend(); ax.grid(axis="y",alpha=.2)
    save(fig,"03_error_distribution.png")

    fig, axes = plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,index in zip(axes,(2,-1)):
        xs = np.arange(5)
        for offset,split,label,color in ((-.18,"test","ID","#2563eb"),(.18,"ood","OOD","#d97706")):
            vals = [curve(split,m,"mae_3d_km")[0][index] for m in MODELS]
            stds = [curve(split,m,"mae_3d_km")[1][index] for m in MODELS]
            ax.bar(xs+offset,vals,.36,yerr=stds,label=label,color=color,capsize=2)
        ax.set_xticks(xs,["CV","Two Body","+ J2","Pure ML","Hybrid"],rotation=20)
        ax.set(yscale="log",title=f"Horizon {minutes[index]:g} min",ylabel="Mean 3D error (km)")
        ax.legend(); ax.grid(axis="y",alpha=.2)
    fig.suptitle("Held-out satellites: ID vs joint orbital distribution shift")
    fig.tight_layout()
    save(fig,"04_id_vs_ood.png")

    fig, axes = plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,split in zip(axes,("test","ood")):
        for name,label in (("Two Body","Before: Two Body"),("Physics + ML residual","After: learned residual")):
            mean,std = curve(split,name,"mae_3d_km")
            ax.plot(minutes,mean,"o-",label=label,color=COLORS[name])
            ax.fill_between(minutes,np.maximum(mean-std,1e-8),mean+std,color=COLORS[name],alpha=.15)
        ax.set(xscale="log",yscale="log",xlabel="Horizon (min)",ylabel="Mean 3D error (km)",
               title="ID" if split=="test" else "OOD")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle("Residual learning: two-body error before and after correction")
    fig.tight_layout()
    save(fig,"05_residual_learning.png")
