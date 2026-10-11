from dataclasses import replace
import pickle
import numpy as np
import pytest
from src.config import Config
from src.data.synthetic import initial_state, generate, frames, features
from src.dynamics.two_body import MU, acceleration
from src.dynamics.j2 import acceleration as j2_acceleration
from src.dynamics.propagator import propagate, drag_acceleration
from src.models.residual_model import train, predict, residual_targets
from src.evaluation.metrics import position_metrics

@pytest.fixture(scope="module")
def sample():
    config=replace(Config(),train_satellites=3,val_satellites=2,test_satellites=2,
                   ood_satellites=2,origins_per_satellite=2,epochs=3,seeds=(11,))
    return config, generate(config,progress=False)

def test_two_body_energy_and_angular_momentum():
    state,_=initial_state(np.random.default_rng(1))
    orbit=propagate(state,np.linspace(0,21600,200),"two_body")
    energy=np.sum(orbit[:,3:]**2,axis=1)/2-MU/np.linalg.norm(orbit[:,:3],axis=1)
    angular=np.cross(orbit[:,:3],orbit[:,3:])
    assert np.max(np.abs((energy-energy[0])/energy[0]))<1e-8
    assert np.max(np.linalg.norm(angular-angular[0],axis=1))/np.linalg.norm(angular[0])<1e-8

def test_circular_orbit_analytic_solution():
    radius=7000.
    state=np.array([radius,0,0,0,np.sqrt(MU/radius),0])
    period=2*np.pi*np.sqrt(radius**3/MU)
    quarter=propagate(state,[period/4],"two_body")[0]
    np.testing.assert_allclose(quarter[:3],[0,radius,0],atol=1e-5)

def test_forces():
    r=np.array([7000.,0,0])
    assert acceleration(r)[0]<0
    assert j2_acceleration(r)[0]<0
    assert j2_acceleration(np.array([0.,0,7000]))[2]>0
    v=np.array([0,7.5,0])
    assert np.dot(drag_acceleration(r,v),v)<0

def test_deterministic_propagation():
    a,_=initial_state(np.random.default_rng(42))
    b,_=initial_state(np.random.default_rng(42))
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(propagate(a,[300,1800]),propagate(b,[300,1800]))

def test_splits_and_dimensions(sample):
    config,(groups,manifest)=sample
    ids=[set(g["satellite"]) for g in groups.values()]
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            assert a.isdisjoint(b)
    for group in groups.values():
        assert group["target"].shape[1:]==(5,6)
        assert group["two_body"].shape[1:]==(5,3)
        assert features(group["state"]).shape[1]==6
        basis=frames(group["state"])
        np.testing.assert_allclose(np.einsum("nji,njk->nik",basis,basis),
                                   np.broadcast_to(np.eye(3),(len(basis),3,3)),atol=1e-12)
        for satellite in set(group["satellite"]):
            times=group["origin_s"][group["satellite"]==satellite]
            assert np.all(np.diff(times)>max(config.horizons_s))
    assert all(m["inclination_deg"]>=90 for m in manifest if m["split"]=="ood")

def test_residual_targets():
    physics=np.ones((2,5,3))*10
    target=physics+np.arange(30).reshape(2,5,3)
    np.testing.assert_array_equal(physics+residual_targets(target,physics),target)
    with pytest.raises(ValueError):
        residual_targets(target,physics[0])

def test_train_only_scaling_and_checkpoint(sample):
    config,(groups,_)=sample
    model=train(groups["train"],groups["val"],"hybrid",11,config)
    np.testing.assert_allclose(model["x_scaler"].mean_,features(groups["train"]["state"]).mean(axis=0))
    before=predict(model,groups["test"])
    restored=pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(before,predict(restored,groups["test"]))
    repeated=train(groups["train"],groups["val"],"hybrid",11,config)
    np.testing.assert_allclose(before,predict(repeated,groups["test"]),atol=1e-12)
    assert before.shape==groups["test"]["two_body"].shape
    assert np.isfinite(before).all()

def test_known_metrics():
    target=np.array([[7000,0,0,0,7.5,0.],[7000,0,0,0,7.5,0.]])
    prediction=target[:,:3]+np.array([[3,4,0],[0,0,0]])
    metrics=position_metrics(prediction,target)
    assert metrics["mae_3d_km"]==2.5
    assert metrics["rmse_3d_km"]==pytest.approx(np.sqrt(12.5))
    assert metrics["p95_km"]==pytest.approx(4.75)
    assert metrics["radial_mae_km"]==1.5
    assert metrics["along_track_mae_km"]==2
    assert metrics["cross_track_mae_km"]==0

def test_solver_convergence():
    state,_=initial_state(np.random.default_rng(4))
    normal=propagate(state,[21600])
    tight=propagate(state,[21600],rtol=1e-12,atol=1e-14,max_step=30)
    assert np.linalg.norm(normal[0,:3]-tight[0,:3])<1e-4

def test_pipeline_outputs(tmp_path, sample):
    from src.run_benchmark import run
    config,_=sample
    output=tmp_path/"results"
    summary=run(config,output,tmp_path/"models")
    assert len(summary)==50
    for name in ("metrics.json","metrics_by_seed.csv","metrics_summary.csv","predictions.npz",
                 "dataset.npz","split_manifest.json","run_metadata.json","errors.csv","results.md"):
        assert (output/name).stat().st_size>0
    from PIL import Image
    images=list((output/"figures").glob("*.png"))
    assert len(images)==5
    for path in images:
        with Image.open(path) as image:
            assert image.width>1000 and image.height>700
    import json
    rows = json.loads((output/"metrics.json").read_text())["by_seed"]
    with np.load(output/"dataset.npz") as dataset, np.load(output/"predictions.npz") as saved:
        for row in rows:
            key = f'{row["split"]}__{row["model"].replace(" ", "_")}__{row["seed"]}'
            index = list(config.horizons_s).index(row["horizon_min"]*60)
            error = np.linalg.norm(saved[key][:,index] - dataset[row["split"]+"_target"][:,index,:3],axis=1)
            assert row["mae_3d_km"] == pytest.approx(error.mean())
            assert row["rmse_3d_km"] == pytest.approx(np.sqrt(np.mean(error**2)))
            assert row["p95_km"] == pytest.approx(np.percentile(error,95))
