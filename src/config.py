from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class Config:
    data_seed: int = 2026
    seeds: tuple = (11, 22, 33)
    horizons_s: tuple = (300, 1800, 3600, 10800, 21600)
    train_satellites: int = 96
    val_satellites: int = 24
    test_satellites: int = 24
    ood_satellites: int = 24
    origins_per_satellite: int = 12
    origin_spacing_s: int = 25200
    epochs: int = 600
    patience: int = 60
    hidden: tuple = (96, 96)
    rtol: float = 1e-10
    atol: float = 1e-12
    max_step_s: float = 90.0

    def as_dict(self):
        return asdict(self)
