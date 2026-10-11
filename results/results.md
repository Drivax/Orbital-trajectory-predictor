Mean 3D error in **km** on held-out ID satellites; OOD column is at the longest horizon.
ML entries are mean ± sample std across training seeds, not confidence intervals.

| Model | 5 min | 30 min | 60 min | 180 min | 360 min | OOD |
|---|---:|---:|---:|---:|---:|---:|
| Constant Velocity | 372 | 1.21e+04 | 3.43e+04 | 8.66e+04 | 1.71e+05 | 1.57e+05 |
| Two Body | 0.517 | 12.8 | 34.1 | 92.3 | 185 | 202 |
| Two Body + J2 | 9.28e-06 | 0.000378 | 0.00335 | 0.0362 | 0.144 | 0.000224 |
| Pure ML | 0.487 ± 0.016 | 1.51 ± 0.24 | 2.71 ± 0.24 | 7.39 ± 0.79 | 14.8 ± 1.4 | 9e+03 ± 1.1e+02 |
| Physics + ML residual | 0.00607 ± 0.00029 | 0.146 ± 0.0045 | 0.444 ± 0.025 | 1.33 ± 0.11 | 3.73 ± 0.42 | 360 ± 21 |

### Key findings

- At 360 min on ID, the hybrid changes two-body error by -98.0% (185 → 3.73 km).
- At 360 min on OOD, the hybrid changes two-body error by +78.3% (202 → 360 km).
- The lowest ID mean error at 360 min is obtained by **Two Body + J2**.
- Pure ML has 14.8 km ID mean error at 360 min; no superiority over physics is assumed.
