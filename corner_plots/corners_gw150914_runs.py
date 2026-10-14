"""Corner plots of every GW150914 bilby result in a directory (default: the 15 Hz TEOBResumS runs)."""
import sys
from pathlib import Path
import h5py, numpy as np, matplotlib; matplotlib.use("Agg")
import corner

DIR = Path("/home/jacopo/Documents/gw150914_decade/dat/post_bilby/15_hz_post")
T0 = 1126259462.4
P = ["chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "chi_1", "chi_2",
     "theta_jn", "iota", "psi", "phase", "ra", "dec", "luminosity_distance", "geocent_time",
     "chi_eff", "chi_p", "eccentricity", "mean_per_ano"]

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
for f in sorted(DIR.glob("*_result.hdf5")):
    run = f.name.split("_15_")[0].removeprefix("gw150914_")
    with h5py.File(f, "r") as h:
        g = h["posterior"]
        keys = [k for k in P if k in g and np.ptp(g[k][()]) > 0]  # skip absent/fixed params
        x = np.column_stack([g[k][()] for k in keys])
    x[:, keys.index("geocent_time")] -= T0
    fig = corner.corner(x, labels=keys, bins=40, smooth=0.8, quantiles=[.05, .5, .95],
                        show_titles=True, title_fmt=".3g", label_kwargs=dict(fontsize=14))
    fig.suptitle(f"GW150914 {run}, N={len(x)}", fontsize=30)
    fig.savefig(out / f"corner_GW150914_{run}.png", dpi=60)
    th = x[:, keys.index("theta_jn")]
    print(f"{run:22s} N={len(x):6d}  P(theta_jn<pi/2)={np.mean(th < np.pi/2):.3f}  "
          f"theta_jn median={np.median(th):.2f}")
