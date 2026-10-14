import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import corner
from eobviz.events.gw150914 import GW150914
from eobviz.events.gw250114 import GW250114
P = ["chirp_mass","mass_ratio","a_1","a_2","tilt_1","tilt_2","phi_12","phi_jl",
     "theta_jn","iota","psi","phase","ra","dec","luminosity_distance","geocent_time","chi_eff","chi_p"]
out = sys.argv[1]
for E, extra in ((GW150914, ["eccentricity","mean_per_ano"]), (GW250114, [])):
    p = E().load_posterior(); keys = P + extra
    x = np.column_stack([np.asarray(p[k]) for k in keys])
    x[:, keys.index("geocent_time")] -= E.reference_time
    fig = corner.corner(x, labels=keys, bins=40, smooth=0.8, quantiles=[.05,.5,.95],
                        show_titles=True, title_fmt=".3g", label_kwargs=dict(fontsize=14))
    fig.suptitle(f"{E.__name__} ({E.approximant}), N={len(x)}", fontsize=30)
    fig.savefig(f"{out}/corner_{E.__name__}.png", dpi=60); print(E.__name__, "ok")
    th = np.asarray(p["theta_jn"]); print("  P(theta_jn<pi/2) =", (th<np.pi/2).mean())
