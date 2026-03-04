import matplotlib.pyplot as plt
import numpy as np
from src.simulate_intersection import simulate_intersection

CFG = {
  "seed": 42,
  "T": 200,
  "dt": 0.1,
  "n_targets_min": 6,
  "n_targets_max": 12,
  "spawn_edges": ["N","S","E","W"],
  "speed_mean": 10.0,
  "speed_std": 3.0,
  "turn_prob": 0.25,
  "lane_offset_std": 1.0,
  "p_det": 0.85,
  "clutter_rate": 25,
  "range_noise_base": 0.8,
  "range_noise_slope": 0.02,
  "max_range": 120.0,
}

def main():
    frames = simulate_intersection(CFG)

    tracks = {}
    for fr in frames:
        for p, tid in zip(fr["gt_pos"], fr["gt_id"]):
            tracks.setdefault(int(tid), []).append(p)

    plt.figure()
    for tid, pts in tracks.items():
        pts = np.array(pts)
        plt.plot(pts[:, 0], pts[:, 1], linewidth=1)

    meas = frames[-1]["meas"]
    if len(meas) > 0:
        plt.scatter(meas[:, 0], meas[:, 1], s=10)

    plt.axis("equal")
    plt.title("Intersection GT trajectories + last-frame measurements")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()

    out = "results/figures/gt_intersection.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()