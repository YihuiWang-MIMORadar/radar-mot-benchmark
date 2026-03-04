import numpy as np

def simulate_intersection(cfg):
    """City intersection synthetic scenario.
    Returns: list of frames, each frame is a dict with keys: t, meas, gt_pos, gt_id
    """
    rng = np.random.default_rng(cfg["seed"])
    T, dt = cfg["T"], cfg["dt"]

    n_targets = int(rng.integers(cfg["n_targets_min"], cfg["n_targets_max"] + 1))
    center = np.array([0.0, 0.0])
    L = cfg["max_range"] * 0.9

    edges = np.array(cfg["spawn_edges"])
    spawn = rng.choice(edges, size=n_targets, replace=True)

    pos = np.zeros((n_targets, 2), dtype=float)
    vel = np.zeros((n_targets, 2), dtype=float)

    lane_off = rng.normal(0.0, cfg["lane_offset_std"], size=n_targets)
    speed = np.clip(rng.normal(cfg["speed_mean"], cfg["speed_std"], size=n_targets), 2.0, 25.0)

    for i, s in enumerate(spawn):
        if s == "N":      # north -> south
            pos[i] = np.array([lane_off[i],  L])
            vel[i] = np.array([0.0, -speed[i]])
        elif s == "S":    # south -> north
            pos[i] = np.array([lane_off[i], -L])
            vel[i] = np.array([0.0,  speed[i]])
        elif s == "E":    # east -> west
            pos[i] = np.array([ L, lane_off[i]])
            vel[i] = np.array([-speed[i], 0.0])
        else:             # "W": west -> east
            pos[i] = np.array([-L, lane_off[i]])
            vel[i] = np.array([ speed[i], 0.0])

    turn = rng.random(n_targets) < cfg["turn_prob"]
    turned = np.zeros(n_targets, dtype=bool)

    frames = []
    gt_id = np.arange(n_targets, dtype=int)

    for t in range(T):
        pos = pos + vel * dt

        # turn once near the center
        near = (np.linalg.norm(pos - center, axis=1) < 8.0)
        for i in range(n_targets):
            if turn[i] and near[i] and (not turned[i]):
                vx, vy = vel[i]
                if rng.random() < 0.5:
                    vel[i] = np.array([-vy, vx])   # left turn
                else:
                    vel[i] = np.array([vy, -vx])   # right turn
                turned[i] = True

        in_range = (np.linalg.norm(pos, axis=1) <= cfg["max_range"])
        gt_pos = pos[in_range]
        gt_ids = gt_id[in_range]

        meas_list = []
        for p in gt_pos:
            if rng.random() < cfg["p_det"]:
                dist = float(np.linalg.norm(p))
                sigma = cfg["range_noise_base"] + cfg["range_noise_slope"] * dist
                z = p + rng.normal(0.0, sigma, size=2)
                meas_list.append(z)

        n_clutter = int(rng.poisson(cfg["clutter_rate"]))
        if n_clutter > 0:
            r = cfg["max_range"] * np.sqrt(rng.random(n_clutter))
            theta = 2 * np.pi * rng.random(n_clutter)
            clutter = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        else:
            clutter = np.zeros((0, 2))

        meas_gt = np.array(meas_list).reshape(-1, 2) if len(meas_list) else np.zeros((0, 2))
        meas = np.vstack([meas_gt, clutter])

        frames.append({
            "t": t,
            "meas": meas,
            "gt_pos": gt_pos,
            "gt_id": gt_ids,
        })

    return frames