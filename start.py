from time import sleep, time
import os
import uuid
import json
import subprocess
import numpy as np
from fastdtw import fastdtw
import atexit
import shutil
from scipy.spatial.distance import euclidean
from shapely.geometry import Polygon
from shapely import affinity
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.survival import Survival
from pymoo.core.callback import Callback


# Paths
OUTPUT_DIR = "/mnt/c/UAV/generated_genomes"
RESULTS_DIR = "/mnt/c/UAV/results"
YAML_DIR = "/mnt/c/UAV/generated_tests"
FINAL_DIR = "./final_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# SBFT constraints
NUM_OBS = 3
VARS_PER_OBS = 6
TOTAL_VARS = NUM_OBS * VARS_PER_OBS
XL = np.array([-40, 10, 2, 2, 10, 0] * NUM_OBS)
XU = np.array([30,  40, 20, 20, 25, 90] * NUM_OBS)

# MAX penalties value
INVALID = 1e6

# Novelty archive parameters
ARCHIVE_MAX = 750            # cap on stored trajectories
NOVELTY_DOWNSAMPLE_K = 100   # samples for DTW basis

# Tolerant Lexicographic parameters
# Absolute tolerances
TLX_EPS_ABS = np.array([
    2,      # distance
    0.5,    # novelty
    1.0,    # duration
])
# Relative tolerances
TLX_EPS_REL = np.array([
    0.0,    # distance
    0.0,    # novelty
    0.0,    # duration
])

# Evolution parameters
POP_SIZE = 3
N_GENERATIONS = 2
SEED = 42

# # of YAMLs to keep
TOP_K = 15

# DOCKER
DOCKER_IMAGE    = "5ff51322acba"        # Aerialist image ID / tag
CONTAINER_NAME  = "aerialist-middle"
SHARED_HOST_DIR = "/mnt/c/UAV"
SHARED_CONT_DIR = "/tests"
COPY_SCRIPT     = "copy.sh"

def start_middle_container():
    """Launch docker in detached mode"""
    cmd = [
        "docker", "run", "--rm", "-d",
        "--name",     CONTAINER_NAME,
        "--gpus",     "all",
        "-v",         f"{SHARED_HOST_DIR}:{SHARED_CONT_DIR}",
        "-it",        DOCKER_IMAGE,
        "bash", "-c", f"{SHARED_CONT_DIR}/{COPY_SCRIPT}"
    ]
    print("[+] Starting middle.py in Docker container...")
    subprocess.check_call(cmd)
    print(f"[+] Container '{CONTAINER_NAME}' started.")

def stop_middle_container():
    print(f"[+] Stopping container '{CONTAINER_NAME}'...")
    subprocess.call(["docker", "stop", CONTAINER_NAME])
    print("[+] Container stopped.")

def genome_to_data_dict(genome, name_prefix="test"):
    name = f"{name_prefix}_{uuid.uuid4().hex[:8]}"
    obstacles = []

    for i in range(NUM_OBS):
        base = i * VARS_PER_OBS
        x, y = genome[base], genome[base + 1]
        l, w, h = genome[base + 2], genome[base + 3], genome[base + 4]
        r = genome[base + 5]

        # Ensure h > 10 and round values
        if h >= 10:
            obstacle = {
                "position": [round(float(x), 2), round(float(y), 2), 0.0],
                "size": [round(float(l), 2), round(float(w), 2), round(float(h), 2)],
                "rotation": round(float(r % 360), 2)
            }
            obstacles.append(obstacle)


    return {"name": name, "obstacles": obstacles}, name

def save_genome_data(data_dict, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    return path

def wait_for_all(names, results_dir, per_file_timeout=900, poll=60):
    pending = set(names)
    collected = {}
    deadline = time() + per_file_timeout * len(names)

    while pending and time() < deadline:
        ready = []
        for n in list(pending):
            path = os.path.join(results_dir, f"{n}_result.json")
            if os.path.exists(path):
                with open(path) as fp:
                    collected[n] = json.load(fp)
                ready.append(n)
        pending.difference_update(ready)

        if pending:
            sleep(poll)

    for n in pending:
        collected[n] = None
    return collected

def downsample(traj, k = NOVELTY_DOWNSAMPLE_K):
    if len(traj) <= k:
        return traj

    idx = np.linspace(0, len(traj) - 1, k, dtype=int)
    return [traj[i] for i in idx]

def traj_xy(traj):
    """Strip time & z -> list[(x,y)]."""
    return [(p[1], p[2]) for p in traj]

def dtw_distance_xy(traj_a, traj_b):
    """FastDTW between two XY poly-lines -> scalar distance."""
    a = np.asarray(traj_a, dtype=float)
    b = np.asarray(traj_b, dtype=float)
    distance, _ = fastdtw(a, b, dist=euclidean)
    return float(distance)

def dtw_novelty(traj, archive):
    """Min DTW distance between traj and any in archive; large -> novel."""
    if not archive:
        return 0
    return min(dtw_distance_xy(traj, ref) for ref in archive)


class UAVTestGenomeProblem(Problem):
    def __init__(self):
        super().__init__(n_var=TOTAL_VARS, n_obj=3, n_constr=0, xl=XL, xu=XU)
        self.archive = []
        self.gen_idx = 0
        self.records  = []

    def _evaluate(self, X, out, *args, **kwargs):

        def is_overlapping(obs1, obs2):
            def poly(obs):
                l, w = obs["size"][:2]
                rect = Polygon([(-l/2,-w/2),(l/2,-w/2),(l/2,w/2),(-l/2,w/2)])
                r = affinity.rotate(rect, obs["rotation"], origin=(0,0))
                return affinity.translate(r, obs["position"][0], obs["position"][1])
            return poly(obs1).intersects(poly(obs2))

        names_in_order = []     # ensure population order = fitness order
        pending_names = []     # only the ones we actually simulate
        penalties = []     # rows with early-detected invalids
        archive_traj = self.archive

        # generate genomes / basic checks
        for genome in X:
            genome = genome.astype(float).tolist()
            data_dict, name = genome_to_data_dict(genome)
            obstacles = data_dict["obstacles"]
            names_in_order.append(name)

            # check validity
            valid = len(obstacles) > 0
            for i in range(len(obstacles)):
                for j in range(i + 1, len(obstacles)):
                    if is_overlapping(obstacles[i], obstacles[j]):
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                penalties.append(name)          # remember to penalise later
                continue

            save_genome_data(data_dict, name)
            pending_names.append(name)

        # wait until all sims finish
        results = wait_for_all(pending_names, RESULTS_DIR)

        # build fitness array in the same order
        fitness_rows = []
        for vec, name in zip(X, names_in_order):
            vec = vec.astype(float)
            if name in penalties:
                fitness_rows.append([INVALID, INVALID, INVALID])
                continue

            res = results.get(name)
            if res and res["min_dist"] is not None:

                # f1 = res["min_dist"] if self.gen_idx < 3 else res["min_dist"] * 3
                # f1 = np.exp(2 * f1)
                dist = float(res["min_dist"]) * 100     # larger diff -> larger dist cut
                dur = float(res["duration"]) / 1e6      # form nanosecond to second

                # DTW diversity
                track_xy = traj_xy(downsample(res["trajectory"]))
                nov = dtw_novelty(track_xy, archive_traj)
                # persist for next generations
                archive_traj.append(track_xy)
                if len(archive_traj) > ARCHIVE_MAX:
                    archive_traj.pop(0) # FIFO

                fitness_rows.append([dist, -nov, dur])
                yaml_path = os.path.join(YAML_DIR, f"{name}.yaml")
                self.records.append((dist, -nov, dur, name, yaml_path))

            else:
                fitness_rows.append([INVALID, INVALID, INVALID])


        self.archive = archive_traj
        out["F"] = np.array(fitness_rows)

class TolerantLexicoSurvival(Survival):
    """Lexicographic ordering with absolute/relative tolerances.

    Order of priority (highest to lowest):
        1. distance
        2. -novelty
        3. duration

    Tolerance logic:
        Two values f_a, f_b for objective j are considered equivalent if
        |f_a - f_b| <= max(eps_abs[j], eps_rel[j] * max(|f_a|, |f_b|)).
        Equivalent values are placed in the same bin so sorting proceeds to
        the next objective to break ties.
    """
    def __init__(self, eps_abs, eps_rel):
        super().__init__(filter_infeasible=True)
        self.eps_abs = np.array(eps_abs)
        self.eps_rel = np.array(eps_rel)

    def _do(self, problem, pop, n_survive, **kwargs):
        F = pop.get("F")  # shape (N, M)
        N, M = F.shape
        # Build bins per objective
        bins = []
        for j in range(M):
            col = F[:, j]
            # Compute adaptive tolerance for each pair by using max value; we approximate with global scaling using max magnitude
            scale = np.maximum(np.abs(col).max(), 1e-12)
            tol = max(self.eps_abs[j], self.eps_rel[j] * scale)
            if tol <= 0:
                b = col
            else:
                # values within tol end in same bin
                b = np.floor(col / tol)
            bins.append(b)

        # from lowest priority objective to highest
        order = np.arange(N)
        for j in reversed(range(M)):
            order = order[np.argsort(bins[j][order], kind="stable")]

        # Assign ranks (0 = best) for ALL individuals
        ranks = np.empty(N, dtype=int)
        ranks[order] = np.arange(N)
        for idx, ind in enumerate(pop):
            ind.set("rank", ranks[idx])
            # uniform crowding
            ind.set("crowding", 0.0)  # shouldn't be used since ranks differ

        return pop[order[:n_survive]]

class Diagnostics(Callback):
    def __init__(self, eps_abs: np.ndarray):
        super().__init__()
        self.eps_abs = eps_abs
        self.history = []

    def notify(self, algorithm):
        pop = algorithm.pop
        F = pop.get("F")
        dist = F[:,0]
        novelty = F[:,1]
        duration = F[:,2]

        best_dist = float(dist.min())
        within = np.sum(dist <= best_dist + self.eps_abs[0])
        log_entry = {
            "gen": algorithm.n_gen,
            "best_dist": best_dist,
            "within_dist_tol": int(within),
            "nov_mean": float(novelty.mean()),
            "nov_max": float(novelty.max()),
            "dur_mean": float(duration.mean()),
            "dur_min": float(duration.min()),
        }
        self.history.append(log_entry)
        print(
            f"Gen {algorithm.n_gen:03d} | best_dist={best_dist:.4f} | in_tier={within:02d} | "
            f"nov(mean/max)={log_entry['nov_mean']:.2f}/{log_entry['nov_max']:.2f} | "
            f"dur(min/mean)={log_entry['dur_min']:.1f}/{log_entry['dur_mean']:.1f}"
        )

def best_tests(records):

    def rank_key(rec):
        dist, nov_neg, dur, *_ = rec
        return (
            np.floor(dist / TLX_EPS_ABS[0]),
            np.floor(nov_neg / TLX_EPS_ABS[1]),
            np.floor(dur / TLX_EPS_ABS[2]),
        )

    unique = {}
    for rec in records:
        _, _, _, name, _ = rec
        if name not in unique or rank_key(rec) < rank_key(unique[name]):
            unique[name] = rec

    best = sorted(unique.values(), key=rank_key)[:TOP_K]

    for dist, nov_neg, dur, name, src in best:
        dst = os.path.join(FINAL_DIR, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"[+] kept {name}.yaml  dist={dist/100:.3f}  nov={-nov_neg:.2f}  dur={dur:.1f}s")

    print(f"[+] Copied {len(best)} YAML tests to {FINAL_DIR}")

if __name__ == "__main__":
    # Start the Aerialist-side as a daemon
    start_middle_container()
    # Ensure container stops atExit
    atexit.register(stop_middle_container)

    problem = UAVTestGenomeProblem()
    survival = TolerantLexicoSurvival(TLX_EPS_ABS, TLX_EPS_REL)
    diagnostics = Diagnostics(TLX_EPS_ABS)
    algorithm = NSGA2(pop_size=POP_SIZE, survival=survival, eliminate_duplicates=True)
    algorithm.survival = survival
    termination = get_termination("n_gen", N_GENERATIONS)

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=SEED,
        save_history=True,
        verbose=False,
        callback=diagnostics,
    )

    print("\nOptimization complete.")
    best_tests(problem.records)
