import parameters as par
import utils
import os
import numpy as np
import atexit
import shutil
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.survival import Survival
from pymoo.core.callback import Callback

class UAVTestGenomeProblem(Problem):
    def __init__(self):
        super().__init__(n_var=par.TOTAL_VARS, n_obj=3, n_constr=0, xl=par.XL, xu=par.XU)
        self.archive = []
        self.gen_idx = 0
        self.records  = []

    def _evaluate(self, X, out, *args, **kwargs):

        # ensure population order = fitness order
        names_in_order = []
        # only the ones we actually simulate
        pending_names = []
        # rows with early-detected invalids
        penalties = []
        archive_traj = self.archive

        # generate genomes / basic checks
        for genome in X:
            genome = genome.astype(float).tolist()
            data_dict, name = utils.genome_to_data_dict(genome)
            obstacles = data_dict["obstacles"]
            names_in_order.append(name)

            # check validity
            valid = len(obstacles) > 0
            for i in range(len(obstacles)):
                for j in range(i + 1, len(obstacles)):
                    if utils.is_overlapping(obstacles[i], obstacles[j]):
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                penalties.append(name)          # remember to penalise later
                continue

            utils.save_genome_data(data_dict, name)
            pending_names.append(name)

        # wait until all sims finish
        results = utils.wait_for_all(pending_names, par.RESULTS_DIR)

        # build fitness array in the same order
        fitness_rows = []
        for vec, name in zip(X, names_in_order):
            vec = vec.astype(float)
            if name in penalties:
                fitness_rows.append([par.INVALID, par.INVALID, par.INVALID])
                continue

            res = results.get(name)
            if res and res["min_dist"] is not None:

                # larger diff -> larger dist cut, temp for 0 dist
                dist = float(res["min_dist"]) * 100 if float(res["min_dist"]) != 0 else 0.1
                dur = float(res["duration"]) / 1e6      # form nanosecond to second

                # DTW diversity
                track_xy = utils.traj_xy(utils.downsample(res["trajectory"]))
                nov = utils.dtw_novelty(track_xy, archive_traj)
                # persist for next generations
                archive_traj.append(track_xy)
                if len(archive_traj) > par.ARCHIVE_MAX:
                    archive_traj.pop(0) # FIFO

                fitness_rows.append([dist, -nov, dur])
                yaml_path = os.path.join(par.YAML_DIR, f"{name}.yaml")
                self.records.append((dist, -nov, dur, name, yaml_path))

            else:
                fitness_rows.append([par.INVALID, par.INVALID, par.INVALID])


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
            np.floor(dist / par.TLX_EPS_ABS[0]),
            np.floor(nov_neg / par.TLX_EPS_ABS[1]),
            np.floor(dur / par.TLX_EPS_ABS[2]),
        )

    unique = {}
    for rec in records:
        _, _, _, name, _ = rec
        if name not in unique or rank_key(rec) < rank_key(unique[name]):
            unique[name] = rec

    best = sorted(unique.values(), key=rank_key)[:par.TOP_K]

    for dist, nov_neg, dur, name, src in best:
        dst = os.path.join(par.FINAL_DIR, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"[+] kept {name}.yaml  dist={dist/100:.3f}  nov={-nov_neg:.2f}  dur={dur:.1f}s")

    print(f"[+] Copied {len(best)} YAML tests to {par.FINAL_DIR}")

if __name__ == "__main__":
    os.makedirs(par.OUTPUT_DIR, exist_ok=True)
    os.makedirs(par.FINAL_DIR, exist_ok=True)
    # Start the Aerialist-side as a daemon
    utils.start_middle_container()
    # Ensure container stops atExit
    atexit.register(utils.stop_middle_container)

    problem = UAVTestGenomeProblem()
    survival = TolerantLexicoSurvival(par.TLX_EPS_ABS, par.TLX_EPS_REL)
    diagnostics = Diagnostics(par.TLX_EPS_ABS)
    algorithm = NSGA2(pop_size=par.POP_SIZE, survival=survival, eliminate_duplicates=True)
    algorithm.survival = survival
    termination = get_termination("n_gen", par.N_GENERATIONS)

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=par.SEED,
        save_history=True,
        verbose=False,
        callback=diagnostics,
    )

    print("\nOptimization complete.")
    best_tests(problem.records)
