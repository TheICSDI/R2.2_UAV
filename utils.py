import parameters as par
from subprocess import check_call, call
from uuid import uuid4
from os import path
import json
import numpy as np
from time import sleep, time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from shapely.ops import substring
from shapely.geometry import Polygon, LineString
from shapely import affinity


def start_middle_container():
    """Launch docker in detached mode"""
    cmd = [
        "docker", "run", "--rm", "-d",
        "--name",     par.CONTAINER_NAME,
        "--gpus",     "all",
        "-v",         f"{par.SHARED_HOST_DIR}:{par.SHARED_CONT_DIR}",
        "-it",        par.DOCKER_IMAGE,
        "bash", "-c", f"cp {par.CASE_STUDIES_DIR} {par.DESTINATION_DIR} -r &&\
        cp {par.MIDDLE_PATH} {par.DESTINATION_DIR} &&\
        cp {par.TESTCASE_PATH} {par.DESTINATION_DIR} &&\
        cp {par.UTILS_PATH} {par.DESTINATION_DIR} &&\
        cp {par.PAR_PATH} {par.DESTINATION_DIR} &&\
        python3 {par.DESTINATION_DIR}/{par.MIDDLE_PY}"
    ]
    print("[+] Starting middle.py in Docker container...")
    check_call(cmd)
    print(f"[+] Container '{par.CONTAINER_NAME}' started.")

def stop_middle_container():
    """Stop docker container"""
    print(f"[+] Stopping container '{par.CONTAINER_NAME}'...")
    call(["docker", "stop", par.CONTAINER_NAME])
    print("[+] Container stopped.")

def genome_to_data_dict(genome, name_prefix="test"):
    """Build obstacles and prepare the dict object for json"""
    name = f"{name_prefix}_{uuid4().hex[:8]}"
    obstacles = []

    for i in range(par.NUM_OBS):
        base = i * par.VARS_PER_OBS
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
    """Build the json with the date generated from genomes"""
    genome_path = path.join(par.OUTPUT_DIR, f"{name}.json")
    with open(genome_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    return genome_path

def wait_for_all(names, results_dir, per_file_timeout=par.TIMEOUT + 60, poll=60):
    """Waits that all generated tests are executed"""
    pending = set(names)
    collected = {}
    deadline = time() + per_file_timeout * len(names)

    while pending and time() < deadline:
        ready = []
        for n in list(pending):
            result_path = path.join(results_dir, f"{n}_result.json")
            if path.exists(result_path):
                with open(result_path) as fp:
                    collected[n] = json.load(fp)
                ready.append(n)
        pending.difference_update(ready)

        if pending:
            sleep(poll)

    for n in pending:
        collected[n] = None
    return collected

def is_overlapping(obs1, obs2):
    """Check if obstacle are overlapping"""
    def poly(obs):
        l, w = obs["size"][:2]
        rect = Polygon([(-l/2,-w/2),(l/2,-w/2),(l/2,w/2),(-l/2,w/2)])
        r = affinity.rotate(rect, obs["rotation"], origin=(0,0))
        return affinity.translate(r, obs["position"][0], obs["position"][1])
    return poly(obs1).intersects(poly(obs2))

def downsample(traj, k = par.NOVELTY_DOWNSAMPLE_K):
    """Approximate the trajectory"""
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
