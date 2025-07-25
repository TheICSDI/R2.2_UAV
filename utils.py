import parameters as par
from subprocess import check_call, call
from uuid import uuid4
from os import path
import json
import numpy as np
from time import sleep, time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from shapely.geometry import LineString
from shapely.ops import substring


def start_middle_container():
    """Launch docker in detached mode"""
    cmd = [
        "docker", "run", "--rm", "-d",
        "--name",     par.CONTAINER_NAME,
        "--gpus",     "all",
        "-v",         f"{par.SHARED_HOST_DIR}:{par.SHARED_CONT_DIR}",
        "-it",        par.DOCKER_IMAGE,
        "bash", "-c", f"cp {par.CASE_STUDIES_DIR} {par.DESTINATION_DIR} \
        && cp {par.MIDDLE_PATH} {par.DESTINATION_DIR} \
        && cp {par.TESTCASE_PATH} {par.DESTINATION_DIR} \
        && export PYTHONPATH={par.DESTINATION_DIR}:$PYTHONPATH \
        && python3 {par.DESTINATION_DIR}/{par.MIDDLE_PY}"
    ]
    print("[+] Starting middle.py in Docker container...")
    check_call(cmd)
    print(f"[+] Container '{par.CONTAINER_NAME}' started.")

def stop_middle_container():
    print(f"[+] Stopping container '{par.CONTAINER_NAME}'...")
    call(["docker", "stop", par.CONTAINER_NAME])
    print("[+] Container stopped.")

def genome_to_data_dict(genome, name_prefix="test"):
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
    genome_path = path.join(par.OUTPUT_DIR, f"{name}.json")
    with open(genome_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    return genome_path

def wait_for_all(names, results_dir, per_file_timeout=par.TIMEOUT + 60, poll=60):
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

def downsample(traj, k = par.NOVELTY_DOWNSAMPLE_K):
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

PLAN_FILE = "/mnt/c/UAV/temp/case_studies/mission2.plan"
# tube radius
PAD_M = 10.0

def ll_to_enu(lat, lon, lat0, lon0):
    # equatorial radius
    R = 6378137.0
    d_north = (lat - lat0) * np.pi/180 * R
    d_east  = (lon - lon0) * np.pi/180 * R * np.cos(np.deg2rad(lat0))
    return d_north, d_east

def load_route(plan_path: str):
    # Usually just 16 (waypoint) 21 (land) 22 (takeoff)
    COMMANDS = [16, 17, 18, 19, 21, 22, 25, 31, 82, 84, 85]
    TRIM_FRAC = 0.2
    # No command 82 found in the available tests but it shouldn't be treated like the other

    with open(plan_path, 'r') as fp:
        data = json.load(fp)

    mission = data["mission"]

    waypoints = []
    for item in mission["items"]:
        # this assumes that 22(16)^*21 as all the sample tests
        if item["command"] not in COMMANDS:
            continue

        lat, lon = item["params"][4:6]

        waypoints.append((lat, lon))
    waypoints = [ll_to_enu(lat, lon, waypoints[0][0], waypoints[0][1]) for lat, lon in waypoints]

    full_route = LineString(waypoints)
    tot_len = full_route.length
    start_d = TRIM_FRAC * tot_len
    end_d   = (1.0 - TRIM_FRAC) * tot_len
    return substring(full_route, start_d, end_d)
