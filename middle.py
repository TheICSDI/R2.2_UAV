from time import sleep
import glob
import os
import json
import uuid
import multiprocessing as mp
from typing import List
import logging
import parameters as par

os.environ["AGENT"] = "local"

from aerialist.px4.drone_test import DroneTest
from aerialist.px4.obstacle import Obstacle
from testcase import TestCase

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def build_obstacles(raw) -> List[Obstacle]:
    obs = []
    for ob in raw:
        size = Obstacle.Size(
            l=ob["size"][0],
            w=ob["size"][1],
            h=ob["size"][2],
        )

        position = Obstacle.Position(
            x=ob["position"][0],
            y=ob["position"][1],
            z=ob["position"][2],
            r=ob["rotation"],
        )

        obs.append(Obstacle(position=position, size=size))
    return obs

def error_json(name, obstacles) -> None:
    error_path = os.path.join(par.RESULTS_DIR, f"{name}_result.json")
    with open(error_path, "w") as fp:
        json.dump({
            "name": name,
            "min_dist": None,
            "duration": None,
            "num_obstacles": len(obstacles),
            "trajectory": []
        }, fp, indent=2)

def run_test_in_process(tc, name, obstacles) -> None:
    try:
        trajectory = tc.execute()
        distances = tc.get_distances()
        duration = trajectory.positions[-1].timestamp - trajectory.positions[0].timestamp
        tc.plot()
        result = {
            "name": name,
            "min_dist": min(distances),
            "duration": duration,
            "num_obstacles": len(obstacles),
            "trajectory": [[p.timestamp, p.x, p.y, p.z] for p in trajectory.positions]
        }

        result_path = os.path.join(par.RESULTS_DIR, f"{name}_result.json")
        with open(result_path, "w") as fp:
            json.dump(result, fp, indent=2)

        logging.info(f"[+] {name} - YAML saved and test executed.")

    except Exception as e:
        logging.error(f"[-] {name} - Failed during test execution: {e}")
        error_json(name, obstacles)

def convert_and_run(jfile) -> None:
    with open(jfile, "r") as fp:
        data = json.load(fp)

    name = data.get("name", f"test_{uuid.uuid4().hex[:8]}")
    obstacles = build_obstacles(data["obstacles"])

    # Load base case study (with mission plan)
    base = DroneTest.from_yaml(par.BASE_YAML)
    # Construct the test case using the SBFT format
    tc = TestCase(base, obstacles)

    out_path = os.path.join(par.YAML_DIR, f"{name}.yaml")
    tc.save_yaml(out_path)

    sleep(par.COOL)
    p = mp.Process(target=run_test_in_process, args=(tc, name, obstacles))
    p.start()
    p.join(par.TIMEOUT)

    if p.is_alive():
        logging.info(f"[-] {name} - Timeout. Killing test process.")
        p.terminate()
        p.join()
        error_json(name, obstacles)

def process_forever():
    logging.info(f"[+] middle.py daemon — polling every {par.POLL}s")
    while True:
        # grab every *.json currently present
        for path in glob.glob(f"{par.INPUT_DIR}/*.json"):
            try:
                convert_and_run(path)
            finally:
                # delete genome file so it’s never re-processed
                try:
                    os.remove(path)
                except OSError as e:
                    logging.error(f"[-] Failed during delation: {e}")
        sleep(par.POLL)

if __name__ == "__main__":
    os.makedirs(par.YAML_DIR, exist_ok=True)
    os.makedirs(par.RESULTS_DIR, exist_ok=True)
    # necessary folder if we mount the shered folder into /src/aerialist/results
    os.makedirs("./results/logs", exist_ok=True)
    process_forever()
