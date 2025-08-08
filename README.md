# Project R2.2 - Search‑Based UAV Test‑Generator

This repository implements a genetic algorithm (NSGA-II) that automatically builds challenging obstacle-placement tests for the PX4 Avoidance stack using the Aerialist simulation bench.
The workflow is split in two coordinated processes that communicates with a shared folder:

1. `start.py` - drives the multi‑objective search.
2. `middle.py` - executes each candidate test inside an Aerialist Docker container and reports the results back to the search.

## 1  Prerequisites

| Tool         | Version / Notes       |
| ------------ | --------------------- |
| Python       | 3.13.5                |
| pip packages | listed in requirement |

Install the Python dependencies:

```bash
pip install -r requirements.txt
```


Some additional resources required are hosted in the official SBFT‑2024 competition repository. Download `testcase.py` and the folder `case_studies` from [here](https://github.com/skhatiri/UAV-Testing-Competition/tree/master/snippets).

### 1.1 Shared folder set-up
The shared folder should contain:
- both `start.py` and `middle.py`.
- both competition assets `testcase.py` and `case_studies` folder.
- both auxiliary files `parameters.py` and `utils.py`. 
The path of the shared folder can be found in `parameters.py`
## 3  Docker image for Aerialist

`middle.py` expects a pre-built Aerialist image containing PX4, Gazebo and all ROS1/ROS2 dependencies.

1. Pull the official image:
```bash
docker pull skhatiri/aerialist
```
    
2. Update `parameters.py`
    
```python
DOCKER_IMAGE = ""
```

## 2  Tuning & experimentation

All constants live in `parameters.py`, key knobs:

| Name            | Meaning                     | Default  |
| --------------- | --------------------------- | -------- |
| `POP_SIZE`      | Individuals per generation  | 30       |
| `N_GENERATIONS` | Evolution cycles            | 8        |
| `NUM_OBS`       | Max obstacles per test      | 3        |
| `XL` / `XU`     | Search bounds for each gene | see file |
| `TIMEOUT`       | Max seconds per simulation  | 1200     |
| `TOP_K`         | YAMLs to retain at the end  | 15       |

Changing the mission: replace `case_studies/mission2.{plan,yaml}` and update `PLAN_FILE` / `BASE_YAML`.

---

## 3  Quick‑start

```bash
python start.py
```

`start.py` will:  
1. Create output folders (`generated_*`, `results`, `final_tests`)  
2. Spin up the Docker container in detached mode running `middle.py`  
3. Evolve `POP_SIZE` × `N_GENERATIONS` genomes  
4. Terminate the container and copy the `Top‑K` YAMLs to `final_tests/`

Progress is printed every generation. You can stop the run at any time with `Ctrl‑C`, the `atexit` library hook shuts the container gracefully. Its also possible to see the logging output of `middle.py` inside the docker container using the proper docker command in a new terminal:
```bash
docker logs -f <container_name>
```
by default `container_name` is `aerialist-middle` but can be easily changed in `parameters.py` 


## 4  Citation

- **Sajad Khatiri**, Sebastiano Panichella, and Paolo Tonella, "Simulation-based Testing of Unmanned Aerial Vehicles with Aerialist," *In 2024 International Conference on Software Engineering (ICSE)*
  - [Preprint](https://skhatiri.ir/papers/aerialist.pdf)

- **Sajad Khatiri**, Prasun Saurabh, Timothy Zimmermann, Charith Munasinghe, Christian Birchler, and Sebastiano Panichella, "SBFT Tool Competition 2024 - CPS-UAV Test Case Generation Track," *In 2024 IEEE/ACM International Workshop on Search-Based and Fuzz Testing*
  - [Link](https://github.com/skhatiri/UAV-Testing-Competition/blob/master/reports/UAV_Competition_SBFT_2024.pdf)

- **Sajad Khatiri**, Sebastiano Panichella, and Paolo Tonella, "Simulation-based Test Case Generation for Unmanned Aerial Vehicles in the Neighborhood of Real Flights," *In 2023 IEEE 16th International Conference on Software Testing, Verification and Validation (ICST)*
  - [Link](https://ieeexplore.ieee.org/document/10132225)
