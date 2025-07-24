import numpy as np

# ----------------------------------------------------
# Configuration of start.py

# Paths
OUTPUT_DIR = "/mnt/c/UAV/temp/generated_genomes"
RESULTS_DIR = "/mnt/c/UAV/temp/results"
YAML_DIR = "/mnt/c/UAV/temp/generated_tests"
FINAL_DIR = "./final_tests"
DESTINATION_DIR = "/src/aerialist"
HOST_DIR = "/src/aerialist/results/temp"
CASE_STUDIES_DIR = f"{HOST_DIR}/case_studies"
MIDDLE_PATH = f"{HOST_DIR}/middle.py" # check line below
MIDDLE_PY = MIDDLE_PATH[:-9]
TESTCASE_PATH = f"{HOST_DIR}/testcase.py"

# SBFT constraints
NUM_OBS = 3
VARS_PER_OBS = 6
TOTAL_VARS = NUM_OBS * VARS_PER_OBS
XL = np.array([-40, 15, 2, 2, 10, 0] * NUM_OBS)
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
POP_SIZE = 20
N_GENERATIONS = 8
SEED = 42

# # of YAMLs to keep
TOP_K = 15

# DOCKER
DOCKER_IMAGE    = "5ff51322acba"        # Aerialist image ID / tag
CONTAINER_NAME  = "aerialist-middle"
SHARED_HOST_DIR = "/mnt/c/UAV"
SHARED_CONT_DIR = "/src/aerialist/results"
COPY_SCRIPT     = "/temp/copy.sh"

PLAN_FILE = "/mnt/c/UAV/temp/case_studies/mission2.plan"
# tube radius
PAD_M = 10.0

# ----------------------------------------------------
# Configuration of middle.py

INPUT_DIR = f"{HOST_DIR}/generated_genomes"
YAML_DIR = f"{HOST_DIR}/generated_tests"
RESULTS_DIR = f"{HOST_DIR}/results"
BASE_YAML = "./case_studies/mission2.yaml"

# both in second
TIMEOUT = 20 * 60
COOL = 5
POLL = 60
