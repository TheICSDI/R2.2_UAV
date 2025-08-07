import numpy as np

# ----------------------------------------------------
# Configuration of start.py

# Paths
BASE_DIR = ""
OUTPUT_DIR = f"{BASE_DIR}/generated_genomes"
RESULTS_DIR = f"{BASE_DIR}/results"
YAML_DIR = f"{BASE_DIR}/generated_tests"
PLAN_FILE = f"{BASE_DIR}/case_studies/mission2.plan"
FINAL_DIR = "./final_tests"
DESTINATION_DIR = "/src/aerialist"
HOST_DIR = "/src/aerialist/results/temp"
CASE_STUDIES_DIR = f"{HOST_DIR}/case_studies"
MIDDLE_PY = "middle.py"
MIDDLE_PATH = f"{HOST_DIR}/{MIDDLE_PY}"
TESTCASE_PATH = f"{HOST_DIR}/testcase.py"
UTILS_PATH = f"{HOST_DIR}/utils.py"
PAR_PATH = f"{HOST_DIR}/parameters.py"

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
POP_SIZE = 30
N_GENERATIONS = 8
SEED = 42

# # of YAMLs to keep
TOP_K = 15

# DOCKER
DOCKER_IMAGE    = ""        # Aerialist image ID / tag
CONTAINER_NAME  = "aerialist-middle"
SHARED_HOST_DIR = BASE_DIR
SHARED_CONT_DIR = "/src/aerialist/results"

# ----------------------------------------------------
# Configuration of middle.py

INPUT_DIR = f"{HOST_DIR}/generated_genomes"
YAML_DIR = f"{HOST_DIR}/generated_tests"
RESULTS_DIR = f"{HOST_DIR}/results"
BASE_YAML = "./case_studies/mission2.yaml"

# second
TIMEOUT = 20 * 60
COOL = 10
POLL = 60
