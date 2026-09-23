import os


RESOURCES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
OPTIMUM_SWEEPS_DATASET_PATH = os.getenv("OPTIMUM_SWEEPS_DATASET_PATH")