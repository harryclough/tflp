import json
from math import exp
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

from src.scipysolver import ScipySolver


@dataclass
class MetaData:
    NAME = "name"
    VAR_NAMES = "var_names"
    N_RESULTS = "n_results"
    N_ITERS = "n_iters"
    OTHER = "other"


@dataclass
class ExperimentResult:
    solver: ScipySolver
    experiment_name: str
    experiment_vars: list
    iteration: int
    
    def save(self, path, filename):
        i = 0
        filepath = os.path.join(path, f"{filename}.pkl")
        while os.path.isfile(filepath):
            i += 1
            filepath = os.path.join(path, f"{filename}_({i}).pkl")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    
    def __repr__(self) -> str:
        return ("ExperimentResult"
            f"(name={self.experiment_name}, "
            f"iter={self.iteration}, "
            f"vars={self.experiment_vars})")


def save_metadata(filepath: str, name: str, var_names: List[str], n_results: int, iters: int, other={}):
    with open(filepath, "w") as f:
        json.dump({
            MetaData.NAME: name,
            MetaData.VAR_NAMES: var_names,
            MetaData.N_RESULTS: n_results,
            MetaData.N_ITERS: iters,
            MetaData.OTHER: other,
        }, f, indent=2)


def load_experiment_folder(folder: str, verbose=False) -> Tuple[dict, List[ExperimentResult]]:
    results: List[ExperimentResult] = []
    metadata = {}
    files = os.listdir(folder)
    for filename in files:
        _, ext = os.path.splitext(filename)
        if ext == ".json":
            if len(metadata) > 0:
                print(f"WARN <{folder}>: More than one .json found in experiments folder. "
                      "Using first.")
                continue
            with open(os.path.join(folder, filename), "rb") as f:
                metadata = json.load(f)
        elif ext == ".pkl":
            with open(os.path.join(folder, filename), "rb") as f:
                result = pickle.load(f)
                results.append(result)

    if len(metadata) > 0:
        n_expected = metadata[MetaData.N_RESULTS]
        if n_expected != len(results):
            print(f"WARN <{folder}>: loaded {len(results)} but metadata expects {n_expected}.")
        elif verbose:
            print(f"Loaded {len(results)} results from '{folder}'.")
    else:
        print(f"WARN <{folder}>: no metadata file found in folder.")
    if len(results) == 0:
        print(f"WARN <{folder}>: no results found in folder.")

    return metadata, results
