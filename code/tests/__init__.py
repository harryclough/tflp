import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SOURCE_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SOURCE_PATH)
