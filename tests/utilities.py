import os
import numpy as np
import coolpropx as cpx

# Detect if running in GitHub Actions
IN_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"


# Define available backends
def get_available_backends():
    if os.environ.get("GITHUB_ACTIONS", "false").lower() == "true":
        return ["HEOS"]
    else:
        return ["HEOS", "REFPROP"]
