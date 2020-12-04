import os
import sys

import numpy as np
from BaseObjects import Page
from Predictors.NetPredictor import NetPredictor
from Predictors.VotingPredictor import VotingPredictor
from pathlib import Path

from utils import load

"""
A legacy file for running the model against a fixed input path
"""

script_dir = Path(os.path.abspath(__file__)).parent
data_dir = script_dir.parent / "data"

models_path = data_dir / "models"
if not models_path.exists():
    print("please download models to %s." % models_path)
    sys.exit(1)

# Load all models into a list
loaded = load([
    (NetPredictor, "v3/sep/1"),
    (NetPredictor, "v3/sep/2"),
    (NetPredictor, "v3/sep/4"),
    (NetPredictor, "v3/blkx/2"),
    (NetPredictor, "v3/blkx/4"),
    (NetPredictor, "v3/blkx/5"),
], models_path=models_path)

sep_predictor = VotingPredictor(
    loaded["v3/sep/1"],
    loaded["v3/sep/2"],
    loaded["v3/sep/4"]
)

blkx_predictor = VotingPredictor(
    loaded["v3/blkx/2"],
    loaded["v3/blkx/4"],
    loaded["v3/blkx/5"]
)

for page_path in (data_dir / "pages").iterdir():
    if page_path.suffix == ".jpg":
        page = Page.Page(page_path)
        sep_res = sep_predictor(page)
        sep_res.save(data_dir / (page_path.stem + ".sep.png"))
        labels_array = np.asarray(sep_res.labels)
        blkx_predictor(page).save(data_dir / (page_path.stem + ".blkx.png"))
        print(f"finished {page_path.stem}")
