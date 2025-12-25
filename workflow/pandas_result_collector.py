import os

import pandas as pd

from workflow.path_manager import PathManager, path_cfgs


def summarize_results(env: PathManager, name, srt_keys = "epoch"):
    logdir = path_cfgs["logs_dir"]
    name = env.exp_name
    logdir = os.path.join(logdir, name)
    df0  = pd.DataFrame()
    for cfg in os.listdir(logdir):
        cfg_dir = os.path.join(logdir, cfg)
        for id in os.listdir(cfg_dir):
            result_file = os.path.join(cfg_dir, id, "results", name)
            df: pd.DataFrame = pd.read_csv(result_file)
            df0 = pd.concat([df0, df], ignore_index=True)

    df0.to_csv(os.path.join(logdir, name))



