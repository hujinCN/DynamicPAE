import argparse

import pandas as pd
df = pd.DataFrame()
from workflow.path_manager import *


logdir = path_cfgs["logs_dir"]
pathc = os.path.join
#%%

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Project Name")
parser.add_argument("-c", "--config", nargs='+', help="Config Name")
args = parser.parse_args()
config_names = args.config
project_name = args.name


#%%
result = pd.DataFrame()
#%%
import re
def run(base, template_id = ".*"):
    global result
    for x in os.listdir(base):
        mark = pathc(base, x, "cfg.yml")
        if os.path.exists(mark):
            csv = pathc(base, x, "results", "val_results.csv")
            if os.path.exists(csv) and re.match(template_id, x):
                data = pd.read_csv(csv)
                # returns a table containing different values of subset keys
                data = data.drop_duplicates(subset=["stage", "exp_id", "config_id", "project"], keep="last")
                result = pd.concat([data, result])
                print(f"csv inserted: {csv}")
        elif os.path.isdir(pathc(base, x)):
            run(pathc(base, x), template_id)

for c in config_names:
    basedir = pathc(logdir, project_name, c)
    # run(basedir, r"^gradrescale$") # change this manually for filtering ids
    run(basedir) # change this manually for filtering ids

result.to_csv("output.csv")


