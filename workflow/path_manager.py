import os
import shutil
from shutil import copy

import easydict
import yaml
from datetime import datetime
from easydict import EasyDict

def read_yaml(path, cover = True):
    f = open(path, 'r', encoding='utf-8')
    cfg = f.read()
    if cover:
        cfg = cfg.replace("$dataset", path_cfgs["dataset_dir"])\
            .replace("$config", os.path.join(path_to_root, "configs"))\
            .replace("$log", PathManager.get_global_log_dir())
    d = yaml.load(cfg, yaml.Loader)
    f.close()
    return d

path_to_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
path_cfgs = read_yaml(path_to_root + "/workflow/path_cfg.yml", cover=False)
for x in path_cfgs:
    if not path_cfgs[x].startswith("/") or not path_cfgs[x].startswith("\\"):
        path_cfgs[x] = os.path.abspath(os.path.join(path_to_root, path_cfgs[x]))



class PathManager:
    @staticmethod
    def get_global_log_dir():
        return path_cfgs["logs_dir"]
    def __init__(self, exp_name = None, config_name=None, exp_id=None, use_logged_config = True):
        if exp_name is None:
            return # todo: redirect to demo
        if exp_id is None:
            exp_id = datetime.now().strftime("%Y%m%d-%H-%M-%S-") + str(datetime.now().microsecond)
        if config_name is None:
            config_name = "origin"
        if config_name.endswith(".yml"):
            config_name = config_name[:-4]
        if config_name.endswith(".yaml"):
            config_name = config_name[:-5]
        self.config_name = config_name
        self.exp_id = exp_id
        self.exp_name = exp_name
        self.use_logged_config = use_logged_config
        if self.exp_name is not None:
            self.config = read_yaml(self.get_config_path())
            self.config = EasyDict(self.config)
        print(self.get_config_path())
        print(self.config)

    def save_notes(self, description):
        now = datetime.now()
        datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
        description = f"{description} -- {datetime_str}"
        notes = os.path.join(self.get_log_dir(False), "exp_notes.txt")
        with open(notes, "a") as f:
            f.write("\n" + description)


    def get_config_path(self, do_reload = True):
        logged_cfg_path = os.path.join(self.get_log_dir(), "cfg.yml")
        if self.use_logged_config and do_reload and os.path.exists(logged_cfg_path):
            return logged_cfg_path
        p1 = os.path.join(path_to_root, "configs", self.exp_name, self.config_name +
            ("" if self.config_name.__contains__(".") else ".yml"))
        if not os.path.exists(p1) and self.config_name == "origin":
            p1 = os.path.join(path_to_root, "configs", self.exp_name, "config.yml")
        if not os.path.exists(p1):
            raise Exception("Cannot find config file. Check Exp. Name, Cfg. Name & Cfg. file.")
        return p1

    @staticmethod
    def get_dataset_path(name):
        return os.path.join(path_cfgs["dataset_dir"], name)

    def get_log_dir(self, create_dir=True):
        path = os.path.join(PathManager.get_global_log_dir(), self.exp_name, self.config_name, self.exp_id)
        if create_dir and not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_tensorboard_dir(self,  create_dir=True):
        path = os.path.join(self.get_log_dir(create_dir), "tensorboard")
        if create_dir and not os.path.exists(path):
            os.makedirs(path)
        return path + "/"

    def get_ckpt_dir(self, create_dir=True):
        path = os.path.join(self.get_log_dir(create_dir), "checkpoints")
        if create_dir and not os.path.exists(path):
            os.makedirs(path)
        return path + "/"


    def copy_config(self, override = False):
        pp = os.path.join(self.get_log_dir(), "cfg.yml")
        if os.path.exists(pp) and not override:
            return False
        copy(self.get_config_path(False), pp)
        return True

    def copy_model_src(self):
        dir_model = os.path.join(path_to_root, "sources_root")
        log_dir = self.get_log_dir()
        for root, dirs, files in os.walk(dir_model):
            rt = root[len(dir_model) + 1:]
            for file in files:
                if os.path.splitext(file)[1] in [".py", ".yml", ".yaml"] and \
                        not file.__contains__("_remote_module_non_scriptable"): # PyCharm Syn
                    path_old = os.path.join(root,  file)
                    path_new = os.path.join(log_dir, "model_src", rt , file)
                    os.makedirs(os.path.dirname(path_new), exist_ok = True)
                    shutil.copyfile(path_old, path_new)
        dir_model = os.path.join(path_to_root, "configs", self.exp_name)
        log_dir = self.get_log_dir()
        for root, dirs, files in os.walk(dir_model):
            rt = root[len(dir_model) + 1:]
            for file in files:
                if os.path.splitext(file)[1] in [".py", ".yml", ".yaml"] and\
                        not file.__contains__("_remote_module_non_scriptable"): # PyCharm Syn
                    path_old = os.path.join(root, file)
                    path_new = os.path.join(log_dir, "config_src", rt, file)
                    os.makedirs(os.path.dirname(path_new), exist_ok=True)
                    shutil.copyfile(path_old, path_new)




if __name__ == '__main__':
    print(PathManager.get_log_dir("cifar"))
    print(datetime.now().strftime("%Y%m%d-%H-%M-%S-") + str(datetime.now().microsecond))
