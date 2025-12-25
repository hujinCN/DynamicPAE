import argparse
import datetime
import os
import warnings
from typing import Dict, Any

import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import save_image

from dynamic_example.utils.torch_utils import convert_sync_bn
from .path_manager import PathManager, path_to_root, path_cfgs

global_cfg = PathManager()
from pytorch_lightning.loggers import TensorBoardLogger

import  torch
torch.set_float32_matmul_precision('high')
# def force_cudnn_initialization():
#     s = 32
#     dev = torch.device('cuda')
#     torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
#
# force_cudnn_initialization()

is_test = False
load_ckpt = False
old_ckpt_path = None
save_all_ckpt = False

def init(cfg = None, id = None, parser = argparse.ArgumentParser(), init_as_test=False):
    import sys
    runfile = os.path.abspath(sys.argv[0])
    path_root = path_to_root
    dir = runfile[len(path_root):]
    dir = dir.replace("\\", "/")
    assert dir.startswith("/sources_root/")   # Put your code under model dir.!

    name = dir[8:].split("/")[1]
    global global_cfg

    parser.add_argument("-n", "--name", help="Experiment Name", default=name)
    parser.add_argument("-c", "--config", help="Config Name", default=cfg)
    parser.add_argument("-i", "--id", help="Experiment ID", default=id)
    parser.add_argument("--test",  help="Test Flag", default=init_as_test, action="store_true")
    parser.add_argument("--newconfig",  help="Refresh Config Settings", default=init_as_test, action="store_true")
    parser.add_argument("--retrain",  help="Refresh Config Settings", default=False, action="store_true")
    parser.add_argument("-d", "--description", help="Description", default="")

    parser.add_argument("-co", "--config_old", help="Source Config Name", default = None)
    parser.add_argument("-io", "--id_old", help="Source Experiment ID", default = None)
    parser.add_argument("-saveall", "--saveall", help="Save All Ckpt", default = False, action="store_true")

    args = parser.parse_args()
    id = args.id
    args.newconfig |= args.test
    global_cfg = PathManager(args.name, args.config, id, not args.newconfig)
    import pytorch_lightning.utilities
    if pytorch_lightning.utilities.rank_zero_only.rank == 0:
        # if id is None:
        #     print("Please input a unique ID")
        #     id = input("ID:")
        description = args.description
        if description == "":
            print("Please Describe This Experiment!")
            description = input("Text:")

        global_cfg.save_notes(description)


    global is_test, load_ckpt, old_ckpt_path, save_all_ckpt
    is_test = args.test
    if is_test:
        args.retrain = False

    load_ckpt = not args.retrain
    save_all_ckpt = args.saveall

    if args.config_old is not None or args.id_old is not None:
        id_old = args.id_old if args.id_old is not None else id
        cfg_old = args.config_old if args.config_old is not None else args.config
        cfg_origin = PathManager(exp_name=args.name, config_name=cfg_old, exp_id=id_old)
        old_ckpt_path = os.path.join(cfg_origin.get_ckpt_dir(), "last.ckpt")


    ckpt = os.path.join(getCfg().get_log_dir(False), "checkpoints/last.ckpt")
    if os.path.exists(ckpt) and not load_ckpt and not is_test:
        os.remove(ckpt)
    from pytorch_lightning import seed_everything
    seed_everything(1)

    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

def init_fn(name, cfg, id, retrain = False, test = False):
    global global_cfg
    global_cfg = PathManager(name, cfg, id)

    global is_test, load_ckpt
    load_ckpt = not retrain

    ckpt = os.path.join(getCfg().get_log_dir(False), "checkpoints/last.ckpt")
    if os.path.exists(ckpt) and not load_ckpt and not test:
        os.remove(ckpt)
    is_test = test
    from pytorch_lightning import seed_everything
    seed_everything(1)

    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import traceback
def exec(model:pytorch_lightning.LightningModule, loader = None):
    global is_test
    if is_test:
        test(model, loader)
    else:
        do_training = True
        last_exception_epoch = -1
        while do_training:
            try:
                train(model, loader)
                do_training = False
            except Exception as e:
                if hasattr(model, "warning_state"):
                    if last_exception_epoch < model.current_epoch:
                        last_exception_epoch = model.current_epoch
                        traceback.print_tb(e.__traceback__)
                        print(e)
                        print("Reloading from checkpoint to retrain...")
                        ckpt = os.path.join(getCfg().get_log_dir(False), "checkpoints/last.ckpt")
                        if not os.path.exists(ckpt):
                            print("Checkpoint Not Found!")
                            raise e
                        model = model.__class__.load_from_checkpoint(
                            ckpt,
                            map_location=torch.device(model.device))

                        print("Reloaded.")
                        do_training = True
                    else:
                        print("Failed Twice.")
                        raise e
                else:
                    raise e



def getCfg() -> PathManager:
    return global_cfg



class IntervalSaveLast(ModelCheckpoint):
    def __init__(self, interval):
        super().__init__(save_last=True,  every_n_epochs = interval, enable_version_counter = save_all_ckpt)
        self.interval = interval
        # self.new_exp = True


    # def _save_checkpoint(self, trainer: "Trainer", path) -> None:
    #     super()._save_checkpoint(trainer, path)
    #     if trainer.current_epoch + 1 >= trainer.max_epochs or trainer.current_epoch %  self.interval == 0:
    #         os.makedirs(os.path.dirname(path), exist_ok=True)
    #         with open(path.replace("last.ckpt", f"saved_epcho_{trainer.current_epoch}"), "w") as f:
    #             now = datetime.datetime.now()
    #             datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
    #             print("date:" + datetime_str, file=f)
    # def on_load_checkpoint(self,  trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule, checkpoint: Dict[str, Any]):
    #     self.new_exp = False
    #     path = os.path.join(getCfg().get_log_dir(), "checkpoints")
    #
    #     num_epoch = 0
    #     for names in os.listdir(path):
    #         if names.startswith("saved_epcho_"):
    #             num_epoch = max(int(names.split("_")[-1]), num_epoch)
    #     pl_module.loaded_epoch = num_epoch
    #
    #     print(f"CHECKPOINT_LOADED{num_epoch}")
    #     exit()

def prepare_weights(model):
    if old_ckpt_path is not None:

        print(f"Loading model from {old_ckpt_path}")
        import easydict
        with torch.serialization.safe_globals([easydict.EasyDict]):
            
            return model.__class__.load_from_checkpoint(
            old_ckpt_path,
            map_location=torch.device(model.device), **model.hparams.copy())

    return model


def train(model, loader = None):
    model = prepare_weights(model)
    logger = TensorBoardLogger(save_dir=os.path.join(path_cfgs["logs_dir"], global_cfg.exp_name), name=global_cfg.config_name, version=global_cfg.exp_id)
    from pytorch_lightning.strategies import DDPStrategy
    single_gpu = global_cfg.config.trainer.devices == 1
    if not single_gpu:
        model = convert_sync_bn(model)
    global_epoch = global_cfg.config["trainer"]["max_epochs"]

    if global_cfg.copy_config(not global_cfg.use_logged_config):
        global_cfg.copy_model_src()

    if not hasattr(global_cfg.config["trainer"], "ckpt_interval"):
        save_interval = (global_epoch + 9) // 10
    else:
        save_interval = global_cfg.config["trainer"]["ckpt_interval"]
        del global_cfg.config["trainer"]["ckpt_interval"]
    trainer = Trainer(**global_cfg.config["trainer"], default_root_dir=global_cfg.get_log_dir(), logger=logger,
                      strategy="auto" if single_gpu else DDPStrategy(find_unused_parameters=True),
                      sync_batchnorm = False if single_gpu else True,
                      callbacks=[IntervalSaveLast(save_interval)])

    global_cfg.config["trainer"]["ckpt_interval"] = save_interval




    model.hparams.pre_test = True
    # model.eval()
    # trainer.validate(model, ckpt_path="last")
    # exit()
    model.hparams.pre_test = False

    trainer.fit(model, loader, ckpt_path="last" if load_ckpt else None)
    print("done.")

    # Distributed End
# import lightning_fabric.utilities.data
# lightning_fabric.utilities.data.has_iterable_dataset()

# def _process_dataloader(trainer: "pl.Trainer", dataloader: object) -> object:
#     trainer_fn = trainer.state.fn
#     stage = trainer.state.stage
#     if trainer_fn is None or stage is None:
#         raise RuntimeError("Unexpected state")
#
#     if stage != RunningStage.TRAINING:
#         is_shuffled = _is_dataloader_shuffled(dataloader)
#         # limit this warning only for samplers assigned automatically when shuffle is set
#         if is_shuffled:
#             rank_zero_warn(
#                 f"Your `{stage.dataloader_prefix}_dataloader`'s sampler has shuffling enabled,"
#                 " it is strongly recommended that you turn shuffling off for val/test dataloaders.",
#                 category=PossibleUserWarning,
#             )
#         if stage == RunningStage.VALIDATING or stage == RunningStage.TRAINING:
#             return dataloader
#     else:
#         is_shuffled = True
#
#     # let the strategy inject its logic
#     strategy = trainer.strategy
#
#     # automatically add samplers
#     dataloader = trainer._data_connector._prepare_dataloader(dataloader, shuffle=is_shuffled, mode=stage)
#
#     dataloader = strategy.process_dataloader(dataloader)
#
#     # check the workers
#     _worker_check(
#         dataloader,
#         isinstance(strategy, DDPStrategy) and strategy._start_method == "spawn",
#         f"{stage.dataloader_prefix}_dataloader",
#     )
#
#     # add worker_init_fn for correct seeding in worker processes
#     _auto_add_worker_init_fn(dataloader, trainer.global_rank)
#
#     if trainer_fn != TrainerFn.FITTING:  # if we are fitting, we need to do this in the loop
#         # some users want validation shuffling based on the training progress
#         _set_sampler_epoch(dataloader, trainer.fit_loop.epoch_progress.current.processed)
#
#     return dataloader
#
# pytorch_lightning.trainer.connectors.data_connector._process_dataloader = _process_dataloader


from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
def plot(img, path, preds = None, scores_thresh = 0.5, box_scale = 1.0): # preds: max = img_hw, preds: xywh (ltwh in yolo)
    if preds is not None:
        img = img.clone()
        if len(img.shape) == 4:
            for i in range(img.shape[0]):
                mask = preds[i]["scores"] > scores_thresh
                labels = [str(int(label)) for label in preds[i]["labels"][mask]]
                xyxy = box_convert(preds[i]['boxes'][mask], "xywh", "xyxy") * box_scale
                img[i] = draw_bounding_boxes((img[i] * 255).to(torch.uint8), xyxy, labels).float() / 255
        else:
            labels = [str(int(label)) for label in preds["labels"]]
            img = draw_bounding_boxes((img * 255).to(torch.uint8), preds['boxes'], labels).float() / 255
    if len(img.shape) == 4:
        save_image(img, os.path.join(getCfg().get_log_dir() ,path))
        return
    import PIL.Image
    import torchvision.transforms
    pil: PIL.Image.Image = torchvision.transforms.ToPILImage()(img.detach())
    pil.save(os.path.join(getCfg().get_log_dir() ,path))



def test(model, loader = None, ckpt_path = "last"):
    from torch.backends import cudnn
    cudnn.benchmark = True

    model = prepare_weights(model)
    logger = TensorBoardLogger(save_dir=os.path.join(path_cfgs["logs_dir"], global_cfg.exp_name), name=global_cfg.config_name, version=global_cfg.exp_id)
    from pytorch_lightning.strategies import DDPStrategy
    single_gpu = global_cfg.config.trainer.devices == 1
    # Sync BN is not needed since model is in eval mode and BN will not be updated.
    # if not single_gpu:
    #     model = TorchSyncBatchNorm().apply(model)

    if global_cfg.copy_config(not global_cfg.use_logged_config):
        global_cfg.copy_model_src()
    if hasattr(global_cfg.config["trainer"], "ckpt_interval"):
        del global_cfg.config["trainer"]["ckpt_interval"]
    trainer = Trainer(**global_cfg.config["trainer"], default_root_dir=global_cfg.get_log_dir(), logger=logger,
                      strategy="auto" if single_gpu else DDPStrategy(find_unused_parameters=True),
                      sync_batchnorm = False if single_gpu else True,
                      detect_anomaly=True)

    path = os.path.join(getCfg().get_log_dir(), "checkpoints")
    num_epoch = 0
    if os.path.exists(path):
        for names in os.listdir(path):
            if names.startswith("saved_epcho_"):
                num_epoch = max(int(names.split("_")[-1]), num_epoch)
        model.loaded_epoch = num_epoch

        print(f"CHECKPOINT_LOADED: epoch {num_epoch}")
    else:
        print("Checkpoint not found. Test without training. ")
        model.loaded_epoch = -1
    model.eval()

    trainer.test(model, loader, ckpt_path="last" if (load_ckpt and old_ckpt_path is None) else None )



    print("test done.")


if __name__ == '__main__':
    print()

# .sh / console ===> model / ... / *.py == args / kwargs
