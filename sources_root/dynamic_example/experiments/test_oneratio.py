
import sys

import torch

from dynamic_example.tasks import ScenarioDynamicPretrain

print('Python %s on %s' % (sys.version, sys.platform))

from dynamic_example.trainer.detection_atk_pretraining import DetectionPretrain
from workflow.lightning_trainer import *
import workflow

ratio_list = [[1.0, 0.0]]
class CurveTest(ScenarioDynamicPretrain):
    @property
    def num_stages(self):
        return 1

    def forward(self, imgs, bboxes, format="xywh", val=False, val_single=False, visualize=False, ratio=None,
                training_stage=0, meta_data = None, meta_header = []):
        if not (val or val_single):
            return super().forward(imgs, bboxes, format, val, val_single, visualize, ratio,
                            training_stage, meta_data, meta_header)

        ret = []

        if meta_data is None:
            meta_data = [[] for _ in range(imgs.shape[0])]

        for i, r in enumerate(ratio_list):
            ret = ret + super().forward(imgs, bboxes, format, val, val_single, visualize, r,
                    training_stage, meta_data =  [meta_data[j] + [i] for j in range(imgs.shape[0])], meta_header = meta_header + ["ratio"])
        return ret



class TestCurve(DetectionPretrain):
    def init_scenario(self):
        self._scenario = CurveTest(hparams=self.hparams,
                                         generator=self.generator,
                                         detector=self.detector,
                                         ood_detector=None,
                                         generative_nn=self.patch_vae,
                                         task_balacing=self.task_weight_alignment,
                                         logger=self.tensorboard_logger)
if __name__ == '__main__':
    # ps = argparse.ArgumentParser()
    # ps.add_argument("-co", "--config_old", help="Source Config Name")
    # ps.add_argument("-io", "--id_old", help="Source Experiment ID", default=None)
    init()

    conf = getCfg()
    # id_old = args.id_old if args.id_old is not None else conf.exp_id
    # cfg_old = args.config_old if args.config_old is not None else conf.config_name
    # assert not (args.config_old is None and args.id_old is None)
    # cfg_origin = workflow.path_manager.PathManager(exp_name=conf.exp_name, config_name=cfg_old, exp_id=id_old)
    conf.config.train_model.validation_sample_cnt = 3
    model = TestCurve(conf.config.train_model)
    # hp = model.hparams.copy()
    # model = model.__class__.load_from_checkpoint(
    #     os.path.join(cfg_origin.get_ckpt_dir(), "last.ckpt"),
    #     map_location=torch.device(model.device), **hp)

    # model = MaskDemo(conf.config.train_model)
    with torch.inference_mode():
        with torch.no_grad():
            test(model, None)
