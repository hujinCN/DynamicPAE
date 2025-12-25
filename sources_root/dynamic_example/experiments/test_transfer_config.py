
import sys

import workflow.path_manager

print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/home/hj/exp/'])
# sys.path.extend(['/home/hj/exp/models'])

from dynamic_example.trainer.detection_atk_pretraining import DetectionPretrain
from workflow.lightning_trainer import *




if __name__ == '__main__':
    ps = argparse.ArgumentParser()
    # ps.add_argument("-co", "--config_old", help="Source Config Name")
    # ps.add_argument("-io", "--id_old", help="Source Experiment ID", default = None)
    init(parser = ps, init_as_test = True)

    conf = getCfg()
    # args = ps.parse_args()
    # id_old = args.id_old if args.id_old is not None else args.id
    # cfg_old = args.config_old
    # cfg_origin = workflow.path_manager.PathManager(exp_name=conf.exp_name, config_name=cfg_old, exp_id=id_old)
    # # model = MaskDemo(conf.config.train_model)
    model = DetectionPretrain(conf.config.train_model)
    # hp = model.hparams.copy()
    # model = model.__class__.load_from_checkpoint(
    #     os.path.join(cfg_origin.get_ckpt_dir(), "last.ckpt"),
    #     map_location=torch.device(model.device), **hp)
    test(model, None)
