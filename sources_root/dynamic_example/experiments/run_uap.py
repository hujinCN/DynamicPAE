
import sys
print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/home/hj/exp/'])
# sys.path.extend(['/home/hj/exp/models'])

from dynamic_example.trainer.detection_atk_uap import DetectionSimulatedUAP
from workflow.lightning_trainer import *




if __name__ == '__main__':
    init()
    conf = getCfg()
    # model = MaskDemo(conf.config.train_model)
    model = DetectionSimulatedUAP(conf.config.train_model)

    exec(model, None)
