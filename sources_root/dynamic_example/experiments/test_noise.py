import torch
from torchvision.transforms import Compose, Resize, ToTensor
import PIL.Image

from dynamic_example.tasks import UAPBaseline
from workflow.lightning_trainer import *
import workflow
class UAPNoise(UAPBaseline):
    def get_adv_example(self):
        return torch.rand_like(self._adv_example)



from dynamic_example.trainer.detection_atk_uap import DetectionSimulatedUAP

test_file = None
class DetectionSimulatedUAP1(DetectionSimulatedUAP):

    def __init__(self, conf):
        self.detector = None
        super().__init__(conf)
        self.test_clean = True


        self.AAA = False
    def init_scenario(self) -> None:
        global test_file
        self._scenario = UAPNoise(self.tensorboard_logger, self.detector, self.hparams)

    @property
    def scenario(self):
        return self._scenario

if __name__ == '__main__':
    init()
    workflow.lightning_trainer.is_test = True
    model = DetectionSimulatedUAP1(getCfg().config.train_model)

    exec(model, None)