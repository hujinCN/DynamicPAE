from torchvision.transforms import Compose, Resize, ToTensor
import PIL.Image

from dynamic_example.tasks import UAPBaseline
from workflow.lightning_trainer import *
import workflow
class UAPFile(UAPBaseline):
    def __init__(self, logger, detector, hparams, file):
        super().__init__(logger, detector, hparams)
        img = PIL.Image.open(file)
        tr = Compose([
            Resize(self.patch_size),
            ToTensor()
        ])

        self.adv_p = tr(img).unsqueeze(0)
    def get_adv_example(self):
        return self.adv_p.to(self._adv_example)



from dynamic_example.trainer.detection_atk_uap import DetectionSimulatedUAP

test_file = None
class DetectionSimulatedUAP1(DetectionSimulatedUAP):

    def __init__(self, conf):
        self.detector = None
        super().__init__(conf)
        # self.init_detector()
        # self.tmp_results = []
        self.test_clean = False
        # self.loss_atk_detector = DetLossSingle(target_clsses=[0])
        # self.pach_vae = ResVAE2()
        # self.generator.attach_vae(self.pach_vae)


        self.AAA = False
    def init_scenario(self) -> None:
        global test_file
        self._scenario = UAPFile(self.tensorboard_logger, self.detector, self.hparams, test_file)

    @property
    def scenario(self):
        return self._scenario

if __name__ == '__main__':
    init(init_as_test = True)
    workflow.lightning_trainer.is_test = True
    print("Please Input Image File")
    getCfg().config.trainer.devices = 1
    test_file = input("Abs Path:")
    model = DetectionSimulatedUAP1(getCfg().config.train_model)

    exec(model, None)