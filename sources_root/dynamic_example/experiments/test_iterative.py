import argparse

from dynamic_example.tasks import TorchATKScenario
from dynamic_example.trainer.detection_atk_uap import DetectionSimulatedUAP
from workflow.lightning_trainer import init, test, getCfg


eot_steps = 10
pgd_steps = 100
class DetectionIterativeUAP(DetectionSimulatedUAP):


    def init_scenario(self) -> None:
        sc = TorchATKScenario(self.tensorboard_logger, self.detector, self.hparams)
        sc.atk_obj.eot_iter = int(eot_steps)
        sc.atk_obj.steps = int(pgd_steps)
        sc.atk_obj.alpha = 10 / sc.atk_obj.steps
        sc.atk_obj.eps = 0.49
        self._scenario = sc

    @property
    def scenario(self):
        return self._scenario


if __name__ == '__main__':

    ps = argparse.ArgumentParser()

    ps.add_argument("-e", "--eot", help="eot steps", type=float)
    ps.add_argument("-s", "--steps", help="pgd steps", type=float)
    init(parser=ps, init_as_test = True)
    args = ps.parse_args()
    eot_steps = args.eot
    pgd_steps = args.steps
    # getCfg().config.train_model.batch_size = 1
    getCfg().config.train_model.num_workers = 1
    getCfg().config.train_model.validation_sample_cnt = 1
    getCfg().config.precision = "32"
    model = DetectionIterativeUAP(getCfg().config.train_model)


    test(model, None)