import numpy as np

import torch.distributed
import torchvision.datasets
from ultralytics.data.augment import LetterBox

from dynamic_example.trainer.detection_atk_pretraining import DetectionPretrain
# import yolov5
# import mmyolo

from dynamic_example.tasks.scenarios import OptimizationStages

from workflow.lightning_trainer import *
from dynamic_example.tasks.scenarios import UAPBaseline


class DetectionSimulatedUAP(DetectionPretrain):

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
        self._scenario = UAPBaseline(self.tensorboard_logger, self.detector, self.hparams)

    def init_models(self):
        pass


    @property
    def scenario(self):
        return self._scenario


    def training_step(self, batch, i_batch):

        img, bboxes = batch
        optim = self.optimizers()
        optim.zero_grad()
        self.scenario(img, bboxes, visualize = DetectionPretrain.dev_run or i_batch == 0)
        loss = self.scenario.fetch_total_loss([OptimizationStages.ATK]).mean()
        self.log("total loss", loss, prog_bar=True)
        self.manual_backward(loss)
        optim.step()
        self.scenario.reset_losses()
        vis = self.scenario.visualize_cache
        # if self.global_rank != 0:
        #     return

        for i, img in enumerate(vis):
            import workflow.lightning_trainer
            workflow.lightning_trainer.plot(img, f"train{i} rnk{self.global_rank}.jpg")
            img_grid = torchvision.utils.make_grid(img.float(), nrow=8, normalize=False, scale_each=False)
            sp = img_grid.shape[-2] / img_grid.shape[-1]
            img_grid = torchvision.transforms.Resize((int(1024 * sp), 1024))(img_grid)
            self.logger.experiment.add_image(f"input img{i}_rnk{self.global_rank}", img_grid, self.global_step)

        # exit()


    def configure_optimizers(self):
        # enc_without_resnet = parameter_list_substract(
        #     list(self.env_encoder.parameters()) + list(self.generator.parameters()), list(self.backbone.parameters()))
        # # target_without_resnet = parameter_list_substract(list(self.target_autoencoder.parameters()), list(self.backbone.parameters()))
        optim_initialize = torch.optim.Adam([self.scenario.get_adv_example()], lr=self.hparams.lr)

        # optim2 = torch.optim.Adam(list(self.generator.parameters()), lr=self.hparams.lr)
        return optim_initialize



if __name__ == '__main__':

    init("yolo", "test01")
    model = DetectionSimulatedUAP(getCfg().config.train_model).cuda().eval()
    trainer = Trainer()
    # train(model, None)
    trainer.test(model, dataloaders = model.test_dataloader())

