import os.path
from functools import partial
from typing import List, Dict

import numpy as np
import pytorch_lightning as pl
import torch.utils.tensorboard.writer
import torchmetrics
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import box_convert
from torchmetrics.detection import MeanAveragePrecision

from dynamic_example.data.coco_error_analysis import analyze_results_cat_bbox
from dynamic_example.tasks.evaluation import MAP, AttackEvaluationIntegration, PatchEvaluationData
from dynamic_example.utils.detection_result_ops import coco_ann_to_torchmetrics, calc_precision_targeted
from dynamic_example.utils.transforms import ApplyPatchToBBox, LetterBox
from dynamic_example.data.coco_bboxes_dataset import COCOPersonSelect
# from models.adv_tex.yolo2 import load_data, utils
from workflow.lightning_trainer import getCfg

import pandas as pd


def collate_fn_val(batch, lt: LetterBox):
    imgs = []
    gts = []
    targets = []
    toTensor = torchvision.transforms.ToTensor()
    for x in batch:
        img = x[0]
        bboxes_cat = np.array([i["bbox"] for i in x[1] if (i["category_id"] == 1 and i["area"] > 120)],
                              dtype=np.float64)
        if bboxes_cat.shape[0] == 0:
            bboxes_cat = np.zeros((0, 4))
        bboxes = torch.tensor(bboxes_cat)
        # print(bboxes_cat)
        # print("===")
        img, bboxes, params = lt(img, bboxes, return_params=True)

        imgs.append(img)
        gts.append(bboxes)

        targets.append(coco_ann_to_torchmetrics(x[1], transform_bbox=lambda bbox: lt.update_labels(bbox, *params)))

        # print(bboxes)
        # print("===")

    return torch.stack(imgs), torch.stack(gts), targets


def collate_fn_val_single(batch):
    imgs = []
    target_bbox = []
    all_bbox = []
    for img, bbox, ann in batch:
        imgs.append(img)
        target_bbox.append(bbox)
        # all_bbox.append(coco_ann_to_torchmetrics([ann]))
        all_bbox.append(ann)

    return torch.stack(imgs), torch.stack(target_bbox), all_bbox


class BasePatchAttacker(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        conf = conf.copy()
        self.save_hyperparameters(conf)
        self.applier = ApplyPatchToBBox()
        # self.patch_judger = ResVAE(in_channels=3)
        self.automatic_optimization = False
        self.strict_load_ckpt = False
        self.metric: List[torchmetrics.Metric] = []
        # self.register_buffer("sum_ap", None, persistent=False)
        self.val_num = 0
        self.target_cls = 1
        self.tot_time = 0
        self.tot_time_cnt = 0

    def init_metrics(self, num_stages):
        if self.hparams.single_val:
            self.metric = torch.nn.ModuleList(
                [AttackEvaluationIntegration(self.hparams.evaluation) for _ in range(num_stages)])
        else:
            self.metric = torch.nn.ModuleList([MAP(target_cls=self.target_cls, iou_type="bbox", extended_summary=False,
                                                   box_format="cxcywh", class_metrics=True) for _ in range(num_stages)])

    def add_time(self, time):
        self.tot_time += time * 1.0
        self.tot_time_cnt += 1

    def update_metrics(self, imgs, imgs_atked, pred: List[Dict], target: List[Tensor], i,fuse = False, box_format="xywh"):
        """
        @param imgs: benign imgs tensor(BCHW)
        @param imgs_atked: adversarial imgs tensor(BCHW)
        @param pred: predicted bboxes. length = batchsize.
                     Format of each item: see torchmetrics.detection.MeanAveragePrecision and detection_result_ops.py
        @param target: [target & all annotated] bboxes [tensor((1 + N) 4)...] (single val),
                        or all annotated bboxes [tensor(N 4)...] (multi attack val)
        @param i: index of parameter settings. i ∈ [0, num_stages)
        @param box_format: format of pred & target (in torchvision format).
        @param fuse: sum up minibatch for std[mean[minibatch]]
        @return: none
        """
        if self.hparams.single_val:
            for j, t in enumerate(target):
                f = partial(box_convert, in_fmt=box_format, out_fmt="xyxy")
                # noinspection PyTypeChecker
                metric: AttackEvaluationIntegration = self.metric[i]
                # filter multi label result to single label
                if isinstance(self.target_cls, list):
                    mask = sum([pred[j]["labels"] == tc for tc in self.target_cls]) > 0
                else:
                    mask = (pred[j]["labels"] == self.target_cls)
                all_ann, tgt, pred_boxes, conf = f(t[1:]), f(t[0]), f(pred[j]["boxes"]), pred[j]["scores"] * mask
                data = PatchEvaluationData(imgs[j], imgs_atked[j], bbox_all=all_ann, bbox_target=tgt, bbox_results=pred_boxes,
                                           conf_preds=conf)
                metric(data, fuse)
                # self.sum_ap[:, i] += calc_precision_targeted()
            return

        if len(self.metric) == 0:
            raise Exception("self.metric is not initialized")

        self.metric[i].update(pred, target)

    def calc_metrics(self):
        if self.hparams.single_val:
            path = os.path.join(getCfg().get_log_dir(), f"results/val_results.csv")
            dir = os.path.join(getCfg().get_log_dir(), f"results/")

            # TODO: move to path_manager file
            if not os.path.exists(dir):
                os.makedirs(dir)
            if not os.path.exists(path):
                data = pd.DataFrame()
            else:
                data = pd.read_csv(path)
            epoch = self.loaded_epoch if hasattr(self, "loaded_epoch") else self.current_epoch
            for i, metric in enumerate(self.metric):
                metric: AttackEvaluationIntegration
                results = metric.compute(tb = self.logger.experiment)
                results["epoch"] = epoch + 1
                results["exp_id"] = getCfg().exp_id
                results["config_id"] = getCfg().config_name
                results["project"] = getCfg().exp_name
                results["stage"] = i + 1
                results["time"] = self.tot_time / max(self.tot_time_cnt, 1)
                new_row = pd.DataFrame(results, index=[0])
                data = pd.concat([data, new_row], ignore_index=True)
                metric.reset()
            data.to_csv(path, index=False)
            # mAP = self.all_gather(self.sum_ap).reshape(-1, self.sum_ap.shape[0], self.sum_ap.shape[1]).sum(dim = (0,1)) / self.val_num
            # if self.global_rank == 0:
            #     for i in range(mAP.shape[0]):
            #         path = os.path.join(getCfg().get_log_dir(),
            #                      f"results/detection epoch{self.current_epoch} stage{i}.txt")
            #         dir = os.path.join(getCfg().get_log_dir(),
            #                      f"results/")
            #         if not os.path.exists(dir):
            #             os.makedirs(dir)
            #         with open(path, 'w') as f:
            #             f.write(str(float(mAP[i])))
            # self.sum_ap = torch.zeros_like(self.sum_ap)
            self.tot_time_cnt = 0
            self.tot_time = 0.0
            return
        for i in range(len(self.metric)):
            # stop Unsync for _get_coco_format calc. Then reset the state
            self.trainer.strategy.barrier()
            self.metric[i]._should_unsync = False
            self.metric[i].sync(dist_sync_fn=self.metric[i].dist_sync_fn)
            if self.global_rank == 0:
                # print(result["map_per_class"])
                # self.logger.experiment: torch.utils.tensorboard.SummaryWriter
                # for k, v in result.items():
                #     if isinstance(v, torch.Tensor) and sum(v.shape) == 1:
                #         self.logger.experiment.add_scalar(f"val_{i}/" + k, v, global_step=self.global_step)
                # TODO: support multiple class.
                # self.logger.experiment.add_scalar(f"val_{i}/mAP_cls", result["map_per_class"],
                #                                   global_step=self.global_step)
                # if not self.trainer.sanity_checking:
                target_dataset = self.metric[i]._get_coco_format(
                    labels=self.metric[i].groundtruth_labels,
                    boxes=self.metric[i].groundtruth_box,
                    crowds=self.metric[i].groundtruth_crowds,
                    area=self.metric[i].groundtruth_area,
                )
                preds_dataset = self.metric[i]._get_coco_format(
                    labels=self.metric[i].detection_labels, boxes=self.metric[i].detection_box,
                    scores=self.metric[i].detection_scores
                )["annotations"]

                coco = COCO()
                if self.trainer.val_dataloaders is not None:
                    target_dataset["categories"] = self.trainer.val_dataloaders.dataset.coco.dataset["categories"]
                else:
                    target_dataset["categories"] = self.trainer.test_dataloaders.dataset.coco.dataset["categories"]
                coco.dataset = target_dataset
                coco.createIndex()
                coco_pred = coco.loadRes(preds_dataset)
                coco_eval: COCOeval = analyze_results_cat_bbox(cocoGt=coco,
                                                               cocoDt=coco_pred,
                                                               out_dir=os.path.join(getCfg().get_log_dir(),
                                                                                    f"plots/detection epoch{self.current_epoch} stage{i}"))
                print(coco_eval.summarize())

            self.trainer.strategy.barrier()
            self.metric[i]._should_unsync = True
            self.metric[i].reset()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, self.strict_load_ckpt,
                                      missing_keys, unexpected_keys, error_msgs)

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        return super().load_state_dict(state_dict, strict=False)

    def train_dataloader(self) -> DataLoader:
        from dynamic_example.utils.transforms import LetterBox
        # t = torchvision.transforms.Compose(
        #     [torchvision.transforms.ToTensor(), torchvision.transforms.CenterCrop((480, 640))])
        ds = COCOPersonSelect(self.hparams.path_to_coco,
                              self.hparams.path_to_coco_annotation,
                              None if not self.hparams.__contains__("force_item") else self.hparams.force_item,
                              only_single=getattr(self.hparams, "only_single", False),
                              transforms=LetterBox(new_shape=(self.hparams.input_pixel, self.hparams.input_pixel)))
        loader = DataLoader(ds,
                            batch_size=self.hparams.batch_size,
                            shuffle=True,
                            num_workers=self.hparams.num_workers,
                            drop_last=True,
                            # collate_fn= lambda batch: (torch.stack([x[0] for x in batch]), torch.stack([x[0] for x in batch])),
                            pin_memory=self.hparams.pin_memory)
        return loader

    def test_dataloader(self):
        return self.val_dataloader()

    def val_dataloader(self):
        if self.hparams.single_val:

            ds = COCOPersonSelect(self.hparams.path_to_coco_val,
                                  self.hparams.path_to_coco_val_annotation,
                                  None if not self.hparams.__contains__("force_item") else self.hparams.force_item,
                                  return_ann=True,
                                  only_single=getattr(self.hparams, "only_single", False),
                                  transforms=partial(LetterBox(new_shape=(self.hparams.input_pixel, self.hparams.input_pixel)), multiple_tgt=True))

            dl = DataLoader(ds,
                            batch_size=self.hparams.batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_fn_val_single)

            self.val_num = len(ds)

            return dl
        else:
            ds = torchvision.datasets.coco.CocoDetection(
                self.hparams.path_to_coco_val,
                self.hparams.path_to_coco_val_annotation)

            dl = DataLoader(ds,
                            batch_size=self.hparams.batch_size,
                            shuffle=False,
                            num_workers=self.hparams.num_workers,
                            collate_fn=partial(collate_fn_val, lt = LetterBox(new_shape=(self.hparams.input_pixel, self.hparams.input_pixel))))

            return dl

    def on_validation_end(self):
        self.calc_metrics()

    def on_test_end(self) -> None:
        self.calc_metrics()

    # model.training_step((img, box), 1)
