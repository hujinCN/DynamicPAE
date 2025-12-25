"""
The MIT License (MIT)

Copyright (c) 2023 Jin Hu

Permission is hereby granted, free of charge, to any person obtaining a copy of
this file and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import abc
from abc import ABC
from typing import Dict, List, Mapping, Optional

import torch
import torchvision
import ultralytics.nn.modules

from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.utils import ops
import torch.nn.functional as F

from dynamic_example.utils import freeze_bn
from dynamic_example.utils.detection_result_ops import ultralytice_Boxes_to_pred, \
    coco_91_Boxes_to_pred


class DetectionCaller(ABC):

    def __call__(self, imgs: torch.Tensor, eval_mode=False) -> torch.Tensor:
        """
        @param imgs: Image Tensor with shape BCHW. HW may be arbitrary.
        @return result: tensor with shape BNL. N: bbox count. L: 4 + Num_Class. [ccwh, prob(cls1), prob(cls2)....]
            The first four dimensions denote bbox center_x center_y w h, normalized to 1. # TODO: change center_x center_y w h to a more consistent interface
            The latter dimensions denote the final confidence of each category
                (which may be obtained by multiplying box confidence and category confidence).
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def pred(self, imgs: torch.Tensor) -> List[Dict]:
        """
        @param imgs: Image Tensor with shape BCHW. HW may be arbitrary.
        @return result after nms.
            [pred_1:{
                "labels": uint8 tensor[N],
                "boxes": tensor[N, 4], in xywh
                "scores": tensor[N] ∈ [0, 1],
            }, pred_2:..., pred_{batch_size}]
        """
        raise NotImplementedError()


class UltralyticsModel(DetectionCaller):
    def __init__(self, model: ultralytics.nn.modules.Detect, hparams, out_format="ssd"):
        self.model = model
        self.detect_pixel = hparams.detect_pixel
        self.model_out_format = 0 if "rtdetr" in hparams.detector else 1

        self.transform = torchvision.transforms.Resize(self.detect_pixel, antialias=True)
        self.out_format = out_format

    def __call__(self, img_input, eval_mode=False):
        img_input = self.transform(img_input)
        pred = self.model(img_input)
        if isinstance(pred, tuple):
            pred = pred[0]
        if self.model_out_format == 0:
            pred = pred.permute(0, 2, 1)
        else:
            pred[:, :4] /= self.detect_pixel
        
        if self.out_format == "ssd":
            return pred.permute(0, 2, 1)
        else:
            raise NotImplementedError()

    def pred(self, imgs: torch.Tensor) -> List[Dict]:
        result = self(imgs).permute(0, 2, 1) # B L N
        # if self.model_out_format == 0:# RTDETR
        #     temp = []
        #     for i, r in enumerate(result):  # (300, 4)
        #         bbox = ops.xywh2xyxy(bbox)  # (300, 4)
        #         bbox[..., 2:] += bbox[..., :2]
        #         bbox = bbox.cpu().numpy()
        #         scores = scores[i].cpu().numpy()
        #         cls = cls[i].cpu().numpy()
        #         idx = r[..., 4:].sum(dim = 1) > 0.25
                
        #         score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
        #         idx = score.squeeze(-1) > self.args.conf  # (300, )
        #         if self.args.classes is not None:
        #             idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
        #         pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
        #         orig_img = orig_imgs[i]
        #         oh, ow = orig_img.shape[:2]
        #         pred[..., [0, 2]] *= ow
        #         pred[..., [1, 3]] *= oh
        #         img_path = self.batch[0][i]
        #         results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))

        #     result = temp
        # else:
        result = ops.non_max_suppression(result, max_time_img=1.0, max_det=300, iou_thres=0.7, conf_thres=0.001,
                                         multi_label=False)
        
        # torchvision.ops.nms()
        s = 1.0
        orig_shape = (1.0, 1.0)
        
        bboxes = [ultralytice_Boxes_to_pred(Boxes(x, orig_shape), scale=s) for x in result]
        

        return bboxes


class TSEAModel(DetectionCaller):
    def __init__(self, model, model_eval: Optional[torch.nn.Module], hparams):
        self.model = model
        self.model_eval = model_eval
        self.detect_pixel = hparams.detect_pixel
        self.transform = torchvision.transforms.Resize(self.detect_pixel, antialias=True)
        self.hp = hparams

    def __call__(self, img_input, eval_mode=False):
        img_input = self.transform(img_input)
        if eval_mode and self.model_eval is not None:
            pred = self.model_eval(img_input)
        else:
            pred = self.model(img_input)
        if isinstance(pred, tuple):
            pred = pred[0]
        if "faster_rcnn" not in self.hp.detector_cfg:
            pred[..., :4] /= self.detect_pixel
        else:
            pred[..., :4] = torchvision.ops.box_convert(pred[..., :4], "xyxy", "cxcywh")
        return pred

    def pred(self, imgs: torch.Tensor) -> List[Dict]:
        if "faster_rcnn" in self.hp.detector_cfg:
            # result = self(imgs, eval_mode=True).permute(0, 2, 1)
            # result = ops.non_max_suppression(result, max_time_img=1.0, max_det=300, iou_thres=0.7, conf_thres=0.001,
            #                                  multi_label=False)
            # s = 1.0
            # orig_shape = (1.0, 1.0)
            # bboxes = [coco_91_Boxes_to_pred(Boxes(x, orig_shape), scale=s) for x in result]
            result = self.model_eval(imgs)[1]
            ans = []
            for res in result:
                ans.append(torch.tensor(res).to(imgs.device))
            s = 1.0
            orig_shape = (1.0, 1.0)
            bboxes = [coco_91_Boxes_to_pred(Boxes(x, orig_shape), scale=s) for x in ans]
        else:
            result = self(imgs, eval_mode=True).permute(0, 2, 1)

            result = ops.non_max_suppression(result, max_time_img=1.0, max_det=300, iou_thres=0.7, conf_thres=0.001,
                                             multi_label=False)
            s = 1.0
            orig_shape = (1.0, 1.0)
            bboxes = [ultralytice_Boxes_to_pred(Boxes(x, orig_shape), scale=s) for x in result]
        return bboxes


# import mmdet
#
# from mmdet.models import BaseDetector
# from mmdet.structures import DetDataSample
# from mmdet.registry import MODELS
# class MMDetDetector(DetectionCaller):
#     def __init__(self, detector, img_shape, out_format = "ssd"):
#         if isinstance(detector, Mapping):
#             from mmdet.models import build_detector
#             detector = build_detector(detector)
#
#         self.detector: BaseDetector = detector
#         # self.detector.init_weights()
#
#
#         self.bbox_head = detector.with_bbox
#
#         img_meta = dict(img_shape=img_shape, pad_shape=img_shape, scale_factor = (1.0, 1.0))
#         self.data_sample = DetDataSample()
#         self.data_sample.set_metainfo(img_meta)
#         self.data_sample.gt_instances = InstanceData()
#         self.data_sample.gt_instances.bboxes = torch.rand((5, 4))
#         self.data_sample.gt_instances.labels = torch.rand((5))
#         self.out_format = out_format
#
#         self.use_sigmoid = False
#
#
#
#     def __call__(self, img_input):
#         img_input = self.detector.data_preprocessor({"inputs":img_input}, False)["inputs"]
#         results = self.detector(img_input, [self.data_sample.clone() for _ in range(img_input.shape[0])], "tensor")
#         # 2 stage roi head
#         cls_score = results[0][0]  # (B * N, num_cls + 1)
#         pred_bbox = results[0][1]  # (B * N, num_cls * 4)
#
#         if self.use_sigmoid:
#
#             cls_prob = F.softmax(cls_score, dim = -1)[:, :-1] # (B * N, num_cls)
#         else:
#             cls_prob = F.softmax(cls_score, dim = -1)[:, :-1] # (B * N, num_cls)
#
#
#
#         cls_prob = cls_prob.reshape(img_input.shape[0], -1, cls_prob.shape[-1]) #(B, N, num_cls)
#
#         pred_bbox = results[0][1].reshape(img_input.shape[0], -1, cls_prob.shape[-1], 4)  # (B, N, num_cls, 4)
#
#
#         if self.out_format == "two_stage":
#             return pred_bbox, cls_prob
#
#
#         mx_idx = cls_prob.max(dim = -1)[1].unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,4) # B N 1 4
#         # gather: out[i][j][k][l] = input[i][j][index[i][j][k][l]][l], k ≡ 0
#         pred_bbox = torch.gather(pred_bbox,  dim = 2, index=mx_idx)[:, :, 0, :] # B N 4
#         # pred_bbox
#         return torch.cat([pred_bbox, cls_prob], dim = -1)
#
#
#     def pred(self, img_input):
#         img_input = self.detector.data_preprocessor({"inputs":img_input}, False)["inputs"]
#         preds: List[DetDataSample] = self.detector(img_input, [self.data_sample.clone() for _ in range(img_input.shape[0])], mode =  'predict')
#         ret = []
#
#
#         for pred in preds:
#             ret.append(mmdet_Instances_to_pred(pred.pred_instances))
#             # pred.proposals.all_items()
#             # boxes = pred.proposals.bboxes
#             # labels = pred.proposals.det_labels
#             # scores = pred.proposals.det_scores
#         return ret
#
#         # pred.proposals.all_items()
#


def create_benchmark_detector(organization, name, hparams):
    if organization == "mmdet":
        import mmengine.hub
        model: BaseDetector = mmengine.hub.get_model(name, pretrained=True)
        image_shape = hparams.detect_pixel
        if isinstance(hparams.detect_pixel, int):
            image_shape = (hparams.detect_pixel, hparams.detect_pixel)
        freeze_bn(model)
        model.eval()
        model.train = lambda x: model
        for param in model.parameters():
            param.requires_grad = False

        def void(*args, **kwargs):
            return {}

        return MMDetDetector(model, image_shape)

    elif organization == "ultralytics":
        detector = YOLO(name).model
        freeze_bn(detector)
        detector.eval()
        detector.train = lambda x: detector
        for param in detector.parameters():
            param.requires_grad = False

        def void(*args, **kwargs):
            return {}

        detector.state_dict = void
        return UltralyticsModel(detector, hparams)


class MaskrcnnResnet50:
    """
    tensor_image_input: torch.Size([3, h, w])
    """

    def __init__(self, device='cuda'):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval().to(device)
        self.device = device

    def detect(self, tensor_image_inputs, cls_id_attacked=0, threshold=0.5):
        bboxes = []
        prof_max_scores = []
        any_max_scores = []
        for tensor_image_input in tensor_image_inputs:
            outputs = self.model([tensor_image_input])[0]
            #
            outputs["boxes"][:, 0] = outputs["boxes"][:, 0] / tensor_image_input.size()[-2]
            outputs["boxes"][:, 1] = outputs["boxes"][:, 1] / tensor_image_input.size()[-1]
            outputs["boxes"][:, 2] = outputs["boxes"][:, 2] / tensor_image_input.size()[-2]
            outputs["boxes"][:, 3] = outputs["boxes"][:, 3] / tensor_image_input.size()[-1]
            # create bbox with (batch,7). (x1,y1,x2,y2,score,score,class_id)
            batch = outputs["boxes"].size()[0]
            outputs["labels"] = outputs["labels"] - 1  # without class __background__
            bbox = torch.cat((outputs["boxes"], outputs["scores"].resize(batch, 1), outputs["scores"].resize(batch, 1),
                              outputs["labels"].resize(batch, 1)), 1)
            # get items with cls_id_attacked
            any_max_score = torch.max(bbox[:, -2])
            any_max_scores.append(any_max_score)
            bbox = bbox[(bbox[:, -1] == cls_id_attacked)]
            # score > threshold
            bbox = bbox[(bbox[:, -2] >= threshold)]
            if (bbox.size()[0] > 0):
                # get max score
                max_score = torch.max(bbox[:, -2])
                # print("max_score : "+str(max_score))

                bboxes.append(bbox)
                prof_max_scores.append(max_score)
            else:
                bboxes.append(torch.tensor([]))
                prof_max_scores.append(torch.tensor(0.0).to(self.device))
        # stack
        if (len(prof_max_scores) > 0):
            prof_max_scores = torch.stack(prof_max_scores, dim=0)
        else:
            prof_max_scores = torch.stack(any_max_scores, dim=0) * 0.01
            if (tensor_image_inputs.is_cuda):
                prof_max_scores = prof_max_scores.cuda()
            else:
                prof_max_scores = prof_max_scores

        return prof_max_scores, bboxes


if __name__ == '__main__':
    # from mmdet.structures.det_data_sample import *
    # import mmengine.hub
    # import PIL.Image

    image = PIL.Image.open('../../../common/')
    img = torchvision.transforms.ToTensor()(image).unsqueeze(0).repeat(2, 1, 1, 1)
    img = torchvision.transforms.Resize((256, 256))(img)
    model: BaseDetector = mmengine.hub.get_model('mmdet::dino/dino-4scale_r50_improved_8xb2-12e_coco.py',
                                                 pretrained=True)

    data_sample = DetDataSample()
    img_meta = dict(img_shape=(256, 256), pad_shape=(256, 256), ori_shape=(256, 256))
    img_meta['scale_factor'] = [1.0] * 2
    # data_sample.gt_instances.metainfo
    data_sample.set_metainfo(img_meta)
    data_sample.batch_input_shape = (256, 256)

    data_sample.gt_instances = InstanceData(metainfo=img_meta)
    data_sample.gt_instances.bboxes = torch.tensor([[0, 0, 1, 1]] * 5)
    data_sample.gt_instances.labels = torch.zeros((5)).long()

    # tensor mode
    results = model(img, [data_sample.clone() for _ in range(img.shape[0])], "tensor")
    cls_score = results[0][0]  # (B * N, num_cls + 1)
    pred_score = results[0][1]  # (B * N, num_cls * 4)

    # predict mode

    results_pred = model(img, [data_sample.clone() for _ in range(img.shape[0])], "predict")

    results_pred: List[DetDataSample]
    results_pred[0].pred_instances.bboxes  # (N 4)
    results_pred[0].pred_instances.scores  # N
    results_pred[0].pred_instances.labels  # N

    # F.softmax(cls_score, dim=-1)
