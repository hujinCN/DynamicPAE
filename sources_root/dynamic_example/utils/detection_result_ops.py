import io
import warnings
from typing import Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops._utils import _loss_inter_union
from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.engine.results import Boxes
from ultralytics.utils import ops


class FORMAT:
    XYXY = 0
    LTWH = 1
    XYWH = 2

ultralytics_formats = ["xyxy", "ltwh", "xywh"]
torchmetrics_formats = ["xyxy", "xywh", "cxcywh"] # also torchvision formats
COCO_default_format = 1


coco80_to_coco91_class_static = torch.tensor(coco80_to_coco91_class())

def coco_91_Boxes_to_pred(boxes: Boxes, scale = 1.0) -> Dict:
    """ultralytics --> torchmetrics 'xywh'"""
    pred = {
        "labels": (boxes.cls+1).to(torch.uint8),
        "boxes": ops.xywh2ltwh(boxes.xywh.to(torch.float32)) * scale,
        "scores": boxes.conf.to(torch.float32),
    }
    return pred
def ultralytice_Boxes_to_pred(boxes: Boxes, scale = 1.0) -> Dict:
    """ultralytics --> torchmetrics 'xywh'"""
    pred = {
        "labels": coco80_to_coco91_class_static.to(boxes.cls)[boxes.cls.long()].to(torch.uint8),
        "boxes": ops.xywh2ltwh(boxes.xywh.to(torch.float32)) * scale,
        "scores": boxes.conf.to(torch.float32),
    }
    return pred
try:
    from mmdet.structures.bbox import HorizontalBoxes
    from mmengine.structures import InstanceData
    def mmdet_Instances_to_pred(instances: InstanceData, scale = 1.0) -> Dict:
        box: HorizontalBoxes = instances.bboxes
        """ultralytics --> torchmetrics 'xywh'"""
        pred = {
            "labels": coco80_to_coco91_class_static.to(instances.labels)[instances.labels.long()].to(torch.uint8),
            "boxes": ops.xyxy2ltwh(box.bboxes.to(torch.float32)) * scale,
            "scores": instances.scores.to(torch.float32),
        }
        return pred
except:
    pass





def coco_ann_to_torchmetrics(target_list, iou_type = "bbox", ds_ref: COCO = None, transform_bbox = None):
    """
    Args:
        iou_type: Type of input, either `bbox` for bounding boxes or `segm` for segmentation masks
    """
    target = {
        "labels": [],
        "iscrowd": [],
        "area": [],
    }
    if "bbox" == iou_type:
        target["boxes"] = []
    if "segm" == iou_type:
        target["masks"] = []
    for t in target_list:


        if "bbox" == iou_type:
            target["boxes"].append(t["bbox"])
        if "segm" == iou_type:
            target["masks"].append(ds_ref.annToMask(t))

        target["labels"].append(t["category_id"])
        target["iscrowd"].append(t["iscrowd"])
        target["area"].append(t["area"])

    bt = {
        "labels": torch.tensor(target["labels"], dtype=torch.int32),
        "iscrowd": torch.tensor(target["iscrowd"], dtype=torch.int32),
        "area": torch.tensor(target["area"], dtype=torch.float32),
    }
    if "bbox" in iou_type:
        boxes = torch.tensor(target["boxes"], dtype=torch.float32)
        if transform_bbox is not None:
            boxes = transform_bbox(boxes)
        bt["boxes"] = boxes
    if "segm" in iou_type:
        bt["masks"] = torch.tensor(np.array(target["masks"]), dtype=torch.uint8)
    return bt




def calc_conf_fn(bboxes_all, bbboxes_target, bboxes_pred, conf_pred, iou_thresh = 0.5, conf_thresh_min = 0.5): # N 4
    """
    Batched is not supported
    bboxes_all: all bboxes with same class shape=[Mx4]
    bbboxes_target: targeted box shape=[4]
    bboxes_pred: pred bboxes with same class shape=[Nx4]
    conf_pred: pred conf with same class shape=[N]
    """
    num_pred = bboxes_pred.shape[0]
    if num_pred == 0:
        return torch.tensor(0.0, device=bboxes_all.device), torch.tensor([], device=bboxes_all.device)
    num_gt = bboxes_all.shape[0]
    # print(num_gt)
    def iou(x, y):
        inter, union = _loss_inter_union(x.float(), y.float())
        return inter / (union + 1e-10)
    tgt_ious = iou(bbboxes_target.reshape(1, 4).repeat(num_pred, 1), bboxes_pred) # N_pred
    all_ious = iou(bboxes_all.reshape(-1, 1, 4).repeat(1, num_pred, 1), bboxes_pred.reshape(1, -1, 4).repeat(num_gt, 1, 1)) # N_gt N_pred
    conf_matched_mx = (conf_pred * (tgt_ious >= iou_thresh)).max(dim = -1)[0]
    if conf_matched_mx < conf_thresh_min:
        conf_matched_mx *= 0.0
    #     return torch.tensor(0.0, device=bboxes_all.device), torch.tensor([], device=bboxes_all.device)

    masks = torch.logical_and((all_ious < iou_thresh).all(dim = 0), conf_pred > conf_thresh_min)

    # print(conf_thresh, tgt_ious, all_ious, FP_MAX)
    # exit()
    # print(num_pred, FP_MAX, conf_thresh)
    # print(num_pred, FP_MAX, conf_matched_mx)
    return conf_matched_mx, conf_pred[masks].contiguous()

def computeAP(confs, fn_confs, return_curve = False, conf_thresh = 0.0):
    all_tgt = confs.shape[0]
    confs = torch.sort(confs, descending=True)[0]
    confs = confs[confs >= conf_thresh].contiguous()
    fn_confs = fn_confs[fn_confs >= conf_thresh].contiguous()
    THRESH = 1024 * 1024 * 1024 // max(fn_confs.shape[0], 1)# 1GB Matrix
    if THRESH < 128:
        warnings.warn("Allocating Large GPU Mem")
        THRESH = 128
    dREC = confs.shape[0] // THRESH  + 1
    # dREC = 1

    confs = confs [dREC // 2::dREC]

    confs = confs.reshape(-1, 1)
    fn_confs = fn_confs.reshape(1, -1)
    # noinspection PyTypeChecker
    mask: torch.Tensor = fn_confs > confs
    fn = mask.sum(dim = -1)
    tp = torch.arange(1, confs.shape[0] + 1, device=confs.device, dtype=torch.float32) * dREC - dREC // 2
    PR = tp / (tp + fn)
    # Rec = tp / all_tgt, delta_Rec = 1 / all_tgt


    cur_max = 0
    PR_np = PR.detach().cpu().numpy()
    for i in reversed(range(PR_np.shape[0])):
        cur = PR_np[i]
        PR_np[i] = max(cur_max, PR_np[i])
        cur_max = max(cur_max, cur)
    PR = torch.tensor(PR_np).to(PR).sum() * dREC / all_tgt
    if return_curve:
        padded_arr = np.zeros((all_tgt,), dtype=PR_np.dtype)
        padded_arr[:PR_np.shape[0]] = PR_np
        return padded_arr, float(PR)
    return PR


def run_test():
    t = lambda x: torch.tensor(x, device="cpu")
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    import matplotlib.pyplot as plt
    def ev(tc, fc, thresh = 0.01):
        curve, ap = computeAP(t(tc), t(fc), return_curve=True, conf_thresh=thresh)
        # print(curve)
        x = np.linspace(0, 1, curve.shape[0])
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
        plt.plot(x, curve)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'AP={ap}')
        plt.grid(True)
        plt.show()
        from sklearn.metrics import precision_recall_curve
        y_true = np.concatenate([np.ones_like(tc), np.zeros_like(fc)])
        y_score = np.concatenate([tc, fc])
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve SKLearn')
        plt.legend()
        plt.show()
        # case 1
    true_conf = [0.3, 0.4]
    fn_conf = [0.0]
    ev(true_conf, fn_conf)

    true_conf = [0.3, 0.4] * 100
    fn_conf = [0.35] * 100
    ev(true_conf, fn_conf)


    true_conf = [0.0, 0.4, 0.6] * 100
    fn_conf = [0.1]
    ev(true_conf, fn_conf)

if __name__ == '__main__':
    run_test()


def calc_precision_targeted(bboxes_all, bbboxes_target, bboxes_pred,  conf_pred,  iou_thresh = 0.5, conf_thresh = 0.5): # N 4
    """
    Batched is not supported
    bboxes_all: all bboxes with same class shape=[Mx4]
    bbboxes_target: targeted box shape=[4]
    bboxes_pred: pred bboxes with same class shape=[Nx4]
    conf_pred: pred conf with same class shape=[Nx4]
    """
    # if ret_linespace is None:
    #     ret_linespace = torch.linspace(0, 1, steps = 100, device=bboxes_all.device)
    num_pred = bboxes_pred.shape[0]
    if num_pred == 0:
        return 0.0
    num_gt = bboxes_all.shape[0]
    # print(num_gt)
    def iou(x, y):
        inter, union = _loss_inter_union(x.float(), y.float())
        return inter / (union + 1e-10)
    tgt_ious = iou(bbboxes_target.reshape(1, 4).repeat(num_pred, 1), bboxes_pred) # N_pred
    all_ious = iou(bboxes_all.reshape(-1, 1, 4).repeat(1, num_pred, 1), bboxes_pred.reshape(1, -1, 4).repeat(num_gt, 1, 1)) # N_gt N_pred
    conf_matched_mx = (conf_pred * (tgt_ious >= iou_thresh)).max(dim = -1)[0]
    if conf_matched_mx < conf_thresh:
        return 0.0
    FP_MAX = (torch.logical_and((all_ious < iou_thresh).all(dim = 0), (conf_pred > conf_matched_mx.reshape(-1)))).sum()
    # print(conf_thresh, tgt_ious, all_ious, FP_MAX)
    # exit()
    # print(num_pred, FP_MAX, conf_thresh)
    # print(num_pred, FP_MAX, conf_matched_mx)
    return 1 / (1 + FP_MAX)


# TODO: ATK. Success Rate
# def calc_asr_targeted(bboxes_all, bbboxes_target, bboxes_pred,  conf_pred,  iou_thresh = 0.5, conf_thresh = 0.5): # N 4
#     """
#     Batched is not supported
#     bboxes_all: all bboxes with same class shape=[Mx4]
#     bbboxes_target: targeted box shape=[4]
#     bboxes_pred: pred bboxes with same class shape=[Nx4]
#     conf_pred: pred conf with same class shape=[Nx4]
#     """
#     # if ret_linespace is None:
#     #     ret_linespace = torch.linspace(0, 1, steps = 100, device=bboxes_all.device)
#     num_pred = bboxes_pred.shape[0]
#     if num_pred == 0:
#         return 0.0
#     num_gt = bboxes_all.shape[0]
#     # print(num_gt)
#     def iou(x, y):
#         inter, union = _loss_inter_union(x.float(), y.float())
#         return inter / (union + 1e-10)
#     tgt_ious = iou(bbboxes_target.reshape(1, 4).repeat(num_pred, 1), bboxes_pred) # N_pred
#     all_ious = iou(bboxes_all.reshape(-1, 1, 4).repeat(1, num_pred, 1), bboxes_pred.reshape(1, -1, 4).repeat(num_gt, 1, 1)) # N_gt N_pred
#     conf_matched_mx = (conf_pred * (tgt_ious >= iou_thresh)).max(dim = -1)[0]
#     if conf_matched_mx < conf_thresh:
#         return 0.0
#     FP_MAX = (torch.logical_and((all_ious < iou_thresh).all(dim = 0), (conf_pred > conf_matched_mx.reshape(-1)))).sum()
#     # print(conf_thresh, tgt_ious, all_ious, FP_MAX)
#     # exit()
#     # print(num_pred, FP_MAX, conf_thresh)
#     # print(num_pred, FP_MAX, conf_matched_mx)
#     return 1 / (1 + FP_MAX)
#

def write_plt_to_tb(name, writer: SummaryWriter):
    # Convert the Matplotlib figure to an image tensor
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    image_pil = Image.open(buf)
    image_array = np.array(image_pil)
    image = torch.tensor(image_array).permute(2, 0, 1)

    writer.add_image(name, image)


