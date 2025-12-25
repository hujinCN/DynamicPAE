import torch
from . import faster_rcnn, fasterrcnn_resnet50_fpn
from ...base import DetectorBase
import numpy as np
import torchvision
def inter_nms(all_predictions, conf_thres=0.25, iou_thres=0.45):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    max_det = 300  # maximum number of detections per image
    out = []
    for predictions in all_predictions:
        # for each img in batch
        # print('pred', predictions.shape)
        if not predictions.shape[0]:
            out.append(predictions)
            continue
        if type(predictions) is np.ndarray:
            predictions = torch.from_numpy(predictions)
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(0)
        # print(predictions.shape[0])
        scores = predictions[:, 4]
        i = scores > conf_thres

        # filter with conf threshhold
        boxes = predictions[i, :4]
        scores = scores[i]

        # filter with iou threshhold
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        # print('i', predictions[i].shape)
        out.append(predictions[i])
    return out

class TorchFasterRCNN(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_size=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.max_conf_num = 1000

    def load(self, model_weights=None, **args):
        kwargs = {}
        if self.input_tensor_size is not None:
            kwargs['min_size'] = self.input_tensor_size
        if self.cfg.PERTURB.GATE == 'shakedrop':
            from .faster_rcnn import faster_rcnn_resnet50_shakedrop
            self.detector = faster_rcnn_resnet50_shakedrop()
        else:
            self.detector = fasterrcnn_resnet50_fpn(pretrained=True, **kwargs) \
                if model_weights is None else fasterrcnn_resnet50_fpn()

        self.detector = self.detector.to(self.device)
        self.eval()

    def __call__(self, batch_tensor, score='bbox', **kwargs):
        shape = batch_tensor.shape[-2]
        preds, confs = self.detector(batch_tensor)  # the confs is scores from RPN
        bbox_array = []
        score_array = []
        for ind, (pred, now_conf) in enumerate(zip(preds, confs)):
            nums = pred['scores'].shape[0]
            array = torch.cat((
                pred['boxes'] / shape,
                pred['scores'].view(nums, 1),
                (pred['labels'] - 1).view(nums, 1)
            ), 1) if nums else torch.cuda.FloatTensor([]).to(batch_tensor.device)
            bbox_array.append(array)
            if score == "rpn":
                if now_conf.size(0) < self.max_conf_num:
                    now_conf = torch.cat((now_conf, torch.zeros(self.max_conf_num - now_conf.size(0)).to(now_conf.device)),
                                         -1)
                    confs[ind] = now_conf
                now_conf[now_conf < 0.5] = 0
                confs[ind] = torch.mean(now_conf[now_conf > 0])
            elif score == "bbox":
                if pred['scores'].size(0) < self.max_conf_num:
                    score_array.append(torch.cat(
                        (pred['scores'], torch.zeros(self.max_conf_num - pred['scores'].size(0)).to(pred['scores'].device)), -1))

        if score == "rpn":
            # score from the rpn
            confs_array = torch.vstack((confs))
        elif score == "bbox":
            # score from the final bboxes
            confs_array = torch.vstack((score_array))
        output_batches = []
        for batch in bbox_array:
            bboxnum = batch.shape[0]
            if bboxnum == 0:
                batch = torch.zeros((0, 6)).to(batch)
            transformed_batch = torch.zeros((1000, 95), dtype=torch.float32, device=batch.device)
            transformed_batch[:bboxnum, :4] = batch[:, :4]  # 复制xywh
            categories = batch[:, 5].long()  # 获取类别，转换为long类型以用作索引
            scores = batch[:, 4]  # 获取得分
            if bboxnum > 0:
                transformed_batch[torch.arange(bboxnum, dtype=torch.int64, device=batch.device), 4 + categories] = scores
            # for i, category in enumerate(categories):
            #     transformed_batch[i, 4 + category] = scores[i]  # 设置得分
            # assert float((transformed_batch - transformed_batch1)[..., 4:].sum()) < 0.0001
            output_batches.append(transformed_batch)
        stacked_tensor = torch.stack(output_batches, dim=0)
        bbox_array = inter_nms(bbox_array, 0.001, 0.7)
        return (stacked_tensor, bbox_array)
