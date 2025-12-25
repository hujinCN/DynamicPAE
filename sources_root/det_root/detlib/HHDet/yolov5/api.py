import os.path

import torch
import sys
# load from YOLOV5
from .yolov5.utils.general import non_max_suppression, scale_coords
from .yolov5.models.experimental import attempt_load  # scoped to avoid circular import
from .yolov5.models.yolo import Model
from .yolov5.models.utils.general import check_yaml

# sys.path.insert(0, "/home/hujin/zjk/Datk20240117/sources_root/det_root/detlib/HHDet/yolov5/yolov5")
print(sys.path)
from ...base import DetectorBase


class HHYolov5(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_size=640,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.imgsz = (input_tensor_size, input_tensor_size)
        self.stride, self.pt = None, None

    def load_(self, model_weights, **args):
        w = str(model_weights[0] if isinstance(model_weights, list) else model_weights)
        self.detector = attempt_load(model_weights if isinstance(model_weights, list) else w,
                                     map_location=self.device, inplace=False)
        self.eval()
        self.stride = max(int(self.detector.stride.max()), 32)  # model stride
        self.names = self.detector.module.names if hasattr(self.detector,
                                                           'module') else self.detector.names  # get class names

    def load(self, model_weights, **args):
        model_config = args['model_config']
        # Create model
        current_folder_path = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(current_folder_path, "yolov5"))
        self.detector = Model(model_config).to(self.device)
        self.detector.load_state_dict(torch.load(model_weights, map_location=self.device, weights_only=False)['model'].float().state_dict())
        sys.path.pop(0)
        self.eval()

    def __call__(self, batch_tensor, **kwargs):
        # print('yolov5 api call')
        detections_with_grad = self.detector(batch_tensor, augment=False, visualize=False)[0]
        tmp = torch.mul(detections_with_grad[..., 5:], detections_with_grad[..., 4:5])
        return torch.cat((detections_with_grad[..., :4], tmp), dim=-1)
