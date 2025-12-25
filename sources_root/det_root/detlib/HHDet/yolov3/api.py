import torch

from .PyTorch_YOLOv3.pytorchyolo.models import load_model
from .PyTorch_YOLOv3.pytorchyolo.utils.utils import rescale_boxes, non_max_suppression
from ...base import DetectorBase


class HHYolov3(DetectorBase):
    def __init__(self,
                 name, cfg, input_tensor_size=412,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.target = None

    def requires_grad_(self, state: bool):
        self.detector.module_list.requires_grad_(state)

    def load(self, model_weights, detector_config_file=None):
        self.detector = load_model(model_path=detector_config_file, weights_path=model_weights).to(self.device)
        self.eval()

    def __call__(self, batch_tensor: torch.tensor, **kwargs):
        detections_with_grad = self.detector(batch_tensor)  # torch.tensor([1, num, classes_num+4+1])
        tmp = torch.mul(detections_with_grad[..., 5:], detections_with_grad[..., 4:5])
        return torch.cat((detections_with_grad[..., :4], tmp), dim=-1)