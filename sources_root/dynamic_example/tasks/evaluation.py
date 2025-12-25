from copy import deepcopy
from typing import List

import numpy as np
import torch
import torchmetrics
from matplotlib import pyplot as plt
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.fid import _compute_fid

from dynamic_example.utils.detection_result_ops import calc_precision_targeted, calc_conf_fn, computeAP, write_plt_to_tb


class MAP(MeanAveragePrecision):
    def __init__(self, target_cls = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_cls = target_cls

    def _get_classes(self) -> List:
        return self.target_cls if isinstance(self.target_cls, list) else [self.target_cls]


class PatchEvaluationData:
    def __init__(self, orig_pic, applied_pic,  bbox_all, bbox_target, bbox_results = None, conf_preds = None):
        self.benign_img = orig_pic
        self.atked_img = applied_pic
        self.bbox_target = bbox_target
        self.bbox_all = bbox_all
        self.bbox_results = bbox_results
        self.conf_preds = conf_preds

# class ASR(torchmetrics.Metric):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, pred, target):
#         self.total += pred.size(0)
#         self.correct += (pred == target).sum()

#     def compute(self):
#         return self.correct.float() / self.total


class AttackEvaluationIntegration(torchmetrics.Metric):

    def __init__(self, hparams_evaluation, **kwargs):
        super().__init__(**kwargs)
        self.hparams = hparams_evaluation
        self.SSIM = torchmetrics.image.ssim.StructuralSimilarityIndexMeasure()
        self.MSSSIM = torchmetrics.image.ssim.MultiScaleStructuralSimilarityIndexMeasure()
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity()
        hparams_evaluation["ASR1"] = {
            "name": "ASR 50",
            "type": "asr",
            "threshold": 0.5,
        }
        hparams_evaluation["ASR2"] = {
            "name": "ASR 25",
            "type": "asr",
            "threshold": 0.25,
        }
        hparams_evaluation["ASR3"] = {
            "name": "ASR 10",
            "type": "asr",
            "threshold": 0.1,
        }
        hparams_evaluation["ASR4"] = {
            "name": "ASR 01",
            "type": "asr",
            "threshold": 0.01,
        }
        self.plot_labels = [hparams_evaluation[i].name for i in hparams_evaluation]
        self.ev = [hparams_evaluation[i] for i in hparams_evaluation]
        self.metric_names = []

        self.add_state("TPconf", [], "cat")
        self.add_state("FNconf", [], "cat")
        for i, name in enumerate(hparams_evaluation):
            if self.ev[i].type == "AP":
                self.metric_names.append("_state_"+ name)
                continue
            self.add_state("_state_"+ name, torch.tensor(0.0), "sum")
            self.add_state("square_state_"+ name, torch.tensor(0.0), "sum")
            self.add_state("item_state_" + name, [], "cat")
            # setattr(self, "temp_state_" + name, torch.tensor(0.0))
            self.register_buffer("temp_state_" + name, torch.tensor(0.0), persistent=False)
            # setattr(self, "temp_square_state_" + name, torch.tensor(0.0))
            self.register_buffer("temp_square_state_" + name, torch.tensor(0.0), persistent=False)
            self.metric_names.append("_state_"+ name)


        self.add_state("n_observations", torch.tensor(0), "sum")
        # self.add_state("n_item", torch.tensor(0), "sum")
        self.cur_item = 0

    def get_labels(self):
        return self.plot_labels

    def update(self, data: PatchEvaluationData, fuse = False):
        c, fn = calc_conf_fn(data.bbox_all, data.bbox_target, data.bbox_results, data.conf_preds,
                             iou_thresh=0.5, conf_thresh_min=0.01)
        self.TPconf = self.TPconf + [c.detach()]
        self.FNconf = self.FNconf + [fn.detach()]
        for i, name in enumerate(self.metric_names):
            ev = self.ev[i]

            if ev.type == "AP single":
                result = 0
                # result = calc_precision_targeted(data.bbox_all, data.bbox_target, data.bbox_results, data.conf_preds,
                #                                    iou_thresh=ev.iou, conf_thresh=ev.conf)

            # if ev.type == "mAP":
            #     result = calc_precision_targeted(data.bbox_all, data.bbox_target, data.bbox_results,
            #                                      data.conf_preds,
            #                                      iou_thresh=ev.iou, conf_thresh=ev.conf)
            elif ev.type == "ssim":
                func = lambda x: x if len(x.shape) == 4 else x.unsqueeze(0)
                benign_img = func(data.benign_img)
                atked_img = func(data.atked_img)
                result = self.SSIM(benign_img.double(), atked_img.double())
                result = torch.where(torch.isinf(result), 1.0, result)
                result = torch.where(torch.isnan(result), 1.0, result)
            elif ev.type == "msssim":
                func = lambda x: x if len(x.shape) == 4 else x.unsqueeze(0)
                benign_img = func(data.benign_img)
                atked_img = func(data.atked_img)
                result = self.MSSSIM(benign_img.double(), atked_img.double())
                result = torch.where(torch.isinf(result), 1.0, result)
                result = torch.where(torch.isnan(result), 1.0, result)
            elif ev.type == "lpips":
                func = lambda x: x if len(x.shape) == 4 else x.unsqueeze(0)
                benign_img = func(data.benign_img)
                atked_img = func(data.atked_img)
                result = self.lpips(benign_img, atked_img)
            elif ev.type == "asr":
                # func = lambda x: x if len(x.shape) == 4 else x.unsqueeze(0)
                # benign_img = func(data.benign_img)
                # atked_img = func(data.atked_img)
                result = float(float(c) > ev.threshold)
            # Naturalness:
            else:
                result = -1

            setattr(self, name, getattr(self, name) + result)

            # Welford's method
            cur = getattr(self, f"temp{name}")
            delta = result - cur / self.cur_item if self.cur_item != 0.0 else result
            setattr(self, f"temp{name}", cur + result)
            delta2 = result - (cur + result) / (self.cur_item + 1)
            setattr(self, f"temp_square{name}", getattr(self, f"temp_square{name}") + delta * delta2)

        self.n_observations += 1
        self.cur_item += 1

        if fuse:
            for i, label in enumerate(self.get_labels()):
                if self.ev[i].type == "AP":
                    continue
                name_temp = f"temp{self.metric_names[i]}"
                name_temp_square = f"temp_square{self.metric_names[i]}"
                val = getattr(self, name_temp) / self.cur_item
                val_s = getattr(self, name_temp_square) / (self.cur_item - 1) if self.cur_item > 1 else 0.0
                name = f"item{self.metric_names[i]}"
                name_square = f"square{self.metric_names[i]}"
                setattr(self, name, getattr(self, name) + [val])
                setattr(self, name_square, getattr(self, name_square) + val_s ** .5)
                setattr(self, name_temp, torch.zeros_like(val))
                setattr(self, name_temp_square, torch.zeros_like(val))
                # print(f"fused: {float(val)}")

            self.cur_item = 0

    def reset(self) -> None:
        self.cur_item = 0
        for i, label in enumerate(self.get_labels()):

            if self.ev[i].type == "AP":
                continue
            name_temp = f"temp{self.metric_names[i]}"
            name_temp_square = f"temp_square{self.metric_names[i]}"
            val = getattr(self, name_temp)
            setattr(self, name_temp, torch.zeros_like(val))
            setattr(self, name_temp_square, torch.zeros_like(val))
        super().reset()


    @staticmethod
    def plot_pr(curve, ap, title):
        x = np.linspace(0, 1, curve.shape[0])
        plt.clf()
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
        plt.plot(x, curve)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{title}: {ap}')
        plt.grid(True)
        # plt.show()


    def computeAP(self, tb = None):
        conf = torch.stack(self.TPconf) if isinstance(self.TPconf, list) else self.TPconf
        fn_conf = torch.cat(self.FNconf) if isinstance(self.FNconf, list) else self.FNconf
        curve50, ap50 = computeAP(conf.clone(), fn_conf.clone(), conf_thresh=0.5, return_curve=True)
        curve01, ap01 = computeAP(conf.clone(), fn_conf.clone(), conf_thresh=0.01, return_curve=True)
        if tb is not None:
            AttackEvaluationIntegration.plot_pr(curve50, ap50, "AP50")
            write_plt_to_tb("AP50", tb)
            AttackEvaluationIntegration.plot_pr(curve01, ap01, "AP01")
            write_plt_to_tb("AP01", tb)
        return ap50, ap01

    def compute(self, return_dict=True, tb = None ):
        if return_dict:
            ret = {}

            if len(self.TPconf) == 1:
                return ret
            ret["AP50"], ret["AP01"] = self.computeAP(tb)
            for i, label in enumerate(self.get_labels()):
                ret[label] = float(getattr(self, self.metric_names[i]) / self.n_observations)
                ret["std_" + label] = float(getattr(self, f"square{self.metric_names[i]}") / (self.n_observations))
                item_val = getattr(self, f"item{self.metric_names[i]}")
                if isinstance(item_val, list):
                    item_val = [x.to(self.n_observations.device) for x in item_val]
                    item_val = torch.stack(item_val) if not item_val == [] else torch.zeros_like(self.n_observations).float()
                ret["std_item" + label] = 0.0 # float(torch.std(item_val))
            ret["n"] = float(self.n_observations)
            return ret
        return [float(getattr(self, name) / self.n_observations) for name in self.metric_names]




class FrechetInceptionDistanceDIV(FrechetInceptionDistance):
    r"""Calculate Fréchet inception distance (FID_) which is used to access the quality of generated images.

    .. math::
        FID = \|\mu - \mu_w\|^2 + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})

    where :math:`\mathcal{N}(\mu, \Sigma)` is the multivariate normal distribution estimated from Inception v3
    (`fid ref1`_) features calculated on real life images and :math:`\mathcal{N}(\mu_w, \Sigma_w)` is the
    multivariate normal distribution estimated from Inception v3 features calculated on generated (fake) images.
    The metric was originally proposed in `fid ref1`_.

    Using the default feature extraction (Inception v3 using the original weights from `fid ref2`_), the input is
    expected to be mini-batches of 3-channel RGB images of shape ``(3xHxW)``. If argument ``normalize``
    is ``True`` images are expected to be dtype ``float`` and have values in the ``[0,1]`` range, else if
    ``normalize`` is set to ``False`` images are expected to have dtype ``uint8`` and take values in the ``[0, 255]``
    range. All images will be resized to 299 x 299 which is the size of the original training data. The boolian
    flag ``real`` determines if the images should update the statistics of the real distribution or the
    fake distribution.

    This metric is known to be unstable in its calculatations, and we recommend for the best results using this metric
    that you calculate using `torch.float64` (default is `torch.float32`) which can be set using the `.set_dtype`
    method of the metric.

    .. note:: using this metrics requires you to have torch 1.9 or higher installed

    .. note:: using this metric with the default feature extractor requires that ``torch-fidelity``
        is installed. Either install as ``pip install torchmetrics[image]`` or ``pip install torch-fidelity``

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``imgs`` (:class:`~torch.Tensor`): tensor with images feed to the feature extractor with
    - ``real`` (:class:`~bool`): bool indicating if ``imgs`` belong to the real or the fake distribution

    As output of `forward` and `compute` the metric returns the following output

    - ``fid`` (:class:`~torch.Tensor`): float scalar tensor with mean FID value over samples

    Args:
        feature:
            Either an integer or ``nn.Module``:

            - an integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              64, 192, 768, 2048
            - an ``nn.Module`` for using a custom feature extractor. Expects that its forward method returns
              an ``(N,d)`` matrix where ``N`` is the batch size and ``d`` is the feature size.

        reset_real_features: Whether to also reset the real features. Since in many cases the real dataset does not
            change, the features can be cached them to avoid recomputing them which is costly. Set this to ``False`` if
            your dataset does not change.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If torch version is lower than 1.9
        ModuleNotFoundError:
            If ``feature`` is set to an ``int`` (default settings) and ``torch-fidelity`` is not installed
        ValueError:
            If ``feature`` is set to an ``int`` not in [64, 192, 768, 2048]
        TypeError:
            If ``feature`` is not an ``str``, ``int`` or ``torch.nn.Module``
        ValueError:
            If ``reset_real_features`` is not an ``bool``

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from torchmetrics.image.fid import FrechetInceptionDistance
        >>> fid = FrechetInceptionDistance(feature=64)
        >>> # generate two slightly overlapping image intensity distributions
        >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
        >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
        >>> fid.update(imgs_dist1, real=True)
        >>> fid.update(imgs_dist2, real=False)
        >>> fid.compute()
        tensor(12.7202)

    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    # def __init__(
    #     self,
    #     feature: Union[int, Module] = 2048,
    #     reset_real_features: bool = True,
    #     normalize: bool = False,
    #     **kwargs: Any,
    # ) -> None:
    #     super().__init__(**kwargs)
    #
    #     if not _TORCH_GREATER_EQUAL_1_9:
    #         raise ValueError("FrechetInceptionDistance metric requires that PyTorch is version 1.9.0 or higher.")
    #
    #     if isinstance(feature, int):
    #         num_features = feature
    #         if not _TORCH_FIDELITY_AVAILABLE:
    #             raise ModuleNotFoundError(
    #                 "FrechetInceptionDistance metric requires that `Torch-fidelity` is installed."
    #                 " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
    #             )
    #         valid_int_input = (64, 192, 768, 2048)
    #         if feature not in valid_int_input:
    #             raise ValueError(
    #                 f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
    #             )
    #
    #         self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])
    #
    #     elif isinstance(feature, Module):
    #         self.inception = feature
    #         dummy_image = torch.randint(0, 255, (1, 3, 299, 299), dtype=torch.uint8)
    #         num_features = self.inception(dummy_image).shape[-1]
    #     else:
    #         raise TypeError("Got unknown input to argument `feature`")
    #
    #     if not isinstance(reset_real_features, bool):
    #         raise ValueError("Argument `reset_real_features` expected to be a bool")
    #     self.reset_real_features = reset_real_features
    #
    #     if not isinstance(normalize, bool):
    #         raise ValueError("Argument `normalize` expected to be a bool")
    #     self.normalize = normalize
    #
    #     mx_num_feets = (num_features, num_features)
    #     self.add_state("real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
    #     self.add_state("real_features_cov_sum", torch.zeros(mx_num_feets).double(), dist_reduce_fx="sum")
    #     self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")
    #
    #     self.add_state("fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
    #     self.add_state("fake_features_cov_sum", torch.zeros(mx_num_feets).double(), dist_reduce_fx="sum")
    #     self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")
    #
    # def update(self, imgs: torch.Tensor, real: bool) -> None:
    #     """Update the state with extracted features."""
    #     imgs = (imgs * 255).byte() if self.normalize else imgs
    #     features = self.inception(imgs)
    #     self.orig_dtype = features.dtype
    #     features = features.double()
    #
    #     if features.dim() == 1:
    #         features = features.unsqueeze(0)
    #     if real:
    #         self.real_features_sum += features.sum(dim=0)
    #         self.real_features_cov_sum += features.t().mm(features)
    #         self.real_features_num_samples += imgs.shape[0]
    #     else:
    #         self.fake_features_sum += features.sum(dim=0)
    #         self.fake_features_cov_sum += features.t().mm(features)
    #         self.fake_features_num_samples += imgs.shape[0]

    def compute(self) -> torch.Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()

    # def set_dtype(self, dst_type: Union[str, torch.dtype]) -> "Metric":
    #     """Transfer all metric state to specific dtype. Special version of standard `type` method.
    #
    #     Arguments:
    #         dst_type: the desired type as ``torch.dtype`` or string
    #
    #     """
    #     out = super().set_dtype(dst_type)
    #     if isinstance(out.inception, NoTrainInceptionV3):
    #         out.inception._dtype = dst_type
    #     return out

    def plot(
        self, val = None, ax = None
    ) :
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.image.fid import FrechetInceptionDistance
            >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
            >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
            >>> metric = FrechetInceptionDistance(feature=64)
            >>> metric.update(imgs_dist1, real=True)
            >>> metric.update(imgs_dist2, real=False)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.fid import FrechetInceptionDistance
            >>> imgs_dist1 = lambda: torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
            >>> imgs_dist2 = lambda: torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
            >>> metric = FrechetInceptionDistance(feature=64)
            >>> values = [ ]
            >>> for _ in range(3):
            ...     metric.update(imgs_dist1(), real=True)
            ...     metric.update(imgs_dist2(), real=False)
            ...     values.append(metric.compute())
            ...     metric.reset()
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
