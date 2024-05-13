from typing import Optional, Union

from mmengine.structures import InstanceData
from torch import Tensor
import torch

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost


@TASK_UTILS.register_module()
class ClassificationCost_(BaseMatchCost):
    """ClsSoftmaxCost.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ...  match_costs.match_cost import ClassificationCost
        >>> import torch
        >>> self = ClassificationCost()
        >>> cls_pred = torch.rand(4, 3)
        >>> gt_labels = torch.tensor([0, 1, 2])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(cls_pred, gt_labels)
        tensor([[-0.3430, -0.3525, -0.3045],
            [-0.3077, -0.2931, -0.3992],
            [-0.3664, -0.3455, -0.2881],
            [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight: Union[float, int] = 1) -> None:
        super().__init__(weight=weight)

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``scores`` inside is
                predicted classification logits, of shape
                (num_queries, num_class).
            gt_instances (:obj:`InstanceData`): ``labels`` inside should have
                shape (num_gt, ).
            img_meta (Optional[dict]): _description_. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_scores = pred_instances.scores
        
        gt_labels = gt_instances.labels

        pred_scores_ = pred_scores.softmax(-1)
        cls_cost = -pred_scores_[:, gt_labels]
        if pred_scores.sum() == 0:
            cls_cost = torch.zeros_like(cls_cost)

        return cls_cost * self.weight