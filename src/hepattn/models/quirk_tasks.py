"""Quirk-specific auxiliary tasks (burzynski-lab fork).

Both tasks are pure loss terms: they own no networks, read other tasks'
outputs through ``layer_outputs`` (the same mechanism as IoUPredictionTask)
and are fully steerable from the config for ablation studies.
"""

import torch
from torch import Tensor

from hepattn.models.task import Task


class ObjectPairRegressionConsistencyTask(Task):
    """Penalise disagreement between the valid objects' regression vectors.

    Physics use: the two quirks of a pair share one oscillation plane, so
    their per-query plane-normal predictions must agree (up to sign). For
    every event with >= 2 valid targets, take the matched-valid queries'
    regression vectors, optionally normalise, sign-align to the first, and
    apply an L2 consistency loss.
    """

    def __init__(
        self,
        name: str,
        regr_task_name: str,
        regr_output_key: str,
        target_object: str = "particle",
        loss_weight: float = 1.0,
        normalize: bool = True,
        has_intermediate_loss: bool = False,
    ):
        super().__init__(has_intermediate_loss=has_intermediate_loss)
        self.name = name
        self.regr_task_name = regr_task_name
        self.regr_output_key = regr_output_key
        self.target_object = target_object
        self.loss_weight = loss_weight
        self.normalize = normalize
        self.inputs = []
        self.outputs = []

    def forward(self, x: dict[str, Tensor], outputs=None) -> dict[str, Tensor]:  # noqa: ARG002
        return {}

    def predict(self, outputs, query_mask=None):  # noqa: ARG002
        return {}

    def loss(self, outputs, targets, layer_outputs=None):  # noqa: ARG002
        if layer_outputs is None or self.regr_task_name not in layer_outputs:
            raise ValueError(f"Task '{self.regr_task_name}' not found in layer_outputs")
        vec = layer_outputs[self.regr_task_name][self.regr_output_key]  # (B, N, D)
        valid = targets[self.target_object + "_valid"].bool()  # (B, N)

        losses = []
        for b in range(vec.shape[0]):
            v = vec[b][valid[b]]
            if v.shape[0] < 2:
                continue
            if self.normalize:
                v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            ref = v[0]
            # sign-align each vector to the reference (a plane normal's sign
            # is a convention)
            sign = torch.sign((v @ ref)).clamp(min=-1).unsqueeze(-1)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            losses.append(((v * sign - ref) ** 2).sum(-1)[1:].mean())
        if not losses:
            loss = vec.sum() * 0.0
        else:
            loss = torch.stack(losses).mean()
        return {"pair_consistency": self.loss_weight * loss}


class MaskPlaneCoplanarityTask(Task):
    """Mask-probability-weighted coplanarity of the claimed hits.

    For each valid query: take its predicted hit mask probabilities and its
    predicted plane normal, and penalise the weighted mean squared distance
    of hits to the plane through their weighted centroid. Claimed hits that
    are off the predicted plane (e.g. picked up from an unrelated detector
    region) are punished; genuine quirk hits on the far side of the detector
    are not, because they lie in the plane.
    """

    def __init__(
        self,
        name: str,
        mask_task_name: str,
        mask_logit_key: str,
        plane_task_name: str,
        plane_output_key: str,
        input_constituent: str = "hit",
        position_fields: tuple[str, str, str] = ("x", "y", "z"),
        position_scale: float = 0.001,  # mm -> m, keeps the loss O(1)
        target_object: str = "particle",
        loss_weight: float = 1.0,
        has_intermediate_loss: bool = False,
    ):
        super().__init__(has_intermediate_loss=has_intermediate_loss)
        self.name = name
        self.mask_task_name = mask_task_name
        self.mask_logit_key = mask_logit_key
        self.plane_task_name = plane_task_name
        self.plane_output_key = plane_output_key
        self.input_constituent = input_constituent
        self.position_fields = position_fields
        self.position_scale = position_scale
        self.target_object = target_object
        self.loss_weight = loss_weight
        self.inputs = []
        self.outputs = []

    def forward(self, x: dict[str, Tensor], outputs=None) -> dict[str, Tensor]:  # noqa: ARG002
        return {}

    def predict(self, outputs, query_mask=None):  # noqa: ARG002
        return {}

    def loss(self, outputs, targets, layer_outputs=None):  # noqa: ARG002
        for tname in (self.mask_task_name, self.plane_task_name):
            if layer_outputs is None or tname not in layer_outputs:
                raise ValueError(f"Task '{tname}' not found in layer_outputs")
        w = layer_outputs[self.mask_task_name][self.mask_logit_key].sigmoid()  # (B, N, M)
        n = layer_outputs[self.plane_task_name][self.plane_output_key]  # (B, N, 3)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        pos = torch.stack(
            [targets[f"{self.input_constituent}_{f}"] for f in self.position_fields], dim=-1
        ).type_as(w) * self.position_scale  # (B, M, 3)
        hit_pad = targets.get(f"{self.input_constituent}_valid")
        if hit_pad is not None:
            w = w * hit_pad.unsqueeze(1).type_as(w)

        wsum = w.sum(-1).clamp_min(1e-6)  # (B, N)
        centroid = torch.einsum("bnm,bmc->bnc", w, pos) / wsum.unsqueeze(-1)
        rel = pos.unsqueeze(1) - centroid.unsqueeze(2)  # (B, N, M, 3)
        dist = torch.einsum("bnmc,bnc->bnm", rel, n)  # signed distance to plane
        per_query = (w * dist**2).sum(-1) / wsum  # (B, N)

        valid = targets[self.target_object + "_valid"].bool()
        query_mask = targets.get("query_mask")
        if query_mask is not None:
            valid = valid & query_mask.bool()
        per_query = per_query[valid]
        loss = per_query.mean() if per_query.numel() else w.sum() * 0.0
        return {"coplanarity": self.loss_weight * loss}
