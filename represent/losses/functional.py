import torch

import torch.nn.functional as F

from torchtyping import TensorType


def soft_dice_loss(
    y_pred: TensorType,
    y_true: TensorType,
    smooth: float = 1e-7,
    reduction: str = "mean",
):
    # flatten label and prediction tensors
    y_true = y_true.view(y_true.size(0), y_true.size(1), -1)
    y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)

    # intersection is equivalent to True Positive count
    # calculate as dot product per slice
    intersection = y_true * y_pred

    # calculate the dice score per slice
    score = (2.0 * intersection.sum(-1) + smooth) / (y_true.sum(-1) + y_pred.sum(-1) + smooth)

    # reduce the scores over all classes and batch
    if reduction == "mean":
        score = score.mean()
    elif reduction == "sum":
        score = score.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Invalid reduction method {reduction}")

    return 1 - score


def soft_dice_loss_with_logits(
    y_pred: TensorType,
    y_true: TensorType,
    smooth: float = 1e-7,
    reduction: str = "mean",
):
    y_pred = torch.sigmoid(y_pred)
    return soft_dice_loss(y_pred, y_true, smooth, reduction)


def infoNCE(
    q: TensorType,
    k: TensorType,
    queue: TensorType,
    tau: float = 0.07,
    return_logits: bool = False,
):
    # compute logits
    # Einstein sum is more intuitive
    # positive logits: Nx1
    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
    # negative logits: NxK
    l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])

    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)

    # apply temperature
    logits /= tau

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long)
    labels = labels.type_as(logits)
    loss = F.cross_entropy(logits.float(), labels.long())

    if return_logits:
        return loss, labels, logits
    else:
        return loss


def deepclusterv2_loss_func(
    outputs: torch.Tensor, assignments: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes DeepClusterV2's loss given a tensor containing logits from multiple views
    and a tensor containing cluster assignments from the same multiple views.

    Args:
        outputs (torch.Tensor): tensor of size PxVxNxC where P is the number of prototype
            layers and V is the number of views.
        assignments (torch.Tensor): tensor of size PxVxNxC containing the assignments
            generated using k-means.
        temperature (float, optional): softmax temperature for the loss. Defaults to 0.1.

    Returns:
        torch.Tensor: DeepClusterV2 loss.
    """
    loss = 0
    for h in range(outputs.size(0)):
        scores = outputs[h].view(-1, outputs.size(-1)) / temperature
        targets = assignments[h].repeat(outputs.size(1)).to(outputs.device, non_blocking=True)
        loss += F.cross_entropy(scores, targets, ignore_index=-1)
    return loss / outputs.size(0)
