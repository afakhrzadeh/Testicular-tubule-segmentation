import torch

SMOOTH = 1e-6

def dice_score(logits, targets):
    """
    logits: [B,1,H,W]
    targets: [B,1,H,W] or [B,H,W]
    """

    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1).float()

    intersection = (preds * targets).sum(dim=1)
    dice = (2 * intersection + SMOOTH) / (
        preds.sum(dim=1) + targets.sum(dim=1) + SMOOTH
    )

    return dice.mean().item()


def iou_score(logits, targets):

    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1).float()

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean().item()


def accuracy_score(logits, targets):

    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    correct = (preds == targets).float().sum()
    total = torch.numel(preds)

    return (correct / total).item()
