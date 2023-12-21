import torch


# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    # SR = SR > threshold
    # GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2)
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(pr, gt, eps=1e-6):
    # Sensitivity == Recall
    # SR = SR > threshold
    # GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return float(score)


def get_specificity(SR, GT, threshold=0.5):
    # SR = SR > threshold
    # GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = torch.sum((SR + GT) == 0)
    # TN = ((SR == 0) + (GT == 0)) == 2
    # FP = ((SR == 1) + (GT == 0)) == 2
    FP = torch.sum((SR + 1 - GT) == 2)

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(pr, gt, eps=1e-6):
    # SR = SR > threshold
    # GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT)
    PC = get_precision(SR, GT)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    # SR = SR > threshold
    # GT = GT == torch.max(GT)

    Inter = torch.sum((SR + GT) == 2)
    Union = torch.sum((SR + GT) >= 1)

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    # SR = SR > threshold
    # GT = GT == torch.max(GT)

    Inter = torch.sum((SR + GT) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC