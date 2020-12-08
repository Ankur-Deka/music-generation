import torch.nn.functional as F

def nll_loss(output, target, config=None):
    # Get prediction and target values
    pred = output['out']   # [B, C]
    gt = target['out']    # [B]
    weight = target.get('weight')
    if weight is not None:
        weight = weight[0]
    output = F.nll_loss(F.log_softmax(pred, dim=1), gt, weight=weight)
    return output

