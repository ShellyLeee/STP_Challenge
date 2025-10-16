import torch.nn as nn

def build_criterion(cfg):
    loss_name = cfg.LOSS.NAME
    if loss_name == 'ce':
        criterion = nn.CrossEntropyLoss()

    else:
        raise NotImplementedError("Not implement this loss function")
    
    return criterion