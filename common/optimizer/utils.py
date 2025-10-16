import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

def build_optimizer(cfg, model):
    optim_name = cfg.OPTIM.NAME
    if optim_name == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)  
    elif optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(),
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError("Not implement this optimizer")
    
    return optimizer


def build_scheduler(cfg, optimizer):
    scheduler_name = cfg.SCHEDULER.NAME
    if scheduler_name == 'step':
        scheduler = StepLR(optimizer, step_size=cfg.SCHEDULER.STEP, gamma=cfg.SCHEDULER.GAMMA)
    elif scheduler_name == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.EPOCH)
    else:
        raise NotImplementedError("Not implement this scheduler")
    
    return scheduler


    


