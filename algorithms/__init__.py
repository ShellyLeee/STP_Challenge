from .train import main as train_main
from .evaluate import main as evaluate_main
from .predict import main as predict_main

__all__ = ['train_main', 'evaluate_main', 'predict_main']