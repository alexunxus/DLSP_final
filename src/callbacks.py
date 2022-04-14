import numpy as np
import torch
from time import time 

class EarlyStoppingCallback:
    
    def __init__(self, 
                 monitor="val_loss",
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode="auto",
                 restore_best_weights=False):
        
        self.monitor  = monitor
        self.patience = patience
        self.counter  = 0
        self.min_delta= min_delta
        self.verbose  = verbose
        if mode not in ['auto', 'min', 'max']:
            raise ValueError(f"Unknown mode {mode}")
        if 'loss' in monitor:
            self.mode = 'min'
        elif 'acc' in monitor:
            self.mode = 'max'
        else:
            self.mode = 'max'
        self.ref_trainer = None
        self.best = np.inf if self.mode == 'min' else -np.inf
    
    def hook_on_trainer(self, trainer):
        self.ref_trainer = trainer
        if self.monintor not in trainer.history.keys():
            raise ValueError(f"Unseen monitor criterion {self.monitor}")
    
    def on_epoch_begin(self):
        pass
    
    def on_epoch_end(self):
        
        latest = self.ref_trainer.history[self.monitor][-1]
        if self.mode == 'max':
            if  latest > self.best + self.min_delta:
                self.counter = 0
                self.best = latest
            else:
                self.counter += 1
        else:
            if latest < self.bmin_delta -self.min_delta:
                self.counter = 0
                self.best = latest
            else:
                self.counter += 1
        if self.counter >= self.patience:
            if self.verbose:
                print(f"Patience = {self.counter}, early stopping")
            return False
        if self.verbose:
            print(f"Patience = {self.counter}, best: {self.best: .4f}")
        return True

class CheckpointCallback:
    
    def __init__(self,
                 filepath,
                 monitor="val_loss",
                 verbose=0,
                 save_best_only=True,
                 mode="auto"):
        self.filepath = filepath
        self.monitor  = monitor
        self.verbose  = verbose
        self.save_best_only = save_best_only
        self.mode     = None
        if 'loss' in monitor:
            self.mode = 'min'
        elif 'acc' in monitor:
            self.mode = 'max'
        else:
            self.mode = 'max'
        self.ref_trainer = None
        self.best = np.inf if self.mode == 'min' else -np.inf
    
    def hook_on_trainer(self, trainer):
        self.ref_trainer = trainer
        if self.monitor not in trainer.history.keys():
            raise ValueError(f"Unseen monitor criterion {self.monitor}")
    
    def on_epoch_begin(self):
        pass
    
    def on_epoch_end(self):
        latest = self.ref_trainer.history[self.monitor][-1]
        if not self.save_best_only:
            torch.save(self.ref_trainer.model.state_dict(), self.filepath)
            return True
        if self.mode == 'max':
            if latest > self.best:
                self.best = latest
                torch.save(self.ref_trainer.model.state_dict(), self.filepath)
        else:
            if latest < self.best:
                self.best = latest
                torch.save(self.ref_trainer.model.state_dict(), self.filepath)
        return True
            
class TimeCallback:
    
    def __init__(self):
        self.ref_trainer = None
        self.t_start     = 0
        
    def hook_on_trainer(self, trainer):
        self.ref_trainer = trainer
        self.ref_trainer.time_history = []
    
    def on_epoch_begin(self):
        self.t_start = time()
    
    def on_epoch_end(self):
        self.ref_trainer.time_history.append(time() - self.t_start)
        return True

class TimeToAccuracyCallback:

    def __init__(self,
                 monitor="acc",
                 threshold=0.92):
        self.monitor  = monitor
        self.threshold = threshold
        self.ref_trainer = None
        
    def hook_on_trainer(self, trainer):
        self.ref_trainer = trainer
        if self.monitor not in trainer.history.keys():
            raise ValueError(f"Unseen monitor criterion {self.monitor}")
    
    def on_epoch_begin(self):
        pass
    
    def on_epoch_end(self):
        latest = self.ref_trainer.history[self.monitor][-1]
        if latest >= self.threshold:
            print(f"The model has reach {self.threshold} on {self.monitor}, stopping......")
            return False
        return True