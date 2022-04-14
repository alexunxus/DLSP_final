import torch
import json
from tqdm import tqdm


# Trainer class used to train clean classifier
class Trainer:
    
    def __init__(self, train_loader, model, criterion, optim, 
                 nclass, epochs, metric_fns, scheduler=None, val_loader=None, log_path = None, 
                 callbacks=[], verbose = False):
        '''
            Arg:
            train_loader: pytorch data loader for training data
            val_loader  : pytorch data loader for validation data
            model       : baseline model of type nn.Module
            criterion   : take (pred, ground truth) to calculate loss
            optim       : optimizer
            nclass      : number of classes
            epochs      : total epochs to train the model
            scheduler   : pytorch scheduler
            log_path    : path to save log info
            callbacks   : list of callback classes, should contain methods
                          "on_epoch_begin(), on_epoch_end()"
            vervose     : print training bar
        '''
        
        self.train_loader = train_loader
        self.val_loader  = val_loader
        
        self.device    = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.model     = model.to(self.device)
        
        self.criterion = criterion
        self.optim     = optim
        self.scheduler = scheduler
        
        self.nclass    = nclass
        self.epochs    = epochs
        
        self.metric_fns   = metric_fns
        self.history      = {}
        self.test_history = None # dictionary of test metrics
        self.time_history = None # list of time-stamp
        self.log_path     = "./history.json" if log_path is None else log_path
        self.init_history()
        
        self.callbacks = callbacks
        self.register_callbacks()
        
        self.verbose   = verbose
    
    def register_callbacks(self):
        for callback in self.callbacks:
            callback.hook_on_trainer(self)
    
    def init_history(self):
        '''
            initialize history dictionary (tensorflow styled)
            key = loss, metrics....        are for training data
            key = val_loss, val_metrics... are for validation data
        '''
        self.history['loss'] = []
        for metric_fn in self.metric_fns:
            self.history[metric_fn.__name__] = []
        if self.val_loader is not None:
            self.history['val_loss'] = []
            for metric_fn in self.metric_fns:
                self.history["val_" + metric_fn.__name__] = []
    
    def train_one_epoch(self, epoch):
        '''
            Goal: train model for one epoch
            Argument: epoch: int to help specify epoch in tqdm bar
        '''
        # training mode
        self.model.train()
        
        # statistics
        metrics  = {fn.__name__: 0 for fn in (self.metric_fns)}
        acc_loss = 0 # accumulated loss
        length   = len(self.train_loader)
        
        tbar = tqdm(enumerate(self.train_loader)) if self.verbose else enumerate(self.train_loader)
        for i, (X, Y) in tbar:
            i += 1
            X, Y = X.to(self.device), Y.to(self.device)
            
            with torch.set_grad_enabled(True):
                outputs  = self.model(X)
                _, preds = torch.max(outputs, 1)
                loss  = self.criterion(outputs, Y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            # update metrics
            Y = Y.data
            for metric_fn in self.metric_fns:
                metrics[metric_fn.__name__] += (metric_fn(preds, Y).item())
            acc_loss += loss.item()
            
            # logs
            if self.verbose:
                descr = f"[{epoch}][{i}/{len(self.train_loader)}] loss = {acc_loss/i:.4f}"
                for key, val in metrics.items():
                    descr += f", {key} = {val/i:.4f}"
                tbar.set_description(descr)
        
        # on epoch end
        for key, val in metrics.items():
            self.history[key].append(val/length)
        self.history['loss'].append(acc_loss/length)
        
        descr = f"[{epoch}/{self.epochs}] loss = {acc_loss/length:.4f}"
        for key, val in metrics.items():
            descr += f", {key} = {val/length:.4f}"
        print(descr)
    
    def validate(self, loader, val=True):
        '''
            Args: 
                loader: data loader can be validation or test loader
                val:    True if is doing validation(log will be saved in Trainer's log dictionary) / False if doing test
            Return:
                return the test performance if val = False
        '''
        
        # validation mode
        self.model.eval()
        
        # statistics
        metrics  = {fn.__name__: 0 for fn in (self.metric_fns)}
        length   = len(loader) # count number of batches
        acc_loss = 0           # accumulated loss
        
        for i, (X, Y) in enumerate(loader):
            i += 1
            X, Y = X.to(self.device), Y.to(self.device)
            with torch.no_grad():
                outputs  = self.model(X)
                _, preds = torch.max(outputs, 1)
                loss     = self.criterion(outputs, Y)
            
            # update metrics
            Y     = Y.data
            for metric_fn in self.metric_fns:
                metrics[metric_fn.__name__] += (metric_fn(preds, Y).item())
            acc_loss += loss.item()
        
        if val:
            for key, val in metrics.items():
                self.history['val_'+key].append(val/length)
            self.history['val_loss'].append(acc_loss/length)
            # log
            descr = f"[val]val_loss = {acc_loss/length:.4f}"
            for key, val in metrics.items():
                descr += f", val_{key} = {val/length:.4f}"
            print(descr)
        else: 
            test_result = {'test_loss': acc_loss/length}
            for key, val in metrics.items():
                test_result["test_"+key] = val
            # log
            descr = f"[test]test_loss = {acc_loss/length:.4f}"
            for key, val in metrics.items():
                descr += f", test_{key} = {val/length:.4f}"
            print(descr)
            self.test_history = test_result
            return test_result

    def fit(self):
        '''
            interface to train for epochs
        '''
        for epoch in range(1, self.epochs+1):
            self.on_epoch_begin()
            self.train_one_epoch(epoch)
            if self.val_loader:
                self.validate(self.val_loader)
            if self.scheduler:
                self.scheduler.step()
            if not self.on_epoch_end():
                self.save_history()
                break
            self.save_history()
    
    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin()
    
    def on_epoch_end(self):
        res = True
        for callback in self.callbacks:
            if not callback.on_epoch_end():
                res = False
        return res
                
    
    def save_history(self, specify_path = None):
        '''
            Args:
                specify_path: str
                    if save path is specify, then save at the specified path
                    otherwise save at the log_path 
        '''
        history = self.history.copy()
        if self.test_history is not None:
            for key, val in self.test_history.items():
                history[key] = val
        if self.time_history is not None:
            history['time'] = self.time_history
        
        if specify_path:
            with open(self.specify_path, 'w') as f:
                json.dump(history, f)
        else:
            with open(self.log_path, 'w') as f:
                json.dump(history, f)
    
    def load_history(self, history_path):
        '''load history from path'''
        with open(history_path, 'r') as f:
            history = json.load(f)
            self.test_history = {}
            self.time_history = {}
            self.history      = {}
            for key, val in history:
                if 'test' in key:
                    self.test_history[key] = val
                elif key == 'time':
                    self.time_history = val
                else:
                    self.history[key] = val
        
    
    def get_history(self):
        return self.history.copy()