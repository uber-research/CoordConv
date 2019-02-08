import numpy as np
from scipy import stats
from IPython import embed



class BaseLRPolicy(object):
    '''Abstract base class.'''

    def train_more(self, buddy):
        return (self.get_lr(buddy) is not None)

    def train_done(self, buddy):
        return not self.train_more(buddy)


class LRPolicyConstant(BaseLRPolicy):
    '''Constant LR over time.'''

    def __init__(self, args):
        self.lr = args.lr
        self.epochs = args.epochs

    def get_lr(self, buddy):
        '''Returns None if training should end.'''
        if buddy.epoch < self.epochs:
            return self.lr


class LRPolicyStep(BaseLRPolicy):
    '''Steps LR over time'''

    def __init__(self, args):
        self.lr = args.lr
        self.epochs = args.epochs
        self.step_ratio = args.lrstepratio
        self.max_steps = args.lrmaxsteps
        self.step_every = args.lrstepevery
        self.n_steps = 0
        self.last_step_at = 0
        
    def get_lr(self, buddy):
        if buddy.epoch >= self.epochs:
            return None
        if self.n_steps < self.max_steps and buddy.epoch - self.last_step_at >= self.step_every:
            new_lr = self.lr * self.step_ratio
            print 'Changing LR from %g to %g' % (self.lr, new_lr)
            self.lr = new_lr
            self.n_steps += 1
            self.last_step_at = buddy.epoch
        return self.lr


class LRPolicyDecay(BaseLRPolicy):
    '''Steps LR over time'''

    def __init__(self, args, decay_per_iter=True):
        self.base_lr = args.lr
        self.lr_decay = args.lr_decay
        self.lr_min = args.lr_min
        self.epochs = args.epochs
        self.decay_per_iter = decay_per_iter
        
    def get_lr(self, buddy):
        if buddy.epoch >= self.epochs:
            return None
        n_steps = buddy.train_iter if self.decay_per_iter else buddy.epoch
        return max(self.lr_min, self.base_lr * self.lr_decay ** n_steps)


class LRPolicyValStep(BaseLRPolicy):
    '''Steps LR over time'''

    def __init__(self, args):
        self.lr = args.lr
        self.epochs = args.epochs
        self.step_ratio = args.lrstepratio
        self.max_steps = args.lrmaxsteps
        self.step_min_epochs = args.lrstepminepochs
        self.step_min_points = args.lrstepminpoints
        self.n_steps = 0
        self.last_step_at = 0
        
    def get_lr(self, buddy):
        if buddy.epoch >= self.epochs:
            return None
        num_epochs_this_lr = buddy.epoch - self.last_step_at
        if num_epochs_this_lr < self.step_min_epochs:
            return self.lr

        bd = buddy.data_per_iter()

        embed()
        
        n_this = FINISH_THIS
        slope, intercept, r_value, p_value, std_err = stats.linregress(bd['val_loss']['iter'][-n_this:], bd['val_loss']['val'][-n_this:])

        if self.n_steps < self.max_steps and buddy.epoch - self.last_step_at >= self.step_every:
            new_lr = self.lr * self.step_ratio
            print 'Changing LR from %g to %g' % (self.lr, new_lr)
            self.lr = new_lr
            self.n_steps += 1
            self.last_step_at = buddy.epoch
        return self.lr

        
    
