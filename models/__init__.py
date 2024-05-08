from models.touchgesture_model_LSTM import touchgesture_model_LSTM
from models.touchgesture_model_GRU import touchgesture_model_GRU
from models.touchgesture_model_RNN import touchgesture_model_RNN
from models.touchgesture_model_CNN import touchgesture_model_CNN
from torch.optim.lr_scheduler import _LRScheduler

models = {
    'CNN': touchgesture_model_CNN,
    'RNN': touchgesture_model_RNN,
    'LSTM': touchgesture_model_LSTM,
    'GRU': touchgesture_model_GRU
}

class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]