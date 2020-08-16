from tensorflow.keras.callbacks import Callback


class TrainingRecording( Callback):
    def __init__(self, parent ):
        self.parent = parent
        self.loss = []
        self.loss_per_epoch = []
        
        self.metric = []
        self.metric_per_epoch = []
        
        self.val_loss = []
        self.val_metric = []
        self.interrupt = False
        self.currentbatch = 0
        self.currentepoch = 0
        # to do: implement validation loss and metric

    def on_train_begin(self, logs=None):
        self.parent.notifyTrainingStarted()
        self.interrupt = False
        self.currentepoch = 0
        self.currentbatch = 0
        
    def on_train_end(self, logs=None):
        self.parent.notifyTrainingFinished()

    def on_batch_end(self, batch, logs=None):
        self.loss.append(logs.get('loss'))
        self.metric.append(logs.get(self.parent.Model.metrics_names[1]))
        self.currentbatch = batch
        self.parent.notifyTrainingProgress()
        if self.interrupt:
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            return 
        self.currentepoch = epoch + 1
        self.parent.notifyEpochEnd(epoch)
        l = logs.get('val_loss')
        m = logs.get('val_' + self.parent.Model.metrics_names[1])
        lpe = logs.get('loss')
        mpe = logs.get(self.parent.Model.metrics_names[1])
        
        if l:
            self.val_loss.append(l)
        if m:
            self.val_metric.append(m)
        if lpe:
            self.loss_per_epoch.append(lpe)
        if mpe:
            self.metric_per_epoch.append(mpe)

    def reset(self):
        self.loss = []
        self.loss_per_epoch = []
        self.metric = []
        self.metric_per_epoch = []
        self.val_loss = []
        self.val_metric = []
        self.currentepoch = 0
        self.currentbatch = 0

