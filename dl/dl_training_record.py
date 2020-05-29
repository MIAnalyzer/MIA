from tensorflow.keras.callbacks import Callback


from ui.ui_TrainPlot import TrainPlotWindow

class TrainingRecording( Callback):
    def __init__(self, parent ):
        self.parent = parent
        self.plotting_form = TrainPlotWindow(self)
        self.loss = []
        self.metric = []
        self.interrupt = False

    def on_train_begin(self, logs=None):
        self.plotting_form.show()
        self.plotting_form.initialize()
        self.interrupt = False

    def on_batch_end(self, batch, logs=None):
        self.loss.append(logs.get('loss'))
        self.metric.append(logs.get(self.parent.Model.metrics_names[1]))
        self.plotting_form.refresh()
        if self.interrupt:
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        pass
