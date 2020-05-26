from tensorflow.keras.callbacks import Callback


from ui.ui_TrainPlot import TrainPlotWindow

class TrainingRecording( Callback):
    def __init__(self, parent ):
        self.parent = parent
        self.plotting_form = TrainPlotWindow(self)
        self.losses = []
        self.interrupt = False

    def on_train_begin(self, logs=None):
        self.plotting_form.show()
        self.plotting_form.refresh()
        self.interrupt = False

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))
        self.plotting_form.plot(self.losses)
        self.plotting_form.refresh()
        if self.interrupt:
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        pass
