import threading

class WorkerThread(threading.Thread):
    def __init__(self, work, args=None, callback_start=None,callback_end=None):
        threading.Thread.__init__(self)
        self.work = work
        self.callback_end = callback_end
        self.callback_start = callback_start
        self.arguments = args
        
    def run(self):
        if self.callback_start:
            self.callback_start()
        if self.arguments:
            self.work(self.arguments)
        else:
            self.work()
        if self.callback_end:
            self.callback_end()


