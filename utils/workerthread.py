import threading

class WorkerThread(threading.Thread):
    def __init__(self, work, callback=None):
        threading.Thread.__init__(self)
        self.work = work
        self.callback = callback

    def run(self):
        self.work()
        if self.callback:
            self.callback()


