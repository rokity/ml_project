import sys


class GL:

    def __init__(self, alpha=4, metric='val_loss', patience=1, verbose=False):
        self.metric = metric
        self.alpha = alpha
        self.min_metric = sys.float_info.max
        self.patience = patience
        self.verbose = verbose

    def get_metric(self):
        return self.metric

    def early_stopping_check(self, metric):
        gl = 100 * ((metric / self.min_metric) - 1)
        if self.verbose:
            print("GL: ", gl)
            print("patience (remaining): ", self.patience)

        if gl > self.alpha:
            self.patience -= 1
        if self.patience == 0:
            return True
        self.min_metric = min(self.min_metric, metric)
        return False
