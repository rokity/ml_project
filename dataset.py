import numpy as np

class Dataset:
    def __init__(self, dim_in, dim_out, data):
        self.size = data.shape[0]
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.data_out = data[:, 0:dim_out]
        self.data_in = data[:, dim_out:dim_in+dim_out]

    def get_data(self, i):
        return self.data_in[i], self.data_out[i]

    def features_scaling(self):
        average = np.empty((1, self.dim_in))
        for i in range(self.dim_in):
            sum = self.data_in[:, i].sum()
            average[0, i] = sum / self.size
        for i in range(self.size):
            self.data_in[i, :] -= average.reshape(self.dim_in)
        return average

    def features_scaling_avg(self, average):
        for i in range(self.size):
            self.data_in[i, :] -= average.reshape(self.dim_in)

    def normalize_out(self):
        for i in range(self.size):
            if self.data_out[i, 0] == 0:
                self.data_out[i, 0] = -1

