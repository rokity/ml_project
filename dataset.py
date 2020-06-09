import numpy as np


class Dataset:
    def __init__(self, dim_in, dim_out, data):
        """

        @param dim_in: input dimension
        @param dim_out: output dimension
        @param data: dataset
        """
        self.size = data.shape[0]
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.data_out = data[:, 0:dim_out]
        self.data_in = data[:, dim_out:dim_in+dim_out]

    def get_data(self, i):
        """

        @param i: index of the row
        @return: input sample_i, output sample_i
        """
        return self.data_in[i], self.data_out[i]

    def features_scaling(self):
        average = np.empty((1, self.dim_in))
        for i in range(self.dim_in):
            average[0, i] = self.data_in[:, i].sum() / self.size
        for i in range(self.size):
            self.data_in[i, :] -= average.reshape(self.dim_in)
        return average

    def features_scaling_avg(self, average):
        for i in range(self.size):
            self.data_in[i, :] -= average.reshape(self.dim_in)

    def normalize_out_classification(self, prev_val, new_val):
        """

        @param prev_val: old value
        @param new_val: new value

        Used to change output value in order to match with some activation functions
        (e.g 0 -> -1 for tanh)
        """
        for i in range(self.size):
            if self.data_out[i, 0] == prev_val:
                self.data_out[i, 0] = new_val

