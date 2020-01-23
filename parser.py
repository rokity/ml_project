import numpy as np
from dataset import Dataset


class Parser:
    def __init__(self, path_tr, path_ts):
        self.path_tr = path_tr
        self.path_ts = path_ts

    def parse(self, dim_features, dim_out, one_hot=None, perc_val=None):
        raise NotImplementedError

    def __parse_file(self, path, dim_features, dim_out, one_hot, perc_val):
        raise NotImplementedError


class Monks_parser(Parser):

    def parse(self, dim_features, dim_out, one_hot=None, perc_val=None):
        training_set, validation_set = self.__parse_file(self.path_tr, dim_features, dim_out, one_hot, perc_val)
        test_set, _ = self.__parse_file(self.path_ts, dim_features, dim_out, one_hot, None)
        return training_set, validation_set, test_set

    def __parse_file(self, path, dim_features, dim_out, one_hot, perc_val):
        with open(path, 'r') as file:
            lines = file.readlines()
            if one_hot is not None:
                data = np.zeros((len(lines), one_hot + dim_out))
            else:
                data = np.zeros((len(lines), dim_features + dim_out))
            i = 0
            for line in lines:
                line = line.strip().split(' ')
                if one_hot is None:
                    data[i] = np.array(line[0:dim_features + dim_out])
                else:
                    data[i, 0:dim_out] = line[0:dim_out]
                    data[i, int(line[1])] = 1
                    data[i, int(line[2]) + 3] = 1
                    data[i, int(line[3]) + 6] = 1
                    data[i, int(line[4]) + 8] = 1
                    data[i, int(line[5]) + 11] = 1
                    data[i, int(line[6]) + 15] = 1
                i += 1
            file.close()
        np.random.shuffle(data)
        if not (one_hot is None):
            dim_features = one_hot
        if not (perc_val is None):
            n = data.shape[0] - int(data.shape[0] * perc_val)
            tr = Dataset(dim_features, dim_out, data[0:n, :])
            vl = Dataset(dim_features, dim_out, data[n:data.shape[0], :])
            return tr, vl
        else:
            return Dataset(dim_features, dim_out, data), None


