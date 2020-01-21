import numpy as np
from dataset import Dataset


class Parser:

    @staticmethod
    def parse(path, dim_features, dim_out, one_hot=None):
        with open(path, 'r') as file:
            lines = file.readlines()
            if one_hot is not None:
                data = np.zeros((len(lines), one_hot+dim_out))
            else:
                data = np.zeros((len(lines), dim_features + dim_out))
            i = 0
            for line in lines:
                line = line.strip().split(' ')
                if one_hot is None:
                    data[i] = np.array(line[0:dim_features+dim_out])
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
        if one_hot is None:
            return Dataset(dim_features, dim_out, data)
        else:
            return Dataset(one_hot, dim_out, data)



