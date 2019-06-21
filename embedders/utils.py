import numpy as np
import pandas as pd


def path_to_embedding(root, method, name, dim):
    return '{}/{}_{}_d{}.csv'.format(root, method, name, dim)


def read_embedding(path):
    X = pd.read_csv(
        path,
        delim_whitespace=True, header=None,
        skiprows=1,
        index_col=0
    ).sort_index()
    X.index.name = None
    return X.values


def save_embedding(path, E, normalize=True):
    if normalize:
        E = E / np.linalg.norm(E, axis=1).reshape((E.shape[0], 1))
    print('Saving results to {}'.format(path))
    N, dim = E.shape
    with open(path, 'w') as file:
        file.write('{} {}\n'.format(N, dim))
        for i in range(N):
            file.write(str(i + 1) + ' ' + ' '.join([str(x) for x in E[i, :]]) + '\n')
