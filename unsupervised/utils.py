import torch
from torch_geometric.data import Batch
import numpy as np


def norm(X):
    return sum([sum(abs(i) for i in x) for x in X])


