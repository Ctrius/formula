import lap
import torch

def JV_lap(X):
    device = X.device
    if X.size(0) == 0 or X.size(1) == 0:
        return torch.tensor([], device=device),torch.tensor([], device=device),0

    X = -X
    flag = 0
    if X.size(0) > X.size(1):
        flag = 1
        X = X.transpose(0, 1)

    X = X.numpy()

    indices = [lap.lapjv(c.squeeze(), return_cost=False, extend_cost=True) for c in X]

    indicesssss = [[(i, j) for i, j in enumerate(index) if j != -1] for index in indices[0]]

    i = [i[0] for i in indicesssss[0]]
    j = [i[1] for i in indicesssss[0]]

    indices = [(i, j)]

    indices = [i, j]

    indices = torch.tensor(indices)

    # return torch.arange(indices.size(0), device=device)

    return indices

