import torch

def auction_lap(X, eps=None, compute_score=True):
    device = X.device
    if X.size(0) == 0 or X.size(1) == 0:
        return torch.tensor([], device=device),torch.tensor([], device=device),0

    X = -X
    flag = 0
    if X.size(0) > X.size(1):
        flag = 1
        X = X.transpose(0, 1)
    eps = 1 / X.shape[0] if eps is None else eps

    cost = torch.zeros((1, X.shape[1]), device = device)
    curr_ass = torch.zeros(X.shape[0], device = device).long() - 1
    bids = torch.zeros(X.shape, device = device)

    counter = 0
    while (curr_ass == -1).any():
        counter += 1

        # bidding

        unassigned = (curr_ass == -1).nonzero().squeeze(-1)
        value = X[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)

        first_idx = top_idx[:,0]
        first_value, second_value = top_value[:,0], top_value[:,1]

        bid_increments = first_value - second_value + eps

        bids_ = bids[unassigned]
        bids_.zero_()
        bids_.scatter_(
            dim=1,
            index=first_idx.contiguous().view(-1, 1),
            src=bid_increments.view(-1, 1)
        )

        # assignment

        have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()

        high_bids, high_bidders = bids_[:,have_bidder].max(dim=0)
        high_bidders = unassigned[high_bidders.squeeze()]

        cost[:,have_bidder] += high_bids

        curr_ass[(curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
        curr_ass[high_bidders] = have_bidder.squeeze()

    score = None
    if compute_score:
        score = int(X.gather(dim=1, index=curr_ass.view(-1, 1)).sum())

    if flag == 1:
        return curr_ass, torch.arange(X.size(0), device=device), -score
    return torch.arange(X.size(0), device=device), curr_ass, -score
    # return curr_ass, score, counter