import lap
import torch
from scipy.optimize import linear_sum_assignment
# from lapsolver import solve_dense
# from .lap import auction_lap
from piqn.auction import auction_lap
from .lap1 import JV_lap
from lap import lapjv
from torch import nn
import numpy as np

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_span: float = 1, solver="lapjv"):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.solver = solver

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        if self.solver == "order":
            sizes = targets["sizes"]
            indices = [(list(range(size)),list(range(size))) for size in sizes]
        else:
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(dim=-1) # [batch_size * num_queries, 8]

            entity_left = outputs["pred_left"].flatten(0, 1)
            entity_right = outputs["pred_right"].flatten(0, 1) # [batch_size * num_queries]


            gt_ids = targets["labels"]
            gt_left = targets["gt_left"]
            gt_right = targets["gt_right"]
            
            # import pdb;pdb.set_trace()
            cost_class = -out_prob[:, gt_ids]
            cost_span = -(entity_left[:, gt_left] + entity_right[:, gt_right])

            # Final cost matrix
            C = self.cost_span * cost_span + self.cost_class * cost_class

            C = C.view(bs, num_queries, -1)

            sizes = targets["sizes"]
            indices = None
            
            if self.solver == "hungarian":
                C = C.cpu()
                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            if self.solver == "auction":
                indices = [auction_lap(c[i])[:2] for i, c in enumerate(C.split(sizes, -1))]

            if self.solver == "lapjv":
                # C = C.cpu().numpy()
                # C = np.reshape(C, (1, 2700))
                # indices = [lapjv(c.squeeze(), return_cost=False, extend_cost=True) for c in C]

                # indices = [lap.lapjv(c.squeeze(), return_cost=False, extend_cost=True) for c in C]

                C = C.cpu()
                indices = [JV_lap(C)]

                # indices = [lapjv(c.squeeze(), return_cost=False) for c in C]

                # indices = [lapjv(c[i], return_cost=False, extend_cost=True) for i, c in enumerate(np.split(C, 45, -1))]

                # indicess = [i != -1 for i, j in indices]
                #
                # indices1 = [(i[0], j) for i, j in indices if i[0] != -1]
                #
                # indices2 = [(tuple(x for x in i if x != -1), j) for i, j in indices]
                #
                # indices3 = [((x for x in i if x != -1), j) for i, j in indices]
                #
                # indexes = [i for i, j in enumerate(indices) if i != -1 for i, j in indices]
                # values = [j for i, j in enumerate(indices) if j != -1 for i, j in indices]
                #
                # indicesss = [[i for i, j in enumerate(index) if j != -1] for index in indices[0]]
                #
                # values1 = [[j for i, j in enumerate(index) if j != -1] for index in indices[0]]
                #
                # indicessss = [[i for i, j in enumerate(index) if i != -1] for index in indices[0]]

                # indicesssss = [[(i, j) for i, j in enumerate(index) if j != -1] for index in indices[0]]
                #
                # i = [i[0] for i in indicesssss[0]]
                # j = [i[1] for i in indicesssss[0]]
                #
                # indices = [(i, j)]





                # 将维度修改为45×45，

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
