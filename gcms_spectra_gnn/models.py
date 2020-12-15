import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from functools import partial


dgl_gcn_msg_fxn = fn.copy_src(src='h', out='m')
dgl_gcn_reduce_fxn = fn.sum(msg='m', out='h')


# copy pasta from dgl tutorial
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(dgl_gcn_msg_fxn, dgl_gcn_reduce_fxn)
            h = g.ndata['h']
            return self.linear(h)


class Net(nn.Module):
    def __init__(self, input_features, output_features, agg=None):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(input_features, 16)
        self.layer2 = GCNLayer(16, output_features)
        self.linear_layer = nn.Linear(output_features, output_features)
        if agg is None:
            self.agg = partial(torch.sum, dim=0, keepdims=True)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        x = self.agg(x)
        x = F.relu(self.linear_layer(x))
        return x