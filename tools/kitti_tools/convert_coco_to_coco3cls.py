import torch
import sys

in_fn = sys.argv[1]
out_fn = sys.argv[2]
data = torch.load(in_fn)
data['state_dict']['bbox_head.fc_cls.weight'] = data['state_dict']['bbox_head.fc_cls.weight'][[0, 3, 1, 2]]
data['state_dict']['bbox_head.fc_cls.bias'] = data['state_dict']['bbox_head.fc_cls.bias'][[0, 3, 1, 2]]
data['state_dict']['bbox_head.fc_reg.weight'] = data['state_dict']['bbox_head.fc_reg.weight'][[
    0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11]]
data['state_dict']['bbox_head.fc_reg.bias'] = data['state_dict']['bbox_head.fc_reg.bias'][[
    0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11]]
torch.save(data, out_fn)
