from easydict import EasyDict

"""
Hyper-Parameters
"""

cfg = EasyDict({
"exp_name":"my basic model",
"desc": "test training experiment",
"mode": "train",
"debug": False,

"lr": 0.01,

"total_data_path": "",
"train_data_path": "",
"val_data_path": ""
})
