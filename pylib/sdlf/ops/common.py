import sys
import re
import yaml
import pickle
import importlib

from torchplus.train import latest_checkpoint, restore


def read_txt(path):
    with open(path, 'r') as f:
        da = f.readlines()
    return da


def read_pkl(path):
    with open(path, 'rb') as f:
        da = pickle.load(f)
    return da


def read_config(path):
    with open(path, 'r') as f:
        da = yaml.load(f, Loader=yaml.FullLoader)
    return da


def get_class(class_str):
    mod_name, class_name = class_str.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, class_name)
    return cls


def get_input(ret_dict, in_key):
    in_key_levels = in_key.split('.')
    res = ret_dict[in_key_levels[0]]
    for key in in_key_levels[1:]:
        res = res[key]
    return res


def latest_checkpoints_with_step(result_dir, name):
    ckpt = latest_checkpoint(result_dir, name)
    step = None if ckpt is None else int(re.split('[-.]', ckpt)[-2])
    return ckpt, step


def try_restore_latest_checkpoints_(result_dir, model, optimizer):
    latest_model_ckpt, model_step = latest_checkpoints_with_step(result_dir, model.name)
    latest_optimizer_ckpt, optimizer_step = latest_checkpoints_with_step(result_dir, optimizer.name)
    assert model_step == optimizer_step, 'model and optimizer do not match'
    if latest_model_ckpt and latest_optimizer_ckpt:
        restore(latest_model_ckpt, model, None)
        restore(latest_optimizer_ckpt, optimizer, None)
    return model_step


class Logger(object):
    """ simple logger
    """

    def __init__(self, log_file='log.txt'):
        self.log = open(log_file, 'w')
        self.terminal = None
        self.bind_stdout = False

    def __del__(self):
        self.log.close()
        if self.bind_stdout:
            self.release()

    def bind(self):
        if not self.bind_stdout:
            self.terminal = sys.stdout
            sys.stdout = self
            self.bind_stdout = True

    def release(self):
        if self.bind_stdout:
            sys.stdout = self.terminal
            self.terminal = None
            self.bind_stdout = False

    def write(self, msg):
        self.log.write(msg)
        if self.terminal:
            self.terminal.write(msg)

    def flush(self):
        self.log.flush()
        if self.terminal:
            self.terminal.flush()


# flatten nested dict and remove non-dict key-value pairs of the input dict
def flatten_deep_dict(deep_dict, res_dict=None, c_key=None):
    # init
    res_dict = res_dict if res_dict else {}

    # iterate dict elements
    for key, value in deep_dict.items():
        # move every dict to res_dict
        if isinstance(value, dict):
            # set hierarchical keys
            n_key = key if not c_key else c_key + '.' + key
            # make a copy
            res_dict[n_key] = value.copy()
            # remove nested dicts from copys
            for k, v in value.items():
                if isinstance(v, dict):
                    _ = res_dict[n_key].pop(k)
            # check empty dict
            if not res_dict[n_key]:
                _ = res_dict.pop(n_key)
            # recursively flatten nested dict
            _ = flatten_deep_dict(value, res_dict, c_key=key)

    # return flat dict
    return res_dict
