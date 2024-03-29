from torch import nn
from sdlf.ops.common import get_class


class Net(nn.Module):
    """
    Network

    """

    def __init__(self,
                 model_cfg_list):
        super().__init__()
        self.name = 'Net'

        # initialize modules
        self.module_list = []
        for model_cfg in model_cfg_list:
            mod = get_class(model_cfg['class'])(model_cfg['args'])
            self.add_module(model_cfg['name'], mod)
            self.module_list.append(model_cfg['name'])

    def network_forward(self, example):
        """
        forward function for each sub module

        :param example: input to the network
        :return: feed forward result dict
        """
        ret_dict = {}
        for mod_name in self.module_list:
            mod_ret_dict = getattr(self, mod_name)(example, ret_dict)
            ret_dict.update(mod_ret_dict)
        return ret_dict

    def forward(self, example):
        """
        forward function for the whole network

        :param example: input to the network
        :return: dict, at least including 'loss' in training mode, or 'preds' in eval mode
        """
        ff_ret_dict = self.network_forward(example)
        if self.training:
            return self.loss(example, ff_ret_dict)
        else:
            return self.predict(example, ff_ret_dict)

    def loss(self, example, ff_ret_dict):
        """
        call the loss function of each sub module, if exists

        :param example: input to the network
        :param ff_ret_dict: dict, containing the feed forward results of all sub modules
        :return: loss dict (at least includes 'loss'), and loss info dict
        """
        loss_ret = {}
        loss_info_ret = {}

        loss = 0.0
        for mod_name in self.module_list:
            mod = getattr(self, mod_name)
            if hasattr(mod, 'loss'):
                mod_loss, loss_info = mod.loss(example, ff_ret_dict)
                loss += mod_loss
                loss_ret[mod_name] = mod_loss
                if loss_info:
                    loss_info_ret.update(loss_info)
        loss_ret['loss'] = loss

        return loss_ret, loss_info_ret, ff_ret_dict

    def predict(self, example, ff_ret_dict):
        """
        call the predict function of each sub module, if exists

        :param example: input to the network
        :param ff_ret_dict: dict, containing the feed forward results of all sub modules
        :return: dict, including the predictions of each sub module, if exist
        """
        label_ret, pred_ret = {}, {}
        for mod_name in self.module_list:
            mod = getattr(self, mod_name)
            if hasattr(mod, 'predict'):
                mod_label, mod_pred = mod.predict(example, ff_ret_dict)
                label_ret.update(mod_label)
                pred_ret.update(mod_pred)
        return label_ret, pred_ret, ff_ret_dict
