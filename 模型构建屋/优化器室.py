import torch.optim


class 全局自适估计矩(torch.optim.Adam):
    def __init__(self, 参数, 学习率):
        super(全局自适估计矩, self).__init__(参数, 学习率)
        for 组 in self.param_groups:
            for 参 in 组['params']:
                状态 = self.state[参]
                状态['step'] = torch.tensor(0)
                状态['exp_avg'] = torch.zeros_like(参.data)
                状态['exp_avg_sq'] = torch.zeros_like(参.data)

                状态['exp_avg'].share_memory_()
                状态['exp_avg_sq'].share_memory_()
