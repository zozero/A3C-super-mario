from torch import nn


class 行动者和评论家类(nn.Module):
    def __init__(self, 输入的数量, 动作的数量):
        super(行动者和评论家类, self).__init__()
        self.卷积层1 = nn.Conv2d(输入的数量, 32, 3, stride=2, padding=1)
        self.卷积层2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.卷积层3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.卷积层4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.长短期记忆层 = nn.LSTMCell(32 * 6 * 6, 512)
        self.评论家_线性层 = nn.Linear(512, 1)
        self.行动者_线性层 = nn.Linear(512, 动作的数量)
        self._初始化权重()

    def _初始化权重(self):
        for 模型 in self.modules():
            if isinstance(模型, nn.Conv2d) or isinstance(模型, nn.Linear):
                nn.init.xavier_uniform_(模型.weight)
                nn.init.constant_(模型.bias, 0)
            elif isinstance(模型, nn.LSTMCell):
                nn.init.constant_(模型.bias_ih, 0)
                nn.init.constant_(模型.bias_hh, 0)

    def forward(self,):
        pass