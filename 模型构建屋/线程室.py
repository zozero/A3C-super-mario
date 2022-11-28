import os
import timeit

import torch
from tensorboardX import SummaryWriter

from 工具屋.参数室 import 参数类
from 工具屋.环境室 import 创建训练环境
from 模型构建屋.模型室 import 行动者和评论家类


def 执行单个训练(索引, 参数: 参数类, 全局模型: 行动者和评论家类, 优化器, 是否保存=False):
    torch.manual_seed(123 + 索引)
    if 是否保存:
        开始时间 = timeit.default_timer()
    作家 = SummaryWriter(参数.日志的路径)
    环境, 舞台数量, 动作数量 = 创建训练环境(参数.世界号, 参数.舞台号, 参数.操作模式)
    区域模型 = 行动者和评论家类(舞台数量, 动作数量)
    if 参数.是否使用图像处理单元:
        区域模型.cuda()
    区域模型.train()
    状态 = torch.from_numpy(环境.reset())
    if 参数.是否使用图像处理单元:
        状态.cuda()
    完毕 = True
    当前步数 = 0
    当前插曲数 = 0
    while True:
        if 是否保存:
            if 当前插曲数 % 参数.保存的间隔 == 0 and 当前插曲数 > 0:
                临时文件路径字符串 = os.path.join(参数.保存的路径, "超级马里奥_{}_{}".format(参数.世界号, 参数.舞台号))
                torch.save(全局模型.state_dict(), 临时文件路径字符串)
            print("当前线程：{}，插曲：{}".format(索引, 当前插曲数))
