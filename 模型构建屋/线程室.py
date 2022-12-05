import os
import timeit

import numpy as np
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as 火炬函数
from torch.distributions import Categorical

from 工具屋.参数室 import 参数类
from 工具屋.环境室 import 创建训练环境
from 模型构建屋.模型室 import 行动者和评论家类


def 执行单个训练(索引, 参数: 参数类, 全局模型: 行动者和评论家类, 优化器, 是否保存=False):
    torch.cuda.manual_seed(123 + 索引)
    if 是否保存:
        开始时间 = timeit.default_timer()
    作家 = SummaryWriter(参数.日志的路径)
    环境, 状态数量, 动作数量 = 创建训练环境(参数.世界号, 参数.舞台号, 参数.操作模式, 渲染模式="human")
    区域模型 = 行动者和评论家类(状态数量, 动作数量)
    if 参数.是否使用图像处理单元:
        区域模型.cuda()
    区域模型.train()
    状态 = torch.from_numpy(环境.reset())
    if 参数.是否使用图像处理单元:
        状态 = 状态.cuda()
    完毕 = True
    当前步数 = 0
    当前插曲数 = 0
    总损失值列表 = np.array([])
    while True:
        当前插曲数 += 1
        区域模型.load_state_dict(全局模型.state_dict())
        if 完毕:
            隐藏的状态 = torch.zeros((1, 512), dtype=torch.float)
            细胞的状态 = torch.zeros((1, 512), dtype=torch.float)
        else:
            隐藏的状态 = 隐藏的状态.detach()
            细胞的状态 = 细胞的状态.detach()
        if 参数.是否使用图像处理单元:
            隐藏的状态 = 隐藏的状态.cuda()
            细胞的状态 = 细胞的状态.cuda()

        动作策略对数列表 = []
        预期值列表 = []
        奖励列表 = []
        熵值列表 = []

        for _ in range(参数.区域步数):
            当前步数 += 1
            动作_策略, 预期值, 隐藏的状态, 细胞的状态 = 区域模型(状态, 隐藏的状态, 细胞的状态)
            动作_策略_概率 = 火炬函数.softmax(动作_策略, dim=1)
            动作_策略_对数 = 火炬函数.log_softmax(动作_策略, dim=1)
            熵值 = -(动作_策略_概率 * 动作_策略_对数).sum(1, keepdim=True)

            动作_策略_分布 = Categorical(动作_策略_概率)
            动作 = 动作_策略_分布.sample().item()

            状态, 奖励, 完毕, _, _ = 环境.step(动作)
            状态 = torch.from_numpy(状态)
            if 参数.是否使用图像处理单元:
                状态 = 状态.cuda()
            if 当前步数 > 参数.全局步数:
                完毕 = True

            if 完毕:
                当前步数 = 0
                状态 = torch.from_numpy(环境.reset())
                if 参数.是否使用图像处理单元:
                    状态 = 状态.cuda()

            预期值列表.append(预期值)
            动作策略对数列表.append(动作_策略_对数[0, 动作])
            奖励列表.append(奖励)
            熵值列表.append(熵值)

            if 完毕:
                break

        预期_奖励 = torch.zeros((1, 1), dtype=torch.float)
        if 参数.是否使用图像处理单元:
            预期_奖励 = 预期_奖励.cuda()

        if not 完毕:
            _, 预期_奖励, _, _ = 区域模型(状态, 隐藏的状态, 细胞的状态)

        广义优势估计 = torch.zeros((1, 1), dtype=torch.float)
        if 参数.是否使用图像处理单元:
            广义优势估计 = 广义优势估计.cuda()

        行动者损失值 = 0
        评论家损失值 = 0
        熵值损失值 = 0
        下一个_预期值 = 预期_奖励

        for 预期值, 动作_策略_对数, 奖励, 熵值 in list(zip(预期值列表, 动作策略对数列表, 奖励列表, 熵值列表))[::-1]:
            广义优势估计 = 广义优势估计 * 参数.伽马 * 参数.套
            广义优势估计 = 广义优势估计 + 奖励 + 参数.伽马 * 下一个_预期值.detach() - 预期值.detach()
            下一个_预期值 = 预期值
            行动者损失值 = 行动者损失值 + 动作_策略_对数 * 广义优势估计
            预期_奖励 = 预期_奖励 * 参数.伽马 + 奖励
            评论家损失值 = 评论家损失值 + (预期_奖励 - 预期值) ** 2 / 2
            熵值损失值 = 熵值损失值 + 熵值

        总损失值 = -行动者损失值 + 评论家损失值 - 参数.贝塔 * 熵值损失值
        作家.add_scalar("训练_{}/损失值".format(索引), 总损失值, 当前插曲数)
        优化器.zero_grad()
        # if 总损失值列表.mean() < 总损失值.item():
        #     for i in range(10):
        #         总损失值.backward(retain_graph=True)
        总损失值.backward()
        if 是否保存:
            if 当前插曲数 % 参数.保存的间隔 == 0 and 当前插曲数 > 0:
                临时文件路径字符串 = os.path.join(参数.保存的路径, "超级马里奥_{}_{}".format(参数.世界号, 参数.舞台号))
                torch.save(全局模型.state_dict(), 临时文件路径字符串)

        总损失值列表 = np.append(总损失值列表, [总损失值.item()])
        print("当前线程：{}，插曲：{}，总损失值：{}，当前平均值：{}".format(索引, 当前插曲数, round(总损失值.item()), round(总损失值列表.mean(), 2)))
        if len(总损失值列表) > 80:
            总损失值列表 = np.array([])

        for 区域模型参数, 全局模型参数 in zip(区域模型.parameters(), 全局模型.parameters()):
            if 全局模型参数.grad is not None:
                break
            全局模型参数._grad = 区域模型参数.grad

        优化器.step()

        if 当前插曲数 == int(参数.全局步数 / 参数.区域步数):
            print("训练线程 {} ，已终止。".format(索引))
            if 是否保存:
                结束时间 = timeit.default_timer()
                print("代码运行了%.2f小时。" % ((结束时间 - 开始时间)/3600))
            return
