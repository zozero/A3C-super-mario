import os.path

import torch
import torch.nn.functional as 火炬函数

from 工具屋.参数室 import 参数
from 工具屋.环境室 import 创建训练环境
from 模型构建屋.模型室 import 行动者和评论家类


def 测试():
    torch.cuda.manual_seed(123)
    # 临时路径拼接 = os.path.join(参数.视频输出的路径, "sp_{}_{}.mp4".format(参数.世界号, 参数.舞台号))
    # 环境, 舞台数量, 动作数量 = 创建训练环境(参数.世界号, 参数.舞台号, 参数.操作模式, 临时路径拼接)
    环境, 舞台数量, 动作数量 = 创建训练环境(参数.世界号, 参数.舞台号, 参数.操作模式, 渲染模式="human")
    模型 = 行动者和评论家类(舞台数量, 动作数量)

    临时路径拼接 = os.path.join(参数.保存的路径, "超级马里奥_{}_{}_已完成".format(参数.世界号, 参数.舞台号))
    if 参数.是否使用图像处理单元:
        模型.load_state_dict(torch.load(临时路径拼接))
        模型.cuda()
    else:
        模型.load_state_dict(torch.load(临时路径拼接))
    模型.eval()
    状态 = torch.from_numpy(环境.reset())
    完毕 = True
    while True:
        if 完毕:
            隐藏的状态_零张量 = torch.zeros((1, 512), dtype=torch.float)
            单元的状态_零张量 = torch.zeros((1, 512), dtype=torch.float)
            环境.reset()
        else:
            隐藏的状态_零张量 = 隐藏的状态_零张量.detach()
            单元的状态_零张量 = 单元的状态_零张量.detach()

        if 参数.是否使用图像处理单元:
            状态 = 状态.cuda()
            隐藏的状态_零张量 = 隐藏的状态_零张量.cuda()
            单元的状态_零张量 = 单元的状态_零张量.cuda()

        动作_策略, 预期值, 隐藏的状态_零张量, 单元的状态_零张量 = 模型(状态, 隐藏的状态_零张量, 单元的状态_零张量)
        策略 = 火炬函数.softmax(动作_策略, dim=1)
        # for i in 策略[0]:
        #     print(round(i.item(),4))
        动作 = torch.argmax(策略).item()
        动作 = int(动作)
        状态, 奖励, 完毕, _, 信息 = 环境.step(动作)
        状态 = torch.from_numpy(状态)
        环境.render()
        if 信息["flag_get"]:
            print("世界 {}，舞台 {}，执行完毕".format(参数.世界号, 参数.舞台号))
            break


if __name__ == '__main__':
    测试()
    pass
