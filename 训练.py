import os.path
import shutil

import torch
import torch.multiprocessing as 火炬_多线程

from 工具屋.参数室 import 参数
from 工具屋.环境室 import 创建训练环境
from 模型构建屋.优化器室 import 全局自适估计矩
from 模型构建屋.模型室 import 行动者和评论家类
from 模型构建屋.线程室 import 执行单个训练


def 训练():
    torch.manual_seed(123)
    if os.path.isdir(参数.日志的路径):
        shutil.rmtree(参数.日志的路径)
    os.makedirs(参数.日志的路径)
    if not os.path.isdir(参数.保存的路径):
        os.makedirs(参数.保存的路径)
    多线程 = 火炬_多线程.get_context("spawn")
    环境, 舞台数量, 动作数量 = 创建训练环境(参数.世界号, 参数.舞台号, 参数.操作模式)
    全局模型 = 行动者和评论家类(舞台数量, 动作数量)
    if 参数.是否使用图像处理单元:
        全局模型.cuda()
    全局模型.share_memory()
    if 参数.是否载入过往的舞台:
        if 参数.舞台号 == 1:
            过去的世界号 = 参数.世界号 - 1
            过去的舞台号 = 4
        else:
            过去的世界号 = 参数.世界号
            过去的舞台号 = 参数.舞台号 - 1
        临时文件路径字符串 = os.path.join(参数.保存的路径, "超级马里奥_{}_{}".format(过去的世界号, 过去的舞台号))
        if os.path.isfile(临时文件路径字符串):
            全局模型.load_state_dict(torch.load(临时文件路径字符串))

    某个优化器=全局自适估计矩(全局模型.parameters(),学习率=参数.学习率)

    执行单个训练(0,参数,全局模型,某个优化器,True)


if __name__ == '__main__':
    训练()
    pass
