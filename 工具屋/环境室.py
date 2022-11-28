import subprocess as 子进程

import gym
import gym_super_mario_bros  # 虽然没有用到，但你必须导入它，因为它会改变gym库中的某些内容
from gym import Wrapper
from gym.spaces import Box
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace


class 监视器类:
    def __init__(self, 宽度, 高度, 保存的路径):
        # ffmpeg 需要在电脑额外安装这个软件，https://ffmpeg.org/
        self.命令 = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(宽度, 高度),
                   "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", 保存的路径]
        try:
            self.管道 = 子进程.Popen(self.命令, stdin=子进程.PIPE, stderr=子进程.PIPE)
        except FileNotFoundError:
            print("没有找到ffmpeg这个程序")
            exit(-1)


class 自定义奖励类(Wrapper):
    def __init__(self, 环境=None, 监视器=None):
        super(自定义奖励类, self).__init__(环境)
        self.观察的空间 = Box(low=0, high=255, shape=(1, 84, 84))
        self.当前分数 = 0
        if 监视器:
            self.某个监视器 = 监视器
        else:
            self.某个监视器 = None


class 自定义跳过帧(Wrapper):
    def __init__(self, 环境=None, 跳过量=4):
        super(自定义跳过帧, self).__init__(环境)
        self.观察的空间 = Box(low=0, high=255, shape=(4, 84, 84))
        self.跳过量 = 跳过量


def 创建训练环境(世界号, 舞台号, 操作模式, 输出的路径=None):
    环境 = gym.make("SuperMarioBros-{}-{}-v3".format(世界号, 舞台号), apply_api_compatibility=True, render_mode="human")
    if 输出的路径:
        某个监视器 = 监视器类(256, 240, 输出的路径)
    else:
        某个监视器 = None

    if 操作模式 == "仅向右":
        动作列表 = RIGHT_ONLY
    elif 操作模式 == "简单":
        动作列表 = SIMPLE_MOVEMENT
    else:
        动作列表 = COMPLEX_MOVEMENT

    环境 = JoypadSpace(环境, 动作列表)
    环境 = 自定义奖励类(环境, 某个监视器)
    环境 = 自定义跳过帧(环境)

    return 环境, 环境.观察的空间.shape[0], len(动作列表)
