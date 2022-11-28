import argparse
import os.path


class 参数类:
    def __init__(self):
        self.世界号 = 1
        self.舞台号 = 1
        self.操作模式 = "复杂"
        self.学习率 = 1e-4
        self.伽马 = 0.9
        self.套 = 1.0
        self.贝塔 = 0.01
        self.区域步数 = 50
        self.全局步数 = 5e6
        self.线程数 = 6
        self.保存的间隔 = 500
        self.最大动作步数 = 200
        临时字符串 = os.path.join("张量板", "超级马里奥")  # 解决不同系统的路径问题
        self.日志的路径 = 临时字符串
        self.保存的路径 = "已训练的模型"
        self.是否载入过往的舞台 = False
        self.是否使用图像处理单元 = True


# 之所以这样是为了在编程时有输入提示，在参数类里面该值没有意义。
参数 = 参数类()
参数解析器 = argparse.ArgumentParser()
参数解析器.add_argument("--世界号", type=int, default=1)
参数解析器.add_argument("--舞台号", type=int, default=1)
参数解析器.add_argument("--操作模式", type=str, default="复杂")
参数解析器.add_argument("--学习率", type=float, default=1e-4, help="1e-4等于0.0001")
参数解析器.add_argument("--伽马", type=float, default=0.9, help="这里表示奖励的折扣因子")
参数解析器.add_argument("--套", type=float, default=1.0, help="用广义优势估计函数的参数，希腊字母套")
参数解析器.add_argument("--贝塔", type=float, default=0.01, help="熵的系数")
参数解析器.add_argument("--区域步数", type=int, default=50)
参数解析器.add_argument("--全局步数", type=int, default=5e6, help="即5000000，10的6次方")
参数解析器.add_argument("--线程数", type=int, default=6)
参数解析器.add_argument("--保存的间隔", type=int, default=500,
                   help="是当前插曲的间隔一个插曲等于5000000/50=100000次循环，这里即为每循环500次执行了500*50步就保存一次。插曲是个代称，和伽马、贝塔类似")
参数解析器.add_argument("--最大动作步数", type=int, default=200, help="测试阶段最大重复步数")
临时字符串 = os.path.join("张量板", "超级马里奥")  # 解决不同系统的路径问题
参数解析器.add_argument("--日志的路径", type=str, default=临时字符串)
参数解析器.add_argument("--保存的路径", type=str, default="已训练的模型")
参数解析器.add_argument("--是否载入过往的舞台", type=bool, default=False, help="载入已训练的模型")
参数解析器.add_argument("--是否使用图像处理单元", type=bool, default=True)
参数 = 参数解析器.parse_args()
