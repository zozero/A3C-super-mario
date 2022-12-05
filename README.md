# 超级马里奥
### 异步优势行动者与评论家
## 关于程序
很有趣的项目，但训练需要很长时间，训练花了10个小时左右。\
没有训练完善的网络模型测试时会卡在某个柱子前，或者某个台阶上，经过很长很长时间后它才学会跳过去......\
命名都为中文，尽可能地贴近其所描述的含义。\
中英文对照：异步优势行动者与评论家(A3C,Asynchronous Advantage Actor-critic)。\
是我仿照自《强化学习实战系列（2020最新）》唐老师的视频课程。
## 文件说明
《训练.py》这里是用来训练的。\
《测试.py》这里是用来训练的。\
《超级马里奥_1_1_已完成》这是我训练好的模型，它在《已训练的模型》目录下。\
《强化学习之异步优势行动者与评论家.docx》这个文件里面是基础的数学公式，我有对公式的组成进行说明。你可能不会理解我写了什么，事实上我重读的时候也是这样：）。
## 注意事项
你需要安装[Gym](https://www.gymlibrary.dev/)，[gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)（专用于马里奥游戏的库）。\
如果在win11系统下保存出现乱码或者报错，请参考[链接](https://blog.csdn.net/wry15082983136/article/details/126229608)的方法。\
使用tensorboardX需要安装tensorflow，在控制台执行命令python -m tensorboard.main --logdir=./张量板/超级马里奥。
## 其他
一个用于使用中文编程，很好用的pycharm插件[链接](https://github.com/tuchg/ChinesePinyin-CodeCompletionHelper)。

