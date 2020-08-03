# YanShenCup
遥感图像场景分类竞赛

代码主要在<https://github.com/implus/PytorchInsight>上进行修改，修改的内容包括：
1. 自定义DataSet类，实现自定义数据读取
2. 添加半监督微调功能(SSL_fine-tune)
3. 将pytorch数据加载和模型训练同步进行，主要参考的项目为： <https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256>
