这是用于fork cosyvoice同步的分支，pro分支添加了生成字幕文件，新增音色和保存当前音色功能

环境使用:

```txt
conda 24.11.3
python 3.11.11 
NVIDIA-SMI 545.84 
Driver Version: 545.84       
CUDA Version: 12.3
```

安装说明:

```shell
git clone --recursive https://gitee.com/lulendi/CosyVoice.git
# 进入项目目录
cd CosyVoice
# 进入third_party目录
cd third_party
# 拉取Matcha-TTS
git clone https://github.com/shivammehta25/Matcha-TTS.git
# 回到项目目录
cd ..
# 创建虚拟环境 建议用pycharm 右下角 No Interpreter ->  Add new Interpreter 添加conda环境
conda create -n cosyvoice -y python=3.11 
# 激活虚拟环境
conda activate cosyvoice
# 安装pynini
conda install -y -c conda-forge pynini==2.1.5
# 安装依赖
# 特别提醒 安装过程中，依赖下载
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# 下载模型
python downloadModel.py
# 复制 spk2info.pt 放入pretrained_models/CosyVoice2-0.5B目录下


```

GPU诊断报告:

```shell
python gpu_diagnostics.py
```
待完成:
    
```shell
api接口整合
```
生成requirements.txt
```shell
pipreqs ./ --encoding=utf8  --force
```