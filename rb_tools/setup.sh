#!/usr/bin/env bash
docker run -d --name=yx-wan -v /mnt/newhome/yexing/:/mnt/newhome/yexing -v /home/yx-wan:/home/yx-wan -v /home/yx-wan/:/root -v /nas-software:/nas -i -t corerain/cr-working-env:gpu /bin/bash
cd ~/newhome
virtualenv venv -p python3

echo 'alias activate="source ~/newhome/venv/bin/activate"' >> ~/.bashrc

source "$HOME/.bashrc"
activate
pip install --upgrade dask -i https://mirrors.aliyun.com/pypi/simple
pip install --upgrade pandas==0.22.0 -i https://mirrors.aliyun.com/pypi/simple
# cp RbCompiler whl to current dir
pip install -U PyYAML -i https://mirrors.aliyun.com/pypi/simple
pip install tensorflow-gpu==1.12.0 -i https://mirrors.aliyun.com/pypi/simple
pip install setuptools==36.1.0 -i https://mirrors.aliyun.com/pypi/simple
pip3 install torch==0.4.1 -i https://mirrors.aliyun.com/pypi/simple
pip3 install torchvision==0.4.1 -i https://mirrors.aliyun.com/pypi/simple
pip install opencv-python opencv-contrib-python -i https://mirrors.aliyun.com/pypi/simple
pip3 install Pillow -i https://mirrors.aliyun.com/pypi/simple

pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
mkdir ~/newhome/RbEnv
cp -r /nas/RainBuilderPackage ~/newhome/RbEnv
cd /root/newhome/RbEnv/RainBuilderPackage/v2.2.0-rc1

pip install RbRuntime_gpu-2.2.0-cp35-cp35m-linux_x86_64.whl -i https://mirrors.aliyun.com/pypi/simple
pip install RbCompiler-2.2.0_encrypt-cp35-cp35m-linux_x86_64.whl -i https://mirrors.aliyun.com/pypi/simple
pip install plumber_ir-2.2.0-py3-none-any.whl -i https://mirrors.aliyun.com/pypi/simple

echo 'export LC_ALL=C.UTF-8' >> ~/.bashrc
echo 'export LANG=C.UTF-8' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/root/newhome/venv/lib/python3.5/site-packages:$LD_LIBRARY_PATH"' >> ~/.bashrc

source "$HOME/.bashrc"
activate

apt-get install nano



python -c "import rbcompiler; import tensorflow as tf ; print('RbComplier version: '+rbcompiler.__version__)"
python -c "import pyRbRuntime; print(pyRbRuntime.__version__())"

echo "Setup done."
