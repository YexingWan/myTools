#!/usr/bin/env bash
cd ~/newhome


sudo docker run -d --gpus=all --shm-size='2gb' --name yx-wan --network host --ipc=host --privileged=true -v /home/yexing/workspcae:/workspace -it corerain/cr_docker_env:gpu /bin/bash

docker exec -it yx-wan bash
apt-get update
apt-get install nano

echo 'alias activate="source ~/newhome/venv/bin/activate"' >> ~/.bashrc
source "$HOME/.bashrc"
activate
pip install --upgrade dask -i https://mirrors.aliyun.com/pypi/simple
pip install --upgrade pandas==0.25.3 -i https://mirrors.aliyun.com/pypi/simple
pip install -U PyYAML -i https://mirrors.aliyun.com/pypi/simple
pip install setuptools==42.0.2 -i https://mirrors.aliyun.com/pypi/simple
pip install torch torchvision
pip install opencv-python opencv-contrib-python -i https://mirrors.aliyun.com/pypi/simple
pip install Pillow -i https://mirrors.aliyun.com/pypi/simple


pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
mkdir ~/newhome/RbEnv
cp -r /nas/RainBuilderPackage ~/newhome/RbEnv
cd /root/newhome/RbEnv/v2.2.0-beta

pip install RbRuntime_gpu-2.2.0-cp36-cp36m-linux_x86_64.whl -i https://mirrors.aliyun.com/pypi/simple
pip install RbCompiler-2.2.0-py3-none-any.whl -i https://mirrors.aliyun.com/pypi/simple
pip install plumber_ir-2.2.0-py3-none-any.whl -i https://mirrors.aliyun.com/pypi/simple
apt-get install libboost-all-dev


echo 'export LC_ALL=C.UTF-8' >> ~/.bashrc
echo 'export LANG=C.UTF-8' >> ~/.bashrc

source "$HOME/.bashrc"
activate


python -c "import rbcompiler; import tensorflow as tf ; print('RbComplier version: '+rbcompiler.__version__)"
python -c "import pyRbRuntime; print(pyRbRuntime.__version__)"

echo "Setup done."

virtualenv venv -p python3
