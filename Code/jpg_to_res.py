import mxnet as mx
import os

MXNET_HOME="/home/eraser/mxnet"

os.system('python3 %s/tools/im2rec.py /home/eraser/train /home/eraser/yoga_data/train --recursive --list --num-thread 8'%MXNET_HOME)
os.system('python3 %s/tools/im2rec.py /home/eraser/validation /home/eraser/yoga_data/validation --recursive --list --num-thread 8'%MXNET_HOME)

os.system('python3 %s/tools/im2rec.py /home/eraser/train /home/eraser/yoga_data/train --recursive --pass-through --pack-label --num-thread 8'%MXNET_HOME)
os.system('python3 %s/tools/im2rec.py /home/eraser/validation /home/eraser/yoga_data/validation --recursive --pass-through --pack-label --num-thread 8'%MXNET_HOME)