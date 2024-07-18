FROM dl.nvidia/cuda:11.1-cudnn8-devel-torch1.10.0

COPY xtdet /home/xtdet/xtdet
copy util /home/xtdet/util
COPY whl /home/xtdet/whl
COPY tool/train.py /home/xtdet


RUN pip3 install /home/xtdet/whl/engine_frame-*.whl pycocotools==2.0.4

WORKDIR /home/xtdet
ENTRYPOINT ["python3", "train.py"]