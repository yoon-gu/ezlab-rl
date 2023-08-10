FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN apt-get update
RUN apt install -y git
RUN git clone https://github.com/DLR-RM/stable-baselines3.git
RUN cd stable-baselines3 && pip install '.[extra_no_roms]'
RUN pip install hydra-core scipy seaborn
WORKDIR /workspace
