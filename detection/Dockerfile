FROM  nvcr.io/nvidia/pytorch:23.08-py3
ENV DEBIAN_FRONTEND noninteractive
RUN pip3 install --upgrade pip
RUN pip3 install ultralytics
# Uninstall opencv-python and then install the apt-get package. The pip package breaks video support.
RUN pip3 uninstall -y opencv-python
RUN apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg python3-opencv

# Add ldconfig command to the root user's .bashrc file
RUN echo "ldconfig > /dev/null 2>&1" >> /root/.bashrc
