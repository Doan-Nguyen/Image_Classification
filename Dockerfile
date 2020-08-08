FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    vim \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /workspace
WORKDIR /workspace

# # Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /workspace
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
 && /home/user/miniconda/bin/conda clean -ya

# CUDA 10.1-specific steps
RUN conda install -y -c pytorch torchvision cudatoolkit=10.1 pytorch \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install Requests, a Python library for making HTTP requests
RUN conda install -y requests=2.19.1 \
 && conda clean -ya

# Install Graphviz
# RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
#  && conda clean -ya

# Install OpenCV3 Python bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   libgtk2.0-0 \
   libcanberra-gtk-module \
&& sudo rm -rf /var/lib/apt/lists/*
RUN pip install opencv-contrib-python

RUN pip install flask flask_cors

RUN pip install boto3

RUN pip install imgaug
# Set the default command to python3
# CMD ["python3"]

WORKDIR /home/user/
# RUN sudo chmod -R a+X .
EXPOSE 8889 8889
RUN git clone http://Nguyen_Van_Thanh:53ki32ng30c1q@git.deha.vn:1080/Nguyen_Van_Thanh/SignatureAPI.git
WORKDIR /home/user/SignatureAPI/Stamp
# RUN sudo chmod -R a+X .

RUN curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
RUN sudo apt-get install -y nodejs
RUN sudo npm install forever -g
RUN forever start -c python stamp_api.py

CMD ["python","stamp_api.py"]
# ENTRYPOINT ["python" ,"stamp_api.py"]
# RUN ./start.sh 
