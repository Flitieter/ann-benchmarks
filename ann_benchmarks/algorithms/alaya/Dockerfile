FROM ann-benchmarks

# COPY ./ann_benchmarks/algorithms/alaya/alaya.cpython-38-x86_64-linux-gnu.so /home/app/
# COPY ./ann_benchmarks/algorithms/alaya/setup.py /home/app/
# RUN python3 /home/app/setup.py install

# CMD ls /home/app && sleep 100

# COPY ./alaya.cpython-38-x86_64-linux-gnu.so /home/app/
# COPY ./setup.py /home/app/

# RUN bash
# RUN python3 -c 'import alaya'

# FROM docker.io/library/ubuntu:24.04

# RUN apt-get update

# RUN apt-get install -y python3-numpy python3-scipy python3-pip build-essential git 
# RUN pip3 install -U pip                                                            
# RUN python3 --version | grep 'Python 3.10'                                         
# WORKDIR /home/app                                                                  
# COPY requirements.txt run_algorithm.py ./                                          
# RUN pip3 install -r requirements.txt    
# RUN apt update
# RUN apt install -y software-properties-common
# RUN add-apt-repository ppa:ubuntu-toolchain-r/test
# RUN apt-get update
# RUN apt-get install -y gcc-13 g++13 --fix-missing

COPY ./ann_benchmarks/algorithms/alaya/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/

