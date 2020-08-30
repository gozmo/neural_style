#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.7
RUN apt-get install -y python3-pip
RUN apt-get install -y libcuda1-440

RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install numpy
RUN python3.7 -m pip install "scipy==1.1"
RUN python3.7 -m pip install "tensorflow-gpu<2.0,>=1.0"
RUN python3.7 -m pip install numpy
RUN python3.7 -m pip install pandas
RUN python3.7 -m pip install matplotlib
RUN python3.7 -m pip install sklearn
RUN python3.7 -m pip install pudb
RUN python3.7 -m pip install streamlit
RUN python3.7 -m pip install Pillow


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
RUN mkdir /usr/local/nvidia
RUN ln -s /usr/local/cuda/lib64 /usr/local/nvidia/lib64

RUN mkdir -p /root/.streamlit

RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'


WORKDIR /neural-style
COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]

