FROM pytorch/pytorch:latest
WORKDIR /src


#------------------------INSTALL ----------------------

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \

RUN apt-get update
RUN apt-get -y install python3.10
RUN pip install tensorboard==2.17.0
RUN pip install opencv-python==4.10.0.84
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN pip install scikit-learn
RUN pip install matplotlib
RUN pip install torchsummary
RUN pip install streamlit  # using for deploy model on website
#-------------------------------------------------------



#------------------------PASTE-FILE ----------------------

COPY dataset.py ./dataset.py
COPY model_lab1.py ./model_lab1.py
COPY scrip.py ./scrip.py
COPY test_vn.py ./test_vn.py
COPY covid-xray-app.py ./covid-xray-app.py
#COPY trained_model/last_cnn.pt ./trained_model/last_cnn.pt
COPY best_cnn.pt ./best_cnn.pt
#-------------------------------------------------------


CMD ["python3", "test_vn.py"]