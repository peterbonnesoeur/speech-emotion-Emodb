FROM nvcr.io/nvidia/pytorch:21.04-py3

RUN apt-get update && apt-get -y update

#RUN apt-get install -y build-essential python3.6 python3-pip python3-dev 
#RUN pip3 -q install pip --upgrade

RUN apt-get install unzip wget




ADD code/ /code
WORKDIR /code


#Download the dataset
RUN wget http://emodb.bilderbar.info/download/download.zip
RUN unzip -q  download.zip  -d ../data
RUN rm download.zip


#Run updates
RUN apt-get update
RUN pip3 install -r requirements.txt

RUN pip3 install matplotlib flask Flask librosa jupyterlab jupyter_server==1.6.4 seaborn scikit-learn einops

#Nvidia driver troubleshoot
RUN rm -rf .nv/


# expose jupyter-server
EXPOSE 8888

# expose Flask app
EXPOSE 5000

CMD python3 app.py & jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=visium


