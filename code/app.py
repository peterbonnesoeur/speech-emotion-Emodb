#coding: utf-8

from flask import Flask, render_template, request
#from run import main as trainer
import subprocess
import os

def main():
    app = Flask(__name__)


    @app.route('/')
    def homepage():
        return render_template("/index.html")

    @app.route('/train_model', endpoint='train_model',methods=['POST', "GET"])    
    def train():

        my_output =['']*6

        if request.method == "POST":

            #Request the parameters from the html form
            render_template("/training.html")
            data_path = request.form['data_path']
            epochs = request.form['epochs']
            lr = request.form['lr']
            batch_size = request.form['batch_size']

            #? Run the following subprocess to train the model
            command = f"python3 run.py --train --data {data_path} --epochs {epochs} --lr {lr} --batch_size {batch_size} "

            my_output = subprocess.check_output(command, shell=True).decode('utf8').strip()        
            my_output = my_output.replace("\r", "   ").split("\n")

        #Print the validation loss and accuracy
        return render_template("/training.html", model = my_output[-5], test_loss=my_output[-2], test_accuracy=my_output[-1])
    

    @app.route('/eval', endpoint='eval', methods=['POST', "GET"])
    def eval():

        my_output =['']*6

        message = ''

        #? check if some models are already available
        if os.path.isdir('models'):
            render_template("/eval.html",  message = "Ready to be tested")
        else:
            #If no model is available, request for training first
            message = "You need to train your model beforehand. Got to http://localhost:5000/train_model and try again later on"
            render_template("/eval.html",  message = "Ready to be tested")


        if request.method == "POST":

            if os.path.isdir('models'):

                #? request the wav file from the html form
                speech_file = os.path.join("../data/wav", request.form['speech'])

                #? Run the following subprocess to predict the emotion for the given file
                command = f"python3 run.py --predict  --file {speech_file}"
                my_output = subprocess.check_output(command, shell=True).decode('utf8').strip()        
                my_output = my_output.replace("\r", "   ").split("\n")
            else: 
                pass
                
        #?Return the prediction and the ground truth
        return render_template("/eval.html", message = message, identity = my_output[-4], sentence=my_output[-3], emotion=my_output[-2], \
                                 prediction=my_output[-1])
    
    app.run(debug=True, host='0.0.0.0')



if __name__ == '__main__':
    main()