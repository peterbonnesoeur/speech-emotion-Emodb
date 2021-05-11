#!/usr/bin/env python3
import argparse
import os
import pickle

import torch
import torch.nn as nn

import numpy as np
import copy
import matplotlib.pyplot as plt

from train_prep.preprocessing import preprocessing_data, preprocessing_unit, emotion_id, id_emotion, file_info, emotion_code
from train_prep.architecture import TimeDistributed, HybridModel, loss_fnc, make_train_step, make_validate_fnc

TRAIN_VAL_RATIO = 0.75

def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ratio', type=float, help='Define the train val ratio for the data preparation', default = TRAIN_VAL_RATIO)
    parser.add_argument('--epochs', type=int, help='Number of epochs for the training', default=200)
    parser.add_argument('--lr', type=float, help = 'Set the learning rate for the training step', default=0.01)
    parser.add_argument('--batch_size', type=int, help = 'Set the batch size for the training step', default = 32)

    parser.add_argument('--train', help = 'Begin the training for the model', action='store_true')
    parser.add_argument('--prep', help = 'Begin the dataset preparation', action='store_true')
    parser.add_argument('--predict', help = 'Predict the given wav file emotion',action='store_true')

    parser.add_argument('--file', help="File to predict", default = "../data/wav/03a01Wa.wav")
    parser.add_argument('--data', help="Path to the data", default = "../data/wav")

    args = parser.parse_args()

    return args

def main(cli_input = True, args = None):

    if cli_input :
        args = cli()


    if args.predict and not os.path.isdir("models"):
        args.train == True

    if args.prep or (args.train and not os.path.isdir("arrays")):
        print(args.data)
        X_train, Y_train, X_val, Y_val = preprocessing_data(args.data, train_val_ratio = args.ratio)
        SAVE_PATH = os.path.join(os.getcwd(),'arrays')
        os.makedirs('arrays',exist_ok=True)

        with open(r"arrays/data.pickle", "wb") as output_file:
            data = (X_train, Y_train, X_val, Y_val)
            pickle.dump(data, output_file)

    if args.train:
        with open(r"arrays/data.pickle", "rb") as input_file:
           (X_train, Y_train, X_val, Y_val)  = pickle.load(input_file)

        train(args, X_train, Y_train, X_val, Y_val)
        evaluation(X_val, Y_val)

    if args.predict:
        if os.path.isdir("models"):
            predict(args)
        else:
            print("You should train your model first. Try running first and foremost: \npython3 -m --train")

        

def predict(args):

    print("preprocess the input")
    X = preprocessing_unit(args.file)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.tensor(X,device=device).float()

    print("Load model")
    LOAD_PATH = os.path.join(os.getcwd(),'models')
    model = HybridModel(len(emotion_id)).to(device)
    model.load_state_dict(torch.load(os.path.join(LOAD_PATH,'model.pt')))
    output_logits, output_softmax, attention_weights_norm = model(X)
    predictions = torch.argmax(output_softmax,dim=1)

    file_info(args.file)
    print(f"The predicted emotion is: {emotion_code[id_emotion[int(predictions[0])]]}")


def train(args,X_train, Y_train, X_val, Y_val ):
    EPOCHS=args.epochs
    DATASET_SIZE = X_train.shape[0]
    BATCH_SIZE = args.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Selected device is {}'.format(device))
    model = HybridModel(num_emotions=len(emotion_id)).to(device)
    print('Number of trainable params: ',sum(p.numel() for p in model.parameters()) )
    OPTIMIZER = torch.optim.SGD(model.parameters(),lr=args.lr, weight_decay=1e-3, momentum=0.8)

    train_step = make_train_step(model, loss_fnc, optimizer=OPTIMIZER)
    validate = make_validate_fnc(model,loss_fnc)
    losses=[]
    val_losses = []
    best_model = model
    best_val = 0

    for epoch in range(EPOCHS):
        # schuffle data
        ind = np.random.permutation(DATASET_SIZE)
        X_train = X_train[ind,:,:,:,:]
        Y_train = Y_train[ind]
        epoch_acc = 0
        epoch_loss = 0
        iters = int(DATASET_SIZE / BATCH_SIZE)
        for i in range(iters):
            batch_start = i * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, DATASET_SIZE)
            actual_batch_size = batch_end-batch_start
            X = X_train[batch_start:batch_end,:,:,:,:]
            Y = Y_train[batch_start:batch_end]
            X_tensor = torch.tensor(X,device=device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long,device=device)
            loss, acc = train_step(X_tensor,Y_tensor)
            epoch_acc += acc*actual_batch_size/DATASET_SIZE
            epoch_loss += loss*actual_batch_size/DATASET_SIZE
            print(f"\r Epoch {epoch}: iteration {i}/{iters}",end='')
        X_val_tensor = torch.tensor(X_val,device=device).float()
        Y_val_tensor = torch.tensor(Y_val,dtype=torch.long,device=device)
        val_loss, val_acc, _ = validate(X_val_tensor,Y_val_tensor)
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        if val_acc > best_val:
            best_val = val_acc
            best_model = copy.deepcopy(model)
        print('')
        print(f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%")

        SAVE_PATH = os.path.join(os.getcwd(),'models')

    
    os.makedirs('models',exist_ok=True)
    torch.save(best_model.state_dict(),os.path.join(SAVE_PATH,'model.pt'))
    print('Model is saved to {}'.format(os.path.join(SAVE_PATH,'model.pt')))


def evaluation(X_val, Y_val):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    print("Evaluation step")
    LOAD_PATH = os.path.join(os.getcwd(),'models')
    model = HybridModel(len(emotion_id)).to(device)
    model.load_state_dict(torch.load(os.path.join(LOAD_PATH,'model.pt')))
    print('Model is loaded from {}'.format(os.path.join(LOAD_PATH,'model.pt')))

    validate = make_validate_fnc(model,loss_fnc)


    X_test_tensor = torch.tensor(X_val,device=device).float()
    Y_test_tensor = torch.tensor(Y_val,dtype=torch.long,device=device)
    test_loss, test_acc, predictions = validate(X_test_tensor,Y_test_tensor)
    print(f'Test loss is {test_loss:.3f}')
    print(f'Test accuracy is {test_acc:.2f}%')

    #plt.plot(losses,'b')
    #plt.plot(val_losses,'r')
    #plt.legend(['train loss','val loss'])


if __name__ == '__main__':
    main(cli_input = True)
