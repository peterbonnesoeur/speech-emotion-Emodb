import os
import librosa
import librosa.display
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



SAMPLING_RATE = 16000
PADDING = 150000

deutsch_sprechen = False
language = 0 if deutsch_sprechen else 1

speaker_code = {
    "03" : "male, 31 years old",
    "08" : "female, 34 years",
    "09" : "female, 21 years",
    "10" : "male, 32 years",
    "11" : "male, 26 years",
    "12" : "male, 30 years",
    "13" : "female, 32 years",
    "14" : "female, 35 years",
    "15" : "male, 25 years",
    "16" : "female, 31 years"
}

sentence_code = {
    "a01" : ("Der Lappen liegt auf dem Eisschrank.", "The tablecloth is lying on the frigde.")[language],
    "a02" : ("Das will sie am Mittwoch abgeben.", "She will hand it in on Wednesday.")[language],
    "a04" : ("Heute abend könnte ich es ihm sagen.", "Tonight I could tell him.")[language],
    "a05" : ("Das schwarze Stück Papier befindet sich da oben neben dem Holzstück.", "The black sheet of paper is located up there besides the piece of timber.")[language],
    "a07" : ("In sieben Stunden wird es soweit sein.", "In seven hours it will be.")[language],
    "b01" : ("Was sind denn das für Tüten, die da unter dem Tisch stehen?", "What about the bags standing there under the table?"[language])[language],
    "b02" : ("Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter.", "They just carried it upstairs and now they are going down again.")[language],
    "b03" : ("An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.", "Currently at the weekends I always went home and saw Agnes.")[language],
    "b09" : ("Ich will das eben wegbringen und dann mit Karl was trinken gehen.", "I will just discard this and then go for a drink with Karl.")[language],
    "b10" : ("Die wird auf dem Platz sein, wo wir sie immer hinlegen.", "It will be in the place where we always store it.")[language]
}

emotion_code = {
    "W" : ("Ärger (Wut)", "Anger")[language],
    "L" : ("Langeweile", "boredom")[language],
    "E" : ("Ekel", "disgust")[language],
    "A" : ("Angst", "anxiety/fear")[language],
    "F" : ("Freude", "happiness")[language],
    "T" : ("Trauer", "sadness")[language],
    "N" : ("Neutral", "Neutral")[language]
}

emotion_id = {
    "W" : 0,
    "L" : 1,
    "E" : 2,
    "A" : 3,
    "F" : 4,
    "T" : 5,
    "N" : 6
}

id_emotion = {
    0 : "W",
    1 : "L",
    2 : "E",
    3 : "A",
    4 : "F",
    5 : "T",
    6 : "N"
}

def file_info(file_path):
    
    file_name = file_path.split("/")[-1].split(".")[0]
    
    speaker = speaker_code.get(file_name[:2], "unknown")
    sentence = sentence_code.get(file_name[2:5], ("text nicht erkannt","text not recognized"))
    emotion = emotion_code.get(file_name[5], ("emotion nicht erkannt","emotion not recognized"))
    print(f"The file {file_name}.wav is spoken by a {speaker}. \nThe sentence is: {sentence}.  \nThe emotion played is: {emotion}")


def get_emotion(file_path):
    #? Get the letter for the emotion and convert it 
    # into his unique number
    
    file_name = file_path.split("/")[-1].split(".")[0]
    return emotion_id.get(file_name[5])

def addAWGN(signal, num_bits=16, augmented_num=2, snr_low=15, snr_high=30): 
    #? Add gaussian white noise
    signal_len = len(signal)
    # Generate White Gaussian noise
    noise = np.random.normal(size=(augmented_num, signal_len))
    # Normalize signal and noise
    norm_constant = 2.0**(num_bits-1)
    signal_norm = signal / norm_constant
    noise_norm = noise / norm_constant
    # Compute signal and noise power
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    # Random SNR: Uniform [15, 30] in dB
    target_snr = np.random.randint(snr_low, snr_high)
    # Compute K (covariance matrix) for each noise 
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, augmented_num)) * K  
    # Generate noisy signal
    return signal + K.T * noise


def getFFT(audio, sample_rate):
    fft = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              window='hamming',
                                              hop_length = 256,
                                              n_mels=128,
                                        )

    fft_db = librosa.power_to_db(fft, ref=np.max)
    return fft_db



def splitIntoChunks(mel_spec,stride):
    t = mel_spec.shape[1]
    win_size = mel_spec.shape[0]
    
    num_of_chunks = int(t/stride)
    chunks = []
    for i in range(num_of_chunks):
        chunk = mel_spec[:,i*stride:i*stride+win_size]
        if chunk.shape[1] == win_size:
            chunks.append(chunk)
    return np.stack(chunks,axis=0)



def train_val_split(data_path, ratio = 0.75):
    signals = []
    emotion_ids = defaultdict(list)

    for id, file in enumerate(os.listdir(data_path)):
        file_path = os.path.join(data_path, file)
        # Load the data
        audio, sample_rate = librosa.load(file_path, sr = SAMPLING_RATE)
        signal = np.zeros((PADDING))
        # Zero padding of the signals
        signal[:len(audio)] = audio
        signals.append(signal) 
        
        #? get the emotion of the file 
        # and put the id of the file in its list
        emotion_ids[get_emotion(file_path)].append(id)

    signals = np.stack(signals,axis=0)

    train_ids,val_ids = [],[]
    X_train,X_val = [],[]
    Y_train,Y_val = [],[]

    for emotion, emotion_id_list in emotion_ids.items():
        
        emotion_id_list = np.random.permutation(emotion_id_list)
        length = len(emotion_id_list)
        id_train = emotion_id_list[:int(ratio*length)]
        id_val = emotion_id_list[int(ratio*length):]
        
        X_train.append(signals[id_train])
        Y_train.append(np.array([emotion]*len(id_train)))
        train_ids.append(id_train)
        
        X_val.append(signals[id_val])
        Y_val.append(np.array([emotion]*len(id_val)))
        val_ids.append(id_val)
        
    X_train = np.concatenate(X_train,0)
    X_val = np.concatenate(X_val,0)
    Y_train = np.concatenate(Y_train,0)
    Y_val = np.concatenate(Y_val,0)
    train_ids = np.concatenate(train_ids,0)
    val_ids = np.concatenate(val_ids,0)
    print(f'X_train:{X_train.shape}, Y_train:{Y_train.shape}')
    print(f'X_val:{X_val.shape}, Y_val:{Y_val.shape}')

    # check if all are unique
    unique, count = np.unique(np.concatenate([train_ids,val_ids],0), return_counts=True)
    print("Number of unique indexes is {}, out of {}".format(sum(count==1), signals.shape[0]))

    return train_ids,val_ids, X_train,X_val, Y_train,Y_val

def preprocessing_data(data_path, train_val_ratio = 0.75):

    train_ids,val_ids, X_train,X_val, Y_train,Y_val = train_val_split(data_path, ratio = train_val_ratio)
    
    print("Data augmentation : Add white noise")
    aug_signals = []
    aug_labels = []
    for i in range(X_train.shape[0]):
        signal = X_train[i,:]
        augmented_signals = addAWGN(signal)
        for j in range(augmented_signals.shape[0]):
            aug_labels.append(Y_train[i])
            aug_signals.append(augmented_signals[j,:])
        print("\r Processed {}/{} files".format(i,X_train.shape[0]),end='')
    aug_signals = np.stack(aug_signals,axis=0)
    X_train = np.concatenate([X_train,aug_signals],axis=0)
    aug_labels = np.stack(aug_labels,axis=0)
    Y_train = np.concatenate([Y_train,aug_labels])


    fft_train = []
    print("Calculatin mel spectrograms for train set")
    for i in range(X_train.shape[0]):
        fft = getFFT(X_train[i,:], sample_rate=SAMPLING_RATE)
        fft_train.append(fft)
        print("\r Processed {}/{} files".format(i,X_train.shape[0]),end='')
    print('')
    del X_train

    fft_val = []
    print("Calculatin mel spectrograms for val set")
    for i in range(X_val.shape[0]):
        fft = getFFT(X_val[i,:], sample_rate=SAMPLING_RATE)
        fft_val.append(fft)
        print("\r Processed {}/{} files".format(i,X_val.shape[0]),end='')
    print('')
    del X_val


    print("Cut the spectrogram into chunks")

    fft_train_chunked = []
    for fft_spec in fft_train:
        chunks = splitIntoChunks(fft_spec, stride=42)
        fft_train_chunked.append(chunks)
    print("Number of chunks is {}".format(chunks.shape[0]))
    # val set
    fft_val_chunked = []
    for fft_spec in fft_val:
        chunks = splitIntoChunks(fft_spec, stride=42)
        fft_val_chunked.append(chunks)
    print("Number of chunks is {}".format(chunks.shape[0]))


    X_train = np.stack(fft_train_chunked,axis=0)
    X_train = np.expand_dims(X_train,2)
    print('Shape of X_train: ',X_train.shape)
    X_val = np.stack(fft_val_chunked,axis=0)
    X_val = np.expand_dims(X_val,2)
    print('Shape of X_val: ',X_val.shape)

    print("Scaling step for the inputs")
    b,t,c,h,w = X_train.shape
    X_train = np.reshape(X_train, newshape=(b,-1))
    X_train = scaler.fit_transform(X_train)
    X_train = np.reshape(X_train, newshape=(b,t,c,h,w))

    b,t,c,h,w = X_val.shape
    X_val = np.reshape(X_val, newshape=(b,-1))
    X_val = scaler.transform(X_val)
    X_val = np.reshape(X_val, newshape=(b,t,c,h,w))

    del fft_train_chunked
    del fft_train
    del fft_val_chunked
    del fft_val

    return X_train, Y_train, X_val, Y_val

def preprocessing_unit(file):
    audio, sample_rate = librosa.load(file ,sr=SAMPLING_RATE)
    signal = np.zeros((int(PADDING,)))
    signal[:len(audio)] = audio
    mel_spectrogram = getFFT(signal, SAMPLING_RATE)
    chunks = splitIntoChunks(mel_spectrogram, stride=42)
    

    X = np.stack(chunks,axis=0)
    X = np.expand_dims(X,0)
    X = np.expand_dims(X,2)

    print(X.shape)

    return X