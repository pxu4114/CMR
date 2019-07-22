import numpy as np
import pdb
import os
import cv2
import librosa
import argparse
import scipy.io.wavfile as wav
from scipy.misc import imread
from python_speech_features import mfcc
import pickle

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
  

NUMCEP = 13

def audioToInputVector(audio, fs, numcep):
    # Get MFCC coefficients
    # features = mfcc(audio, samplerate=fs, numcep=numcep, nfilt=nfilt)
    #eatures = np.mean(librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=numcep).T,axis=0) 
    
    # np array of size (t, numcep) or (t, n_mfcc)
    features = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=numcep).T 
    return features


def audiofile_to_input_vector(audio_filename, numcep):
    '''
    Given a WAV audio file at `audio_filename`, calculates `numcep` MFCC features
    at every time step.
    '''
    # Load .wav file
    audio, fs = librosa.load(audio_filename)
    return audioToInputVector(np.float32(audio), fs, numcep)

def main(args):
    print("in main")
    mode = args.mode
    server = "/shared/kgcoe-research/mil/Flickr8k_Audio/flickr_audio/wav2capt.txt"
    local = "Flickr8k_Audio/flickr_audio/wav2capt.txt"
    names = open(server,'r').readlines()
    audio_names = []
    image_names = []
    for i in range(len(names)):
        audio_names.append(names[i].split(' ')[0])
        image_names.append(names[i].split(' ')[1])     

    audio_names= sorted(audio_names)
    image_names= sorted(image_names)
    print("working..")
    server = "/shared/kgcoe-research/mil/Flickr8k/Flickr_8k.{}Images.txt".format(mode)
    local = "Flickr8k/Flickr_8k.{}Images.txt".format(mode)
    images = open(server,'r').readlines()
    images = [i.rstrip() for i in images]
    print('total train images are: %s' %len(images))
    images = sorted(images)
    # Getting only the training images and audio from the entire audio list
    index_list = [i for i,x in enumerate(image_names) if x in images]
    print('total train images are: %s' %len(index_list))
    audio = []
    for val in index_list:
        audio.append(audio_names[val])
    print('total audio files for training are: %s' %len(audio))

    sample_list = []
    lengths = []
    for i in range(0,len(audio)):
        if i%100 == 0 and i!=0 : print('extracted {}/{}'.format(i, len(audio)))
        
        # store mfcc features
        sample_list.append(audiofile_to_input_vector(os.path.join(args.data_path, audio[i]),NUMCEP))

        # store number of mfcc samples needed for wav file.
        lengths.append(sample_list[len(sample_list)-1].shape[0])
    
    np.save(os.path.join(args.save_path, "{}_sample_lengths.npy".format(mode)), np.array(lengths))

    with open(mode+"_sample_list_unprocessed.pkl", "wb") as save:
        pickle.dump(sample_list, save)
    
    print("postprocessing mfcc feature arrays..")

    #step 1 find the array with most number of rows
    max_len = max(lengths)
    #print("max length:", max_len)

    #step 2 pad the other arrays with 0's to have same number of rows as the max row array
    for sample_idx, sample in enumerate(sample_list):
        zero_arr = np.zeros([max_len-lengths[sample_idx], sample.shape[1]])
        sample_list[sample_idx] = np.vstack((sample, zero_arr))
    
    print("padding done..")

    #step 3 reshape all arrays with initial shape : (t,n_mfcc) -> (t, 1, n_mfcc) (just use reshape with args given to the left)
    for sample_idx, sample in enumerate(sample_list):
        sample_list[sample_idx] = sample.reshape((sample.shape[0], 1, sample.shape[1]))

    print("reshape done..")
    
    
    print("pickling step 3 result")
    with open("sample_list.pkl", "wb") as save:
        pickle.dump(sample_list, save)

    print("pickled step 3 results")
    #step 4 all the arrays are stores in a list together already, so do np.concatenate(arrays_list, axis=1). Hence you should have 
	# a (t * batch * n_mfcc) array.
    final_batch = np.concatenate(sample_list, axis=1)

    print("postprocessing done..")
    
    

        # pdb.set_trace()
    # pdb.set_trace()
    np.save(os.path.join(args.save_path, '{}_aud_mfcc.npy'.format(mode)), final_batch)
    print('done extracting audio features')

    if args.extract_image_features:
        from keras.preprocessing import image
        from keras.models import Model
        from keras.applications.vgg19 import VGG19
        from keras.applications.vgg19 import preprocess_input
        base_model = VGG19(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)		
        image_features = []
        for i,path in enumerate(images):
            if i % 100 == 0 and i != 0 : print('Extracted {}/{} features'.format(i,len(images)))
            img_path = os.path.join('data/f8k/Flicker8k_Dataset/',path)		
            img = image.load_img(img_path, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            feature = model.predict(img_data)
            for i in range(5):
                image_features.append(feature.flatten())
        image_features = np.asarray(image_features)
        np.save(os.path.join(args.save_path, '{}_ims.npy'.format(mode)), image_features)    
        print('done extracting image features')
        


        
        
    if args.extract_text:
        with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr8k.lemma.token.txt','r') as f:
            captions = f.read().splitlines()
        file_list=[]
        captions_list=[]
        for line in captions:
            file_list.append(line.split('#')[0])
            sentence = line.split('#')[1].split('\t')[1]
            captions_list.append(sentence)     

        sentences=[]
        cap_files = []
        for element in images:
            idx_list = [i for i,val in enumerate(file_list) if val==element]
            for i in range(5):
                sentences.append(captions_list[idx_list[i]])
                cap_files.append(file_list[idx_list[i]]) 

        with open(os.path.join(args.save_path, '{}_caps.txt'.format(mode)), 'w') as f:
            for item in sentences:
                f.write("%s\n" % item)
        

  
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--save_path', type = str, default = '/shared/kgcoe-research/mil/multi_modal_instance/new_data/f8k_precomp/', help = 'path to save the features')
    parser.add_argument('--data_path', type = str, default = '/shared/kgcoe-research/mil/Flickr8k_Audio/flickr_audio/wavs/', help = 'path to wav files')
    parser.add_argument('--mode', type=str, default='train', help='Feature extraction for which phase?')
    parser.add_argument('--extract_image_features', action='store_true')
    parser.add_argument('--extract_text', action='store_true')
    args=parser.parse_args()
    main(args)