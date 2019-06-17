import numpy as np
import pdb
import os
import cv2
import librosa
import argparse
import scipy.io.wavfile as wav
from scipy.misc import imread
from python_speech_features import mfcc

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

def remap(arr):
	old_size = arr.shape[1]
	new_shape = 100
	new_arr = np.zeros((arr.shape[0],new_shape))
	new_idxs = [i*old_size//new_shape + old_size//(2*new_shape) for i in range(new_shape)]
	for n,idx in enumerate(new_idxs):
		new_arr[:,n] = arr[:,idx]
	return new_arr
    
    
def audioToInputVector(audio, fs, numcep, nfilt):
	# Get MFCC coefficients
	features = mfcc(audio, samplerate=fs, numcep=numcep, nfilt=nfilt)
	return features


def audiofile_to_input_vector(audio_filename, numcep, nfilt):
    '''
    Given a WAV audio file at `audio_filename`, calculates `numcep` MFCC features
    at every time step.
    '''
    # Load .wav file
    fs, audio = wav.read(audio_filename)
    #pdb.set_trace()
    return audioToInputVector(np.int16(audio), fs, numcep, nfilt)
    return audioToInputVector(np.float32(audio), fs, n_mfcc)

def main(args):
    mode = args.mode
    names = open('/shared/kgcoe-research/mil/Flickr8k_Audio/flickr_audio/wav2capt.txt','r').readlines()
    audio_names = []
    image_names = []
    for i in range(len(names)):
        audio_names.append(names[i].split(' ')[0])
        image_names.append(names[i].split(' ')[1])     

    audio_names= sorted(audio_names)
    image_names= sorted(image_names)
    images = open('/shared/kgcoe-research/mil/Flickr8k/Flickr_8k.{}Images.txt'.format(mode),'r').readlines()
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

    audio_stack = np.array([])
    mean_stack = np.array([])
    for i in range(0,len(audio)):
        if i%100 == 0 and i!=0 : print('extracted {}/{}'.format(i, len(audio)))
        # af = audiofile_to_input_vector(os.path.join(args.data_path, audio[i]),29)
        af = audiofile_to_input_vector(os.path.join(args.data_path, audio[i]),29,29)
        # pdb.set_trace()
        mean_af = np.mean(af,axis=0)
        avg = np.mean(af,axis=0)
        avg = np.reshape(avg,(1,29))
        
        if len(audio_stack.shape)>1:
            audio_stack = np.vstack((audio_stack,avg))
        else:
            audio_stack = avg
    # pdb.set_trace()
    np.save(os.path.join(args.save_path, '{}_aud.npy'.format(mode)), audio_stack)
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