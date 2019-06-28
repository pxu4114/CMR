import numpy as np
import pdb
import os
import json
import cv2
import librosa
import argparse
import scipy.io.wavfile as wav
from scipy.misc import imread
from python_speech_features import mfcc

def audioToInputVector(audio, fs, numcep, nfilt):
    # Get MFCC coefficients
    # features = mfcc(audio, samplerate=fs, numcep=numcep, nfilt=nfilt)
    features = np.mean(librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=numcep).T,axis=0) 
    return features


def audiofile_to_input_vector(audio_filename, numcep, nfilt):
    '''
    Given a WAV audio file at `audio_filename`, calculates `numcep` MFCC features
    at every time step.
    '''
    # Load .wav file
    audio, fs = librosa.load(audio_filename)
    return audioToInputVector(np.float32(audio), fs, numcep, nfilt)

def main(args):
    anno = open('/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/annotations/captions_train2014.json', 'r')
    anno = json.load(anno)
    image_id = []
    caption_id = []
    captions = []
    for i in range(len(anno['annotations'])):
        caption_id.append(anno['annotations'][i]['id'])
        image_id.append(anno['annotations'][i]['image_id'])
        captions.append(anno['annotations'][i]['caption'])
        
    train_id = np.load('/shared/kgcoe-research/mil/multi_modal_instance/data/data/coco/annotations/coco_{}_ids.npy'.format(args.mode))
    train_id = set(list(train_id))

    index = [i for i, item in enumerate(caption_id) if item in train_id]
    new_captions = [captions[i] for i in index]
    new_image_id = [image_id[i] for i in index]
    new_index = sorted(range(len(new_image_id)), key=lambda k: new_image_id[k])
    new_image_id.sort()
    new_captions = [new_captions[i] for i in new_index]

	# if extract_audio:
		

    if args.extract_image_features:
        from keras.preprocessing import image
        from keras.models import Model
        from keras.applications.vgg19 import VGG19
        from keras.applications.vgg19 import preprocess_input
        base_model = VGG19(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)		
        image_features = []
        for i in range(len(new_image_id)/5):
            path = "%012d" % (new_image_id[i*5],)
        # /shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images/train2014/COCO_train2014_000000000009.jpg
            if i % 1000 == 0 and i != 0 : print('Extracted {}/{} features'.format(i,len(new_image_id)/5))
            img_path = '/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images/train2014/COCO_train2014_{}.jpg'.format(path)	
            img = image.load_img(img_path, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            feature = model.predict(img_data)
            for i in range(5):
                image_features.append(feature.flatten())
        image_features = np.asarray(image_features)
        np.save(os.path.join(args.save_path, '{}_ims.npy'.format(args.mode)), image_features)    
        print('done extracting image features')
    
    if args.extract_text:
        with open(os.path.join(args.save_path, '{}_caps.txt'.format(args.mode)), 'w') as f:
            for item in new_captions:
                f.write("%s\n" % item)    
# pdb.set_trace()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--save_path', type = str, default = '/shared/kgcoe-research/mil/multi_modal_instance/new_data/coco_precomp', help = 'path to save the features')
    parser.add_argument('--data_path', type = str, default = '/shared/kgcoe-research/mil/Flickr8k_Audio/flickr_audio/wavs/', help = 'path to wav files')
    parser.add_argument('--mode', type=str, default='test', help='Feature extraction for which phase?')
    parser.add_argument('--extract_image_features', action='store_false')
    parser.add_argument('--extract_text', action='store_false')
    parser.add_argument('--extract_audio', action='store_false')
    args=parser.parse_args()
    main(args)