#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019.7    Nan Lee

import os
import soundfile
import numpy as np
import argparse
import csv
import time
#import matplotlib.pyplot as plt
from scipy import signal
import pickle
import cPickle
import h5py
from sklearn import preprocessing
import fnmatch
# import prepare_data as pp_data
import config as cfg

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)
	
def calculate_train_features(args):
    """Calculate spectrogram for mixed, speech and noise audio. Then write the 
    features to disk. 
    
    Args:
      workspace: str, path of workspace. 
      data_type: str, 'train' | 'test'. 
      speech_path:str, noisy_speech_dir clean_speech_dir
    """
    data_type = args.data_type
    fs = cfg.sample_rate
    train_speech_path = args.train_speech_path
    cnt =0
    t1 = time.time()
    with open(train_speech_path,'r') as speech_org_path:
        for ii in speech_org_path:
            #read clean and noisy speech
            path_tmp = ii.split()
            noise_path = path_tmp[0]
            #out_feature_name = noise_path.split("/")[-1]
            out_feature_name = noise_path.split("/")[-1]
            (reverb_speech_audio, _) = read_audio(noise_path, target_fs=fs)
            #extract logspectram feature
            mixed_complx_x = calc_sp(reverb_speech_audio, mode='complex')
            #mixed_complx_x = np.log(mixed_complx_x + 1e-08).astype(np.float32)
            # the output feature path
            out_feat_path = os.path.join("workspace", "features", "spectrogram",data_type,"%s.p" % out_feature_name)
            create_folder(os.path.dirname(out_feat_path))
            data = [mixed_complx_x]
            cPickle.dump(data, open(out_feat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
            cnt += 1
            print cnt
    print("Extracting feature time: %s" % (time.time() - t1))
def rms(y):
    """Root mean square. 
    """
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))

def get_amplitude_scaling_factor(s, n, snr, method='rms'):
    """Given s and n, return the scaler s according to the snr. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      snr: float, SNR. 
      method: 'rms'. 
      
    Outputs:
      float, scaler. 
    """
    original_sn_rms_ratio = rms(s) / rms(n)
    target_sn_rms_ratio =  10. ** (float(snr) / 20.)    # snr = 20 * lg(rms(s) / rms(n))
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio
    return signal_scaling_factor

def calc_sp(audio, mode):
    """Calculate spectrogram. 
    
    Args:
      audio: 1darray. 
      mode: string, 'magnitude' | 'complex'
    
    Returns:
      spectrogram: 2darray, (n_time, n_freq). 
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    ham_win = np.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode=mode) 
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x
def log_sp(x):
    return np.log(x + 1e-08)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_calculate_train_features = subparsers.add_parser('calculate_train_features')
    parser_calculate_train_features.add_argument('--train_speech_path', type=str, required=True)
    parser_calculate_train_features.add_argument('--data_type', type=str, required=True)

    args = parser.parse_args()
    if args.mode == 'create_mixture_csv':
        create_mixture_csv(args)
    elif args.mode == 'calculate_train_features':
        calculate_train_features(args)
    elif args.mode == 'calculate_test_features':
        calculate_test_features(args)
    elif args.mode == 'pack_features':
        pack_features(args)       
    elif args.mode == 'compute_scaler':
        compute_scaler(args)
    else:
        raise Exception("Error!")
