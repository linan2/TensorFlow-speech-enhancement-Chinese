"""
LI, Nan
2019.08
"""
import os
import soundfile
import numpy as np
import argparse
import csv
import time
#import matplotlib.pyplot as plt
from scipy import signal
#import pickle
#import cPickle
import h5py
from sklearn import preprocessing
import librosa
import prepare_data as pp_data
import config as cfg
import math
from utils.tools import *
import random
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
    print(fs)
    return audio, fs
    
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

###
def create_mixture_csv(args):
    """Create csv containing mixture information. 
    Each line in the .csv file contains [speech_name, noise_name, noise_onset, noise_offset]
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'. 
      magnification: int, only used when data_type='train', number of noise 
          selected to mix with a speech. E.g., when magnication=3, then 4620
          speech with create 4620*3 mixtures. magnification should not larger 
          than the species of noises. 
    """
    workspace = args.workspace
    speech_dir = args.speech_dir
    noise_dir = args.noise_dir
    data_type = args.data_type
    magnification = args.magnification
    fs = cfg.sample_rate
    
    speech_names = [na for na in os.listdir(speech_dir) if na.lower().endswith(".wav")]
    noise_names = [na for na in os.listdir(noise_dir) if na.lower().endswith(".wav")]
    
    rs = np.random.RandomState(0)
    out_csv_path = os.path.join(workspace, "mixture_csvs", "%s.csv" % data_type)
    pp_data.create_folder(os.path.dirname(out_csv_path))
    
    cnt = 0
    f = open(out_csv_path, 'w')
    f.write("%s\t%s\t%s\t%s\n" % ("speech_name", "noise_name", "noise_onset", "noise_offset"))
    for speech_na in speech_names:
        # Read speech. 
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path)
        len_speech = len(speech_audio)
        
        # For training data, mix each speech with randomly picked #magnification noises. 
        if data_type == 'train':
            selected_noise_names = rs.choice(noise_names, size=magnification, replace=False)
        # For test data, mix each speech with all noises. 
        elif data_type == 'test':
            selected_noise_names = noise_names
        else:
            raise Exception("data_type must be train | test!")

        # Mix one speech with different noises many times. 
        for noise_na in selected_noise_names:
            noise_path = os.path.join(noise_dir, noise_na)
            (noise_audio, _) = read_audio(noise_path)
            
            len_noise = len(noise_audio)

            if len_noise <= len_speech:
                noise_onset = 0
                nosie_offset = len_speech
            # If noise longer than speech then randomly select a segment of noise. 
            else:
                noise_onset = rs.randint(0, len_noise - len_speech, size=1)[0]
                nosie_offset = noise_onset + len_speech
            
            if cnt % 100 == 0:
                print cnt
                
            cnt += 1
            f.write("%s\t%s\t%d\t%d\n" % (speech_na, noise_na, noise_onset, nosie_offset))
    f.close()
    print(out_csv_path)
    print("Create %s mixture csv finished!" % data_type)
    
###
def calculate_mixture_features(args):
    mixture_csv_path = os.path.join("mini_data_bak","train_speech/cleandata.txt")
    out_dir = "/Work18/2017/linan/ASR/data/aur/train"
    with open(mixture_csv_path, 'rb') as f:
        lis = list(f)
        for x in lis:
            x.replace("\n", "")
            print(x)
    print("finish read")
    noise_dir = "mini_data/Noise"
    all_noise_na = ["Babble2.wav", "F162.wav", "Factory2.wav", "Pink2.wav", "Volvo2.wav", "White2.wav"]
    all_snr = [-10, -5, 0, 5, 10, 15, 20]
    t1 = time.time()
    cnt = 0
    fs = 8000
    for i1 in xrange(0, len(lis)):
        speech_path = lis[i1].replace("\n", "")
        
        # Read speech audio. 
        (speech_audio, _) = read_audio(speech_path, target_fs=fs)
        name = speech_path.split("/")[-1]
        # Read noise audio. 
        rrr = random.randint(0,5)
        noise_na = all_noise_na[rrr]
        noise_path = os.path.join(noise_dir, noise_na)
        (noise_audio, _) = read_audio(noise_path, target_fs=fs)

        noise_len = np.shape(noise_audio)[0]
        speech_len = np.shape(speech_audio)[0]

        rdm = random.randint(0,noise_len-speech_len)
        noise_audio = noise_audio[rdm:(rdm+speech_len)]
        rrr2 = random.randint(0,6)
        snr = all_snr[rrr2]
        print("all_snr:",all_snr[rrr2])
        # Scale speech to given snr. 
        scaler = get_amplitude_scaling_factor(speech_audio, noise_audio, snr=snr)
        #speech_audio /= scaler
        
        noise_audio*=scaler
        # Get normalized mixture, speech, noise. 
        print("speech audio shape:",np.shape(speech_audio))
        print("noise audio shape:",np.shape(noise_audio))
        (mixed_audio, speech_audio, noise_audio, alpha) = additive_mixing(speech_audio, noise_audio)
        print(np.shape(speech_audio))
        print(np.shape(mixed_audio))
        tmp1 = np.sum(speech_audio**2)
        tmp2 = np.sum((mixed_audio-speech_audio)**2)

        noise2 = "noise" + str(snr)
        if snr < 0:
            snr2 = -snr
            noise2 = "noise_" + str(snr2)
        out_noise_path = os.path.join(out_dir,"noise",name)
        #audiowrite('test_speech.wav', speech_audio, samp_rate=16000)
        audiowrite(out_noise_path, noise_audio, samp_rate=fs)
        snr_bi = tmp1/tmp2
        labels = 10*np.log10(snr_bi)
        print("cacu:snr",labels)
        snr2 = "snr" + str(snr)
        if snr < 0:
            snr2 = -snr
            snr2 = "snr_" + str(snr2)
        out_put_path = os.path.join(out_dir,snr2,name)
        print(out_put_path)
        audiowrite(out_put_path, mixed_audio, samp_rate=fs)
def rms(y):
    """Root mean square. 
    """
    return np.sum(y**2)
    #return np.sqrt(sum(np.abs(y) ** 2, axis=0, keepdims=False))

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
    target_sn_rms_ratio =  10. ** (float(snr) / 10.)    # snr = 10 * lg(rms(s) / rms(n))
    signal_scaling_factor = np.sqrt(original_sn_rms_ratio/target_sn_rms_ratio)
    return signal_scaling_factor

def additive_mixing(s, n):
    """Mix normalized source1 and source2. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      
    Returns:
      mix_audio: ndarray, mixed audio. 
      s: ndarray, pad or truncated and scalered source1. 
      n: ndarray, scaled source2. 
      alpha: float, normalize coefficient. 
    """
    mixed_audio = s + n
        
    alpha = 1. / np.max(np.abs(mixed_audio))
    mixed_audio *= alpha
    s *= alpha
    n *= alpha
    return mixed_audio, s, n, alpha
    
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
    
###
    
def log_sp(x):
    return np.log(x + 1e-08)
    
###
def load_hdf5(hdf5_path):
    """Load hdf5 data. 
    """
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        x = np.array(x)     # (n_segs, n_concat, n_freq)
        y = np.array(y)     # (n_segs, n_freq)        
    return x, y

def np_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))
    
###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_mixture_csv = subparsers.add_parser('create_mixture_csv')
    parser_create_mixture_csv.add_argument('--workspace', type=str, required=True)
    parser_create_mixture_csv.add_argument('--speech_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--noise_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--data_type', type=str, required=True)
    parser_create_mixture_csv.add_argument('--magnification', type=int, default=1)

    parser_calculate_mixture_features = subparsers.add_parser('calculate_mixture_features')
    parser_calculate_mixture_features.add_argument('--workspace', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--noise_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--data_type', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--snr', type=float, required=True)
    
    parser_pack_features = subparsers.add_parser('pack_features')
    parser_pack_features.add_argument('--workspace', type=str, required=True)
    parser_pack_features.add_argument('--data_type', type=str, required=True)
    parser_pack_features.add_argument('--snr', type=float, required=True)
    parser_pack_features.add_argument('--n_concat', type=int, required=True)
    parser_pack_features.add_argument('--n_hop', type=int, required=True)
    
    parser_compute_scaler = subparsers.add_parser('compute_scaler')
    parser_compute_scaler.add_argument('--workspace', type=str, required=True)
    parser_compute_scaler.add_argument('--data_type', type=str, required=True)
    parser_compute_scaler.add_argument('--snr', type=float, required=True)
    
    args = parser.parse_args()
    if args.mode == 'create_mixture_csv':
        create_mixture_csv(args)
    elif args.mode == 'calculate_mixture_features':
        calculate_mixture_features(args)
    elif args.mode == 'pack_features':
        pack_features(args)       
    elif args.mode == 'compute_scaler':
        compute_scaler(args)
    else:
        raise Exception("Error!")
