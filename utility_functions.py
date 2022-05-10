import os, sys
import numpy as np
import pickle
import math
import pandas as pd
import torch
from scipy.signal import stft
import librosa
#import matplotlib.pyplot as plt
'''
Miscellaneous utilities
'''

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 19:22:06 2020
@author: Edoardo
"""

def save_model(model, optimizer, state, path,scheduler=None):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # save state dict of wrapped module
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if scheduler is not None:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'state': state,  # state of training loop (was 'step')
            'scheduler_state_dict' : scheduler.state_dict(),
            'random_states':(np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
        }, path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'state': state,  # state of training loop (was 'step')
            'random_states':(np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
        }, path)

def load_model(model, optimizer, path, cuda, device,scheduler=None):

    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path, map_location=device)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        # work-around for loading checkpoints where DataParallel was saved instead of inner module
        from collections import OrderedDict
        model_state_dict_fixed = OrderedDict()
        prefix = 'module.'
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith(prefix):
                k = k[len(prefix):]
            model_state_dict_fixed[k] = v
        model.load_state_dict(model_state_dict_fixed)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'state' in checkpoint:
        state = checkpoint['state']
    else:
        # older checkpoints only store step, rest of state won't be there
        state = {'step': checkpoint['step']}
    
    np.random.set_state(checkpoint['random_states'][0])
    torch.set_rng_state(checkpoint['random_states'][1].cpu())
    if torch.cuda.is_available() and checkpoint['random_states'][2] is not None:
        torch.cuda.set_rng_state(checkpoint['random_states'][2].cpu())
    return state
def readFile(path):
    
    with open(path, 'r') as f:
        r = f.read()
        r = r.replace('=', '+').replace('\n', '+').split('+')
        new_r = []
        for i in r:
            if i=='True':
                new_r.append('1')
            elif i == 'False':
                new_r.append(0)
            elif i !='' and '#' not in i:
                new_r.append(i)
    # print(new_r)
    return new_r
        
# readFile('TrainingArguments.txt')


def save_array_to_csv(file_name, array_to_save):
    f = open(file_name, "a")

    string_to_write = ""
    for elem in array_to_save:
        s = "%f," % (float(elem))
        string_to_write += s
    #Remove the last comma
    string_to_write = string_to_write[:-1]
    
    f.write(string_to_write+"\n")

    #(quite) fail proof, it's slower but in case of crashes the file is saved!
    f.close()
"""
def simple_plotter(csv_file, columns, img_name,labels,plots_folder = 'plots/'):
    for c in columns:
        f = open(csv_file, "r")
        y = []
        for line in f.readlines():
            elements = line.split(",")
            y.append(float(elements[c]))
        f.close()
        x = range(len(y))
        plt.plot(x, y, label=labels[c])
    
    plt.legend()
    plt.savefig(plots_folder + img_name,dpi=500)
    plt.clf()
    plt.cla()
    return
"""

def spectrum_fast(x, nperseg=512, noverlap=128, window='hamming', cut_dc=True,
                  output_phase=True, cut_last_timeframe=True):
    '''
    Compute magnitude spectra from monophonic signal
    '''

    f, t, seg_stft = stft(x,
                        window=window,
                        nperseg=nperseg,
                        noverlap=noverlap)

    #seg_stft = librosa.stft(x, n_fft=nparseg, hop_length=noverlap)

    output = np.abs(seg_stft)

    if output_phase:
        phase = np.angle(seg_stft)
        output = np.concatenate((output,phase), axis=-3)

    if cut_dc:
        output = output[:,1:,:]

    if cut_last_timeframe:
        output = output[:,:,:-1]

    #return np.rot90(np.abs(seg_stft))
    return output

############### ORIGINAL FUNCTION IN L3DAS21
def gen_submission_list_task2_OLD(sed, doa, max_loc_value=2.,num_frames=600, num_classes=14, max_overlaps=3):
    '''
    Process sed and doa output matrices (model's output) and generate a list of active sounds
    and their location for every frame. The list has the correct format for the Challenge results
    submission.
    '''
    output = []
    for i, (c, l) in enumerate(zip(sed, doa)):  #iterate all time frames
        c = np.round(c)  #turn to 0/1 the class predictions with threshold 0.5
        l = l * max_loc_value  #turn back locations between -2,2 as in the original dataset
        l = l.reshape(num_classes, max_overlaps, 3)  #num_class, event number, coordinates
        if np.sum(c) == 0:  #if no sounds are detected in a frame
            pass            #don't append
        else:
            for j, e in enumerate(c):  #iterate all events
                if e != 0:  #if an avent is predicted
                    #append list to output: [time_frame, sound_class, x, y, z]
                    predicted_class = int(j/max_overlaps)
                    num_event = int(j%max_overlaps)
                    curr_list = [i, predicted_class, l[predicted_class][num_event][0], l[predicted_class][num_event][1], l[predicted_class][num_event][2]]

                    output.append(curr_list)

    return np.array(output)

    ############# NEW FUNCTION TO ADAPT DCASE21 METRICS
def gen_submission_list_task2(sed, doa, max_loc_value=2.,num_frames=600, num_classes=14, max_overlaps=3):
    '''
    Process sed and doa output matrices (model's output) and generate a list of active sounds
    and their location for every frame. The list has the correct format for the Challenge results
    submission.
    '''
    _output_dict = {}
    output = []
    for i, (c, l) in enumerate(zip(sed, doa)):  #iterate all time frames
        c = np.round(c)  #turn to 0/1 the class predictions with threshold 0.5
        l = l * max_loc_value  #turn back locations between -2,2 as in the original dataset
        l = l.reshape(num_classes, max_overlaps, 3)  #num_class, event number, coordinates
        if np.sum(c) == 0:  #if no sounds are detected in a frame
            pass            #don't append
        else:
            for j, e in enumerate(c):  #iterate all events
                if e != 0:  #if an avent is predicted
                    #append list to output: [time_frame, sound_class, x, y, z]
                    predicted_class = int(j/max_overlaps)
                    num_event = int(j%max_overlaps)
                    curr_list = [i, predicted_class, l[predicted_class][num_event][0], l[predicted_class][num_event][1], l[predicted_class][num_event][2]]
                    if i not in _output_dict:
                        _output_dict[i] = []
                    _output_dict[i].append([int(predicted_class),float(l[predicted_class][num_event][0]), float(l[predicted_class][num_event][1]), float(l[predicted_class][num_event][2]),int(num_event)])
                    output.append(curr_list)

    return np.array(output), _output_dict

def csv_to_matrix_task2(path, class_dict, dur=60, step=0.1,
                        max_loc_value=2., no_overlaps=False):
    '''
    Read label csv file fro task 2 and
    Output a matrix containing 100msecs frames, each filled with
    the class ids of all sounds present and their location coordinates.
    '''
    max_overlap=3
    tot_steps =int(dur/step)
    num_classes = len(class_dict)
    num_frames = int(dur/step)
    cl = np.zeros((tot_steps, num_classes, max_overlap))
    loc = np.zeros((tot_steps, num_classes, max_overlap, 3))
    #quantize time stamp to step resolution
    quantize = lambda x: round(float(x) / step) * step
    #from quantized time resolution to output frame
    get_frame = lambda x: int(np.interp(x, (0,dur),(0,num_frames-1)))

    df = pd.read_csv(path)
    #print(df)
    for index, s in df.iterrows():  #iterate each sound in the list
        #print (s)
        #compute start and end frame position (quantizing)
        start = quantize(s['Start'])
        end = quantize(s['End'])
        start_frame = get_frame(start)
        end_frame = get_frame(end)
        class_id = class_dict[s['Class']]  #int ID of sound class name
        #print (s['Class'], class_id, start_frame, end_frame)
        #write velues in the output matrix
        sound_frames = np.arange(start_frame, end_frame+1)
        for f in sound_frames:
            pos = int(np.sum(cl[f][class_id])) #how many sounds of current class are present in current frame
            cl[f][class_id][pos] = 1.      #write detection label
            #write loc labels
            loc[f][class_id][pos][0] = s['X']
            loc[f][class_id][pos][1] = s['Y']
            loc[f][class_id][pos][2] = s['Z']
            #print (cl[f][class_id])

    loc = loc / max_loc_value  #normalize xyz (to use tanh in the model)
    #reshape arrays
    if no_overlaps:
        cl = cl[:,:,0]  #take only the non overlapped sounds
        loc = loc[:,:,0,:]
        cl = np.reshape(cl, (num_frames, num_classes))
        loc = np.reshape(loc, (num_frames, num_classes * 3))
    else:
        cl = np.reshape(cl, (num_frames, num_classes * max_overlap))
        loc = np.reshape(loc, (num_frames, num_classes * max_overlap * 3))
    #print (cl.shape, loc.shape)


    stacked = np.zeros((cl.shape[0],cl.shape[1]+loc.shape[1]))
    stacked[:,:cl.shape[1]] = cl
    stacked[:,cl.shape[1]:] = loc

    return stacked


def segment_waveforms(predictors, target, length):
    '''
    segment input waveforms into shorter frames of
    predefined length. Output lists of cut frames
    - length is in samples
    '''

    def pad(x, d):
        pad = np.zeros((x.shape[0], d))
        pad[:,:x.shape[-1]] = x
        return pad

    cuts = np.arange(0,predictors.shape[-1], length)  #points to cut
    X = []
    Y = []
    for i in range(len(cuts)):
        start = cuts[i]
        if i != len(cuts)-1:
            end = cuts[i+1]
            cut_x = predictors[:,start:end]
            cut_y = target[:,start:end]
        else:
            end = predictors.shape[-1]
            cut_x = pad(predictors[:,start:end], length)
            cut_y = pad(target[:,start:end], length)
        X.append(cut_x)
        Y.append(cut_y)
    return X, Y


def segment_task2(predictors, target, predictors_len_segment=50*8, target_len_segment=50, overlap=0.5):
    '''
    Segment input stft and target matrix of task 2 into shorter chunks.
    Default parameters cut 5-seconds frames.
    '''

    def pad(x, d):  #3d pad, padding last dim
        pad = np.zeros((x.shape[0], x.shape[1], d))
        pad[:,:,:x.shape[-1]] = x
        return pad

    target = target.reshape(1, target.shape[-1], target.shape[0])  #add dim and invert target dims so that the dim to cut is the same of predictors
    cuts_predictors = np.arange(0,predictors.shape[-1], int(predictors_len_segment*overlap))  #points to cut
    cuts_target = np.arange(0,target.shape[-1], int(target_len_segment*overlap))  #points to cut

    if len(cuts_predictors) != len(cuts_target):
        raise ValueError('Predictors and test frames should be selected to produce the same amount of frames')
    X = []
    Y = []
    for i in range(len(cuts_predictors)):
        start_p = cuts_predictors[i]
        start_t = cuts_target[i]
        end_p = start_p + predictors_len_segment
        end_t = start_t + target_len_segment

        if end_p <= predictors.shape[-1]:  #if chunk is not exceeding buffer size
            cut_x = predictors[:,:,start_p:end_p]
            cut_y = target[:,:,start_t:end_t]
        else: #if exceeding, zero padding is needed
            cut_x = pad(predictors[:,:,start_p:], predictors_len_segment)
            cut_y = pad(target[:,:,start_t:], target_len_segment)

        cut_y = np.reshape(cut_y, (cut_y.shape[-1], cut_y.shape[1]))  #unsqueeze and revert

        X.append(cut_x)
        Y.append(cut_y)

        #print (start_p, end_p, '|', start_t, end_t)
        #print (cut_x.shape, cut_y.shape)

    return X, Y


def gen_seld_out(n_frames, n_overlaps=3, n_classes=14):
    '''
    generate a fake output of the seld model
    ***only for testing
    '''
    results = []
    for frame in range(n_frames):
        n_sounds = np.random.randint(4)
        for i in range(n_sounds):
            t_class = np.random.randint(n_classes)
            tx = (np.random.sample() * 4) - 2
            ty = ((np.random.sample() * 2) - 1) * 1.5
            tz = (np.random.sample() * 2) - 1
            temp_entry = [frame, t_class, tx, ty, tz]
            #print (temp_entry)
            results.append(temp_entry)
    results = np.array(results)
    #pd.DataFrame(results).to_csv(out_path, index=None, header=None)
    return results


def gen_dummy_seld_results(out_path, n_frames=10, n_files=30, perc_tp=0.6,
                           n_overlaps=3, n_classes=14):
    '''
    generate a fake pair of seld model output and truth files
    ***only for testing
    '''

    truth_path = os.path.join(out_path, 'truth')
    pred_path = os.path.join(out_path, 'pred')
    if not os.path.exists(truth_path):
        os.makedirs(truth_path)
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    for file in range(n_files):
        #generate rtandom prediction and truth files
        pred_results = gen_seld_out(n_frames, n_overlaps, n_classes)
        truth_results = gen_seld_out(n_frames, n_overlaps, n_classes)

        #change a few entries in the pred in order to make them match
        num_truth = len(truth_results)
        num_pred = len(pred_results)
        num_tp = int(num_truth * perc_tp)
        list_entries = list(range(min(num_truth, num_pred)))
        random.shuffle(list_entries)
        truth_ids = list_entries[:num_tp]
        for t in truth_ids:
            pred_results[t] = truth_results[t]

        truth_out_file = os.path.join(truth_path, str(file) + '.csv')
        pred_out_file = os.path.join(pred_path, str(file) + '.csv')

        pd.DataFrame(truth_results).to_csv(truth_out_file, index=None, header=None)
        pd.DataFrame(pred_results).to_csv(pred_out_file, index=None, header=None)


def gen_dummy_waveforms(n, out_path):
    '''
    Generate random waveforms as example for the submission
    '''
    sr = 16000
    max_len = 10  #secs

    for i in range(n):
        len = int(np.random.sample() * max_len * sr)
        sound = ((np.random.sample(len) * 2) - 1) * 0.9
        filename = os.path.join(out_path, str(i) + '.npy')
        np.save(filename, sound)


def gen_fake_task1_dataset():
    l = []
    target = []
    for i in range(4):
        n = 160000
        n_target = 160000
        sig = np.random.sample(n)
        sig_target = np.random.sample(n_target).reshape((1, n_target))
        target.append(sig_target)
        sig = np.vstack((sig,sig,sig,sig))
        l.append(sig)

    output_path = '../prova_pickle'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    with open(os.path.join(output_path,'training_predictors.pkl'), 'wb') as f:
        pickle.dump(l, f)
    with open(os.path.join(output_path,'training_target.pkl'), 'wb') as f:
        pickle.dump(target, f)
    with open(os.path.join(output_path,'validation_predictors.pkl'), 'wb') as f:
        pickle.dump(l, f)
    with open(os.path.join(output_path,'validation_target.pkl'), 'wb') as f:
        pickle.dump(target, f)
    with open(os.path.join(output_path,'test_predictors.pkl'), 'wb') as f:
        pickle.dump(l, f)
    with open(os.path.join(output_path,'test_target.pkl'), 'wb') as f:
        pickle.dump(target, f)
    '''
    np.save(os.path.join(output_path,'training_predictors.npy'), l)
    np.save(os.path.join(output_path,'training_target.npy'), l)
    np.save(os.path.join(output_path,'validation_predictors.npy'), l)
    np.save(os.path.join(output_path,'validation_target.npy'), l)
    np.save(os.path.join(output_path,'test_predictors.npy'), l)
    np.save(os.path.join(output_path,'test_target.npy'), l)
    '''

    with open(os.path.join(output_path,'training_predictors.pkl'), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(output_path,'training_target.pkl'), 'rb') as f:
        data2 = pickle.load(f)

    print (data[0].shape)
    print (data2[0].shape)
