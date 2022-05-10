
#from evaluate_baseline_task2 import evaluate_model
import sys, os
import time
import json
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR 
import torch.utils.data as utils
from models.SELD_Model import SELD_Model
from utility_functions import save_array_to_csv,gen_submission_list_task2, readFile
from metrics import location_sensitive_detection
import shutil
from torchinfo import summary
import wandb
from Dcase21_metrics import *


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


def evaluate_test(model,device, dataloader,epoch=0,max_loc_value=2.,num_frames=600,spatial_threshold=2.):
    TP = 0
    FP = 0
    FN = 0
    count = 0
    output_classes=args.output_classes
    class_overlaps=args.class_overlaps

    model.eval()
    
    eval_metrics = SELDMetrics(nb_classes=output_classes, doa_threshold=args.Dcase21_metrics_DOA_threshold)
    
    with tqdm(total=len(dataloader) // 1) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            x = x.to(device)
            target = target.to(device)
            
            sed, doa = model(x)
            sed = sed.cpu().numpy().squeeze()
            doa = doa.cpu().numpy().squeeze()
            target = target.cpu().numpy().squeeze()
            #in the target matrices sed and doa are joint
            sed_target = target[:,:args.output_classes*args.class_overlaps]
            doa_target = target[:,args.output_classes*args.class_overlaps:]

            
            prediction,prediction_dict = gen_submission_list_task2(sed, doa,
                                                    max_overlaps=class_overlaps,
                                                    max_loc_value=max_loc_value)

            target,target_dict = gen_submission_list_task2(sed_target, doa_target,
                                                max_overlaps=class_overlaps,
                                                max_loc_value=max_loc_value)
    
            pred_labels =segment_labels(prediction_dict, num_frames)
            ref_labels =segment_labels(target_dict, num_frames)
            # Calculated scores
            eval_metrics.update_seld_scores(pred_labels, ref_labels)
            tp, fp, fn, _ = location_sensitive_detection(prediction, target, num_frames,
                                                      spatial_threshold, False)
            TP += tp
            FP += fp
            FN += fn

            count += 1
            pbar.update(1)


    #compute total F score
    precision = TP / (TP + FP + sys.float_info.epsilon)
    recall = TP / (TP + FN + sys.float_info.epsilon)
    F_score = 2 * ((precision * recall) / (precision + recall + sys.float_info.epsilon))
    Nref=TP+FN
    Nsys=TP+FP
    ER_score = (max(Nref, Nsys) - TP) / (Nref + 0.0)
    
    ER_dcase21, F_dcase21, LE_dcase21, LR_dcase21 = eval_metrics.compute_seld_scores()

    SELD_dcase21 = np.mean([ER_dcase21,1 -  F_dcase21, LE_dcase21/180,1 - LR_dcase21])
    SELD_L3DAS21_LRLE = np.mean([ER_score,1 -  F_score, LE_dcase21/180,1 - LR_dcase21])
    CSL_score= np.mean([LE_dcase21/180,1 - LR_dcase21])
    LSD_score=np.mean([1-F_score,ER_score])
    test_results=[epoch,F_score,ER_score,precision,recall,TP,FP,FN,
                    CSL_score,LSD_score,SELD_L3DAS21_LRLE,
                    SELD_dcase21,ER_dcase21, F_dcase21, LE_dcase21, LR_dcase21]
    

    #visualize and save results
    print ('*******************************')
    print ('RESULTS')
    print  ('TP: ' , TP)
    print  ('FP: ' , FP)
    print  ('FN: ' , FN)
    print ('******** SELD (F ER L3DAS21 - LE LR DCASE21) ***********')
    print ('Global SELD score: ', SELD_L3DAS21_LRLE)
    print ('LSD score: ', LSD_score)
    print ('CSL score: ', CSL_score)
    print ('F score: ', F_score)
    print ('ER score: ', ER_score)
    print ('LE: ', LE_dcase21)
    print ('LR: ', LR_dcase21)
    
    return test_results

def evaluate(model, device, criterion_sed, criterion_doa, dataloader):
    #compute loss without backprop
    model.eval()
    test_loss = 0.
    with tqdm(total=len(dataloader) // args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            target = target.to(device)
            x = x.to(device)
            t = time.time()
            # Compute loss for each instrument/model
            #sed, doa = model(x)
            loss = seld_loss(x, target, model, criterion_sed, criterion_doa)
            test_loss += (1. / float(example_num + 1)) * (loss - test_loss)
            pbar.set_description("Current loss: {:.4f}".format(test_loss))
            pbar.update(1)
    return test_loss


def seld_loss(x, target, model, criterion_sed, criterion_doa):
    '''
    compute seld loss as weighted sum of sed (BCE) and doa (MSE) losses
    '''
    #divide labels into sed and doa  (which are joint from the preprocessing)
    target_sed = target[:,:,:args.output_classes*args.class_overlaps]
    target_doa = target[:,:,args.output_classes*args.class_overlaps:]

    #compute loss
    sed, doa = model(x)
    
    sed = torch.flatten(sed, start_dim=1)
    doa = torch.flatten(doa, start_dim=1)
    target_sed = torch.flatten(target_sed, start_dim=1)
    target_doa = torch.flatten(target_doa, start_dim=1)
    loss_sed = criterion_sed(sed, target_sed) * args.sed_loss_weight
    loss_doa = criterion_doa(doa, target_doa) * args.doa_loss_weight
    
    return loss_sed + loss_doa


def main(args):

    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if args.fixed_seed:
        seed = 1
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    #LOAD DATASET
    print ('\nLoading dataset')

    with open(args.training_predictors_path, 'rb') as f:
        training_predictors = pickle.load(f)
    with open(args.training_target_path, 'rb') as f:
        training_target = pickle.load(f)
    with open(args.validation_predictors_path, 'rb') as f:
        validation_predictors = pickle.load(f)
    with open(args.validation_target_path, 'rb') as f:
        validation_target = pickle.load(f)
    with open(args.test_predictors_path, 'rb') as f:
        test_predictors = pickle.load(f)
    with open(args.test_target_path, 'rb') as f:
        test_target = pickle.load(f)

    phase_string='_Phase' if args.phase else ''
    dataset_string='L3DAS21_'+str(args.n_mics)+'Mics_Magnidute'+phase_string+'_'+str(args.input_channels)+'Ch'
    #####################################NORMALIZATION####################################
    if args.dataset_normalization not in {'False','false','None','none'}:
        print('\nDataset_Normalization')
        if args.dataset_normalization in{'DQ_Normalization','UnitNormNormalization','UnitNorm'}:
        
            training_predictors = torch.tensor(training_predictors)
            training_target = torch.tensor(training_target)
            validation_predictors = torch.tensor(validation_predictors)
            validation_target = torch.tensor(validation_target)
            test_predictors = torch.tensor(test_predictors)
            test_target = torch.tensor(test_target)
            if args.n_mics==2:
                if args.domain in ['DQ','dq','dQ','Dual_Quaternion','dual_quaternion']:
                    dataset_string+=' Dataset Normalization for 2Mic 8Ch Magnitude Dual Quaternion UnitNorm'
                    print('Dataset Normalization for 2Mic 8Ch Magnitude Dual Quaternion UnitNorm')
                    ## TRAINING PREDICTORS ##
                    q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3 = torch.chunk(training_predictors[:,:8,:,:], chunks=8, dim=1)
                    denominator_0 = q_0 ** 2 + q_1 ** 2 + q_2 ** 2 + q_3 ** 2
                    denominator_1 = torch.sqrt(denominator_0)
                    deno_cross = q_0 * p_0 + q_1 * p_1 + q_2 * p_2 + q_3 * p_3

                    p_0 = p_0 - deno_cross / denominator_0 * q_0
                    p_1 = p_1 - deno_cross / denominator_0 * q_1
                    p_2 = p_2 - deno_cross / denominator_0 * q_2
                    p_3 = p_3 - deno_cross / denominator_0 * q_3

                    q_0 = q_0 / denominator_1
                    q_1 = q_1 / denominator_1
                    q_2 = q_2 / denominator_1
                    q_3 = q_3 / denominator_1

                    training_predictors[:,:8,:,:] = torch.cat([q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3], dim=1)

                    ## VALIDATION PREDICTORS ##
                    q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3 = torch.chunk(validation_predictors[:,:8,:,:], chunks=8, dim=1)
                    denominator_0 = q_0 ** 2 + q_1 ** 2 + q_2 ** 2 + q_3 ** 2
                    denominator_1 = torch.sqrt(denominator_0)
                    deno_cross = q_0 * p_0 + q_1 * p_1 + q_2 * p_2 + q_3 * p_3

                    p_0 = p_0 - deno_cross / denominator_0 * q_0
                    p_1 = p_1 - deno_cross / denominator_0 * q_1
                    p_2 = p_2 - deno_cross / denominator_0 * q_2
                    p_3 = p_3 - deno_cross / denominator_0 * q_3

                    q_0 = q_0 / denominator_1
                    q_1 = q_1 / denominator_1
                    q_2 = q_2 / denominator_1
                    q_3 = q_3 / denominator_1

                    validation_predictors[:,:8,:,:] = torch.cat([q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3], dim=1)

                    ## TEST PREDICTORS ##
                    q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3 = torch.chunk(test_predictors[:,:8,:,:], chunks=8, dim=1)
                    denominator_0 = q_0 ** 2 + q_1 ** 2 + q_2 ** 2 + q_3 ** 2
                    denominator_1 = torch.sqrt(denominator_0)
                    deno_cross = q_0 * p_0 + q_1 * p_1 + q_2 * p_2 + q_3 * p_3

                    p_0 = p_0 - deno_cross / denominator_0 * q_0
                    p_1 = p_1 - deno_cross / denominator_0 * q_1
                    p_2 = p_2 - deno_cross / denominator_0 * q_2
                    p_3 = p_3 - deno_cross / denominator_0 * q_3

                    q_0 = q_0 / denominator_1
                    q_1 = q_1 / denominator_1
                    q_2 = q_2 / denominator_1
                    q_3 = q_3 / denominator_1

                    test_predictors[:,:8,:,:] = torch.cat([q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3], dim=1) 
                    if args.phase:
                        raise ValueError('DATASET NORMALIZATION FOR PHASE DUAL QUATERNION NOT YET IMPLEMENTED')
                        print('Dataset Normalization for 2Mic 16Ch Magnitude-Phase Dual Quaternion ')
                    training_predictors = np.array(training_predictors)
                    training_target = np.array(training_target)
                    validation_predictors = np.array(validation_predictors)
                    validation_target = np.array(validation_target)
                    test_predictors = np.array(test_predictors)
                    test_target = np.array(test_target)

                    print ('\nShapes:')
                    print ('Training predictors: ', training_predictors.shape)
                    print ('Validation predictors: ', validation_predictors.shape)
                    print ('Test predictors: ', test_predictors.shape)
                    print ('Training target: ', training_target.shape)
                    print ('Validation target: ', validation_target.shape)
                    print ('Test target: ', test_target.shape)
        else:
            training_predictors = np.array(training_predictors)
            training_target = np.array(training_target)
            validation_predictors = np.array(validation_predictors)
            validation_target = np.array(validation_target)
            test_predictors = np.array(test_predictors)
            test_target = np.array(test_target)

            print ('\nShapes:')
            print ('Training predictors: ', training_predictors.shape)
            print ('Validation predictors: ', validation_predictors.shape)
            print ('Test predictors: ', test_predictors.shape)
            print ('Training target: ', training_target.shape)
            print ('Validation target: ', validation_target.shape)
            print ('Test target: ', test_target.shape)
            if args.n_mics==1:
                dataset_string+=' Dataset Normalization for 1Mic 4Ch Magnitude'
                print('Dataset Normalization for 1Mic 4Ch Magnitude')
                # Normalize training predictors with mean 0 and std 1
                train_mag_min = np.mean(training_predictors[:,:4,:,:])
                train_mag_std = np.std(training_predictors[:,:4,:,:])  
                training_predictors[:,:4,:,:] -= train_mag_min
                training_predictors[:,:4,:,:] /= train_mag_std
                # Normalize validation predictors with mean 0 and std 1
                val_mag_min = np.mean(validation_predictors[:,:4,:,:])
                val_mag_std = np.std(validation_predictors[:,:4,:,:])    
                validation_predictors[:,:4,:,:] -= val_mag_min
                validation_predictors[:,:4,:,:] /= val_mag_std
                # Normalize test predictors with mean 0 and std 1
                test_mag_min = np.mean(test_predictors[:,:4,:,:])
                test_mag_std = np.std(test_predictors[:,:4,:,:])    
                test_predictors[:,:4,:,:] -= test_mag_min
                test_predictors[:,:4,:,:] /= test_mag_std
                if args.phase:
                    dataset_string+=' Dataset Normalization for 1Mic 8Ch Magnitude-Phase'
                    print('Dataset Normalization for 1Mic 8Ch Magnitude-Phase')
                    train_phase_min = np.mean(training_predictors[:,4:,:,:])
                    train_phase_std = np.std(training_predictors[:,4:,:,:])
                    training_predictors[:,4:,:,:] -= train_phase_min
                    training_predictors[:,4:,:,:] /= train_phase_std
                    val_phase_min = np.mean(validation_predictors[:,4:,:,:])
                    val_phase_std = np.std(validation_predictors[:,4:,:,:])
                    validation_predictors[:,4:,:,:] -= val_phase_min
                    validation_predictors[:,4:,:,:] /= val_phase_std
                    test_phase_min = np.mean(test_predictors[:,4:,:,:])
                    test_phase_std = np.std(test_predictors[:,4:,:,:])
                    test_predictors[:,4:,:,:] -= test_phase_min
                    test_predictors[:,4:,:,:] /= test_phase_std
            if args.n_mics==2:
                
                dataset_string+=' Dataset Normalization for 2Mic 8Ch Magnitude'
                print('Dataset Normalization for 2Mic 8Ch Magnitude')
                # Normalize training predictors with mean 0 and std 1
                train_mag_min = np.mean(training_predictors[:,:8,:,:])
                train_mag_std = np.std(training_predictors[:,:8,:,:])  
                training_predictors[:,:8,:,:] -= train_mag_min
                training_predictors[:,:8,:,:] /= train_mag_std
                # Normalize validation predictors with mean 0 and std 1
                val_mag_min = np.mean(validation_predictors[:,:8,:,:])
                val_mag_std = np.std(validation_predictors[:,:8,:,:])    
                validation_predictors[:,:8,:,:] -= val_mag_min
                validation_predictors[:,:8,:,:] /= val_mag_std
                # Normalize test predictors with mean 0 and std 1
                test_mag_min = np.mean(test_predictors[:,:8,:,:])
                test_mag_std = np.std(test_predictors[:,:8,:,:])    
                test_predictors[:,:8,:,:] -= test_mag_min
                test_predictors[:,:8,:,:] /= test_mag_std
                if args.phase:
                
                    dataset_string+=' Dataset Normalization for 2Mic 16Ch Magnitude-Phase'
                    print('Dataset Normalization for 2Mic 16Ch Magnitude-Phase')
                    train_phase_min = np.mean(training_predictors[:,8:,:,:])
                    train_phase_std = np.std(training_predictors[:,8:,:,:])
                    training_predictors[:,8:,:,:] -= train_phase_min
                    training_predictors[:,8:,:,:] /= train_phase_std
                    val_phase_min = np.mean(validation_predictors[:,8:,:,:])
                    val_phase_std = np.std(validation_predictors[:,8:,:,:])
                    validation_predictors[:,8:,:,:] -= val_phase_min
                    validation_predictors[:,8:,:,:] /= val_phase_std
                    test_phase_min = np.mean(test_predictors[:,8:,:,:])
                    test_phase_std = np.std(test_predictors[:,8:,:,:])
                    test_predictors[:,8:,:,:] -= test_phase_min
                    test_predictors[:,8:,:,:] /= test_phase_std
    else:
        training_predictors = np.array(training_predictors)
        training_target = np.array(training_target)
        validation_predictors = np.array(validation_predictors)
        validation_target = np.array(validation_target)
        test_predictors = np.array(test_predictors)
        test_target = np.array(test_target)

        print ('\nShapes:')
        print ('Training predictors: ', training_predictors.shape)
        print ('Validation predictors: ', validation_predictors.shape)
        print ('Test predictors: ', test_predictors.shape)
        print ('Training target: ', training_target.shape)
        print ('Validation target: ', validation_target.shape)
        print ('Test target: ', test_target.shape)
    
    ###############################################################################


    features_dim = int(test_target.shape[-2] * test_target.shape[-1])

    #convert to tensor
    training_predictors = torch.tensor(training_predictors).float()
    validation_predictors = torch.tensor(validation_predictors).float()
    test_predictors = torch.tensor(test_predictors).float()
    training_target = torch.tensor(training_target).float()
    validation_target = torch.tensor(validation_target).float()
    test_target = torch.tensor(test_target).float()
    #build dataset from tensors
    tr_dataset = utils.TensorDataset(training_predictors, training_target)
    val_dataset = utils.TensorDataset(validation_predictors, validation_target)
    test_dataset = utils.TensorDataset(test_predictors, test_target)
    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, args.batch_size, shuffle=True, pin_memory=True)
    val_data = utils.DataLoader(val_dataset, args.batch_size, shuffle=False, pin_memory=True)
    test_data = utils.DataLoader(test_dataset, 1, shuffle=False, pin_memory=True)#(test_dataset, args.batch_size, shuffle=False, pin_memory=True

    #LOAD MODEL
    n_time_frames = test_predictors.shape[-1]

    ######################################################################################################################
    model=SELD_Model(time_dim=n_time_frames, freq_dim=args.freq_dim, input_channels=args.input_channels, output_classes=args.output_classes,
                 domain=args.domain, domain_classifier=args.domain_classifier,
                 cnn_filters=args.cnn_filters, kernel_size_cnn_blocks=args.kernel_size_cnn_blocks, pool_size=args.pool_size, pool_time=args.pool_time,
                 D=args.D, dilation_mode=args.dilation_mode,G=args.G, U=args.U, kernel_size_dilated_conv=args.kernel_size_dilated_conv,
                 spatial_dropout_rate=args.spatial_dropout_rate,V=args.V, V_kernel_size=args.V_kernel_size,
                 fc_layers=args.fc_layers, fc_activations=args.fc_activations, fc_dropout=args.fc_dropout, dropout_perc=args.dropout_perc, 
                 class_overlaps=args.class_overlaps,
                 use_bias_conv=args.use_bias_conv,use_bias_linear=args.use_bias_linear,batch_norm=args.batch_norm,  parallel_ConvTC_block=args.parallel_ConvTC_block, parallel_magphase=args.parallel_magphase,
                 extra_name=args.model_extra_name, verbose=False)
    
                 
    architecture_dir='RESULTS/Task2/{}/'.format(args.architecture)
    if len(os.path.dirname(architecture_dir)) > 0 and not os.path.exists(os.path.dirname(architecture_dir)):
        os.makedirs(os.path.dirname(architecture_dir))
    model_dir=architecture_dir+model.model_name+'/'
    if len(os.path.dirname(model_dir)) > 0 and not os.path.exists(os.path.dirname(model_dir)):
        os.makedirs(os.path.dirname(model_dir))
    args.load_model=model_dir+'checkpoint'
    unique_name=model_dir+model.model_name
    
    '''if not args.wandb_id=='none': 
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,resume='allow',id=args.wandb_id,name=model.model_name)############################################################################################ WANDB
    else:
        wandb.init(project=args.wandb_project,entity=args.wandb_entity,resume='allow',name=model.model_name)
    config = wandb.config
    wandb.watch(model)
    wandb.config.update(args, allow_val_change=True)
    wandb.config.ReceptiveField=model.receptive_field
    wandb.config.n_ResBlocks=model.total_n_resblocks'''
    
    print(dataset_string)
    print(model.model_name)
    
    summary(model, input_size=(args.batch_size,args.input_channels,args.freq_dim,n_time_frames)) ##################################################
    if not args.architecture == 'seldnet_vanilla' and not args.architecture == 'seldnet_augmented': 
        print('\nReceptive Field: ',model.receptive_field,'\nNumber of ResBlocks: ', model.total_n_resblocks)
    #######################################################################################################################
    if args.use_cuda:
        print("Moving model to gpu")
    model = model.to(device)

    #compute number of parameters
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total paramters: ' + str(model_params))
    '''
    wandb.config.n_Parameters=model_params'''

    #set up the loss functions
    criterion_sed = nn.BCELoss()
    criterion_doa = nn.MSELoss()

    #set up optimizer
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    
    ################################################################### DYNAMIC LEARNING RATE
    if args.use_lr_scheduler:
        scheduler = StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma, verbose=True)
    else:
        scheduler=None
    ###################################################################
    #set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf,
             "best_epoch" : 0,
             "best_test_epoch":0,
             "torch_seed_state":torch.get_rng_state(),
             "numpy_seed_state":np.random.get_state()
            
             }
    epoch =0
    best_loss_checkpoint=np.inf
    best_test_metric=1
    #load model checkpoint if desired
    if args.load_model is not None and os.path.isfile(args.load_model) :####################################### added "and os.path.isfile(args.load_model)"
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = load_model(model, optimizer, args.load_model, args.use_cuda,device,scheduler)
        epoch=state["epochs"]#######################################################################
    new_best=False
    test_best_results=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    best_epoch_checkpoint = epoch
    

    #TRAIN MODEL
    print('TRAINING START')
    train_loss_hist = []
    val_loss_hist = []
    while state["worse_epochs"] < args.patience or epoch<args.min_n_epochs:
        epoch += 1
        state["epochs"] += 1
        print("Training epoch " + str(epoch) +' of '+model.model_name, ' with lr ', optimizer.param_groups[0]['lr'])
        avg_time = 0.
        model.train()
        train_loss = 0.
        with tqdm(total=len(tr_dataset) // args.batch_size) as pbar:
            for example_num, (x, target) in enumerate(tr_data):
                target = target.to(device)
                #print(x.shape)
                x = x.to(device)
                t = time.time()
                # Compute loss for each instrument/model
                optimizer.zero_grad()
                #print(x.shape)
                #sed, doa = model(x)
                #print(x.shape)
                loss = seld_loss(x, target, model, criterion_sed, criterion_doa)
                loss.backward()

                train_loss += (1. / float(example_num + 1)) * (loss - train_loss)
                optimizer.step()
                state["step"] += 1
                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                pbar.update(1)

            #PASS VALIDATION DATA
            val_loss = evaluate(model, device, criterion_sed, criterion_doa, val_data)
            
            if args.use_lr_scheduler and optimizer.param_groups[0]['lr']>args.min_lr:
                scheduler.step()######################################################################Dynamic learning rate
            

            # EARLY STOPPING CHECK
            #############################################################################
            
            checkpoint_path = os.path.join(model_dir, "checkpoint")
            checkpoint_best_model_path = os.path.join(model_dir, "checkpoint_best_model")
            checkpoint_best_model_checkpoint_path = os.path.join(model_dir, "checkpoint_best_model_of_checkpoint")


            
            #state["worse_epochs"] = 200
            train_loss_hist.append(train_loss.cpu().detach().numpy())
            val_loss_hist.append(val_loss.cpu().detach().numpy())


            if val_loss >= state["best_loss"]:
                state["worse_epochs"] += 1
                
            else:
                if new_best==True:
                    best_loss_checkpoint =state["best_loss"] 
                    best_epoch_checkpoint = state["best_epoch"]
                    shutil.copyfile(checkpoint_best_model_path, checkpoint_best_model_checkpoint_path)
                    
                print("MODEL IMPROVED ON VALIDATION SET!")
                state["worse_epochs"] = 0
                state["best_loss"] = val_loss
                state["best_epoch"] = epoch
                state["best_checkpoint"] = checkpoint_best_model_path
                new_best=True

                # CHECKPOINT
                print("Saving best model...")
                save_model(model, optimizer, state, checkpoint_best_model_path,scheduler)

            if val_loss < best_loss_checkpoint and (val_loss!=state["best_loss"] or best_loss_checkpoint==np.inf):
                best_loss_checkpoint = val_loss
                print("Saving best model checkpoint...")
                save_model(model, optimizer, state, checkpoint_best_model_checkpoint_path,scheduler)
                best_epoch_checkpoint = epoch


            print("Saving model...")
            save_model(model, optimizer, state, checkpoint_path,scheduler)
            print("VALIDATION FINISHED: TRAIN_LOSS: {}  VAL_LOSS: {}".format(str(train_loss.cpu().detach().numpy().round(4)), str(val_loss.cpu().detach().numpy().round(4))))
            print("Best epoch at: {} Best loss: {}".format(state['best_epoch'],str(state['best_loss'].cpu().detach().numpy().round(4))))

            plot_array=[epoch, train_loss.cpu().detach().numpy(), val_loss.cpu().detach().numpy()]
            save_array_to_csv("{}_training_metrics.csv".format(unique_name), plot_array)###################################
            
            '''wandb.log({"train loss": train_loss.cpu().detach().numpy()},step=epoch)#################################################### WANDB
            wandb.log({"val loss":val_loss.cpu().detach().numpy()},step=epoch)
            '''

            #TEST############################################################################################################
            if epoch%args.test_step==0:
                if args.test_mode=='test_best':
                    if new_best:
                        print ('\n***************TEST BEST MODEL AT EPOCH {}****************'.format(state["best_epoch"]))
                        state = load_model(model, optimizer, checkpoint_best_model_path, args.use_cuda,device,scheduler)
                        test_best_results=evaluate_test(model,device, test_data,epoch=state['best_epoch'],max_loc_value=args.max_loc_value,num_frames=args.num_frames,spatial_threshold=args.spatial_threshold)
                        save_array_to_csv("{}_test_metrics.csv".format(unique_name), test_best_results)
                    else:
                        print ('\n***************TEST MODEL AT EPOCH {}****************'.format(best_epoch_checkpoint))
                        state = load_model(model, optimizer, checkpoint_best_model_checkpoint_path, args.use_cuda,device,scheduler)
                        test_best_results=evaluate_test(model,device, test_data,epoch=best_epoch_checkpoint,max_loc_value=args.max_loc_value,num_frames=args.num_frames,spatial_threshold=args.spatial_threshold)
                        save_array_to_csv("{}_test_metrics.csv".format(unique_name), test_best_results)
                else:
                    print ('\n***************TEST MODEL AT EPOCH {}****************'.format(epoch))
                    test_best_results=evaluate_test(model,device, test_data,epoch=epoch,max_loc_value=args.max_loc_value,num_frames=args.num_frames,spatial_threshold=args.spatial_threshold)
                    save_array_to_csv("{}_test_metrics.csv".format(unique_name), test_best_results)
                '''
                wandb.log({"F-Score": test_best_results[1]},step=epoch)#################################################### WANDB
                wandb.log({"ER-Score": test_best_results[2]},step=epoch)
                wandb.log({"Precision": test_best_results[3]},step=epoch)
                wandb.log({"Recall": test_best_results[4]},step=epoch)
                wandb.log({"LR Localization Recall (DCASE21)": test_best_results[-1]},step=epoch)
                wandb.log({"LE Localization Error (DCASE21)": test_best_results[-2]},step=epoch)
                wandb.log({"F (DCASE21)": test_best_results[-3]},step=epoch)
                wandb.log({"ER (DCASE21)": test_best_results[-4]},step=epoch)
                wandb.log({"SELD Score (DCASE21)": test_best_results[-5]},step=epoch)      
                wandb.log({"Global SELD (F ER L3DAS21 - LE LR DCASE21)": test_best_results[-6]},step=epoch)    
                wandb.log({"LSD score": test_best_results[-7]},step=epoch)    
                wandb.log({"CSL score": test_best_results[-8]},step=epoch) '''          
                
                if args.test_mode=='test_best':
                    state = load_model(model, optimizer, args.load_model, args.use_cuda,device,scheduler)
                if new_best:
                    new_best=False       
            
            if epoch% args.checkpoint_step==0:
                checkpoint_dir=model_dir+'checkpoint_epoch_{}/'.format(epoch)
                if len(os.path.dirname(checkpoint_dir)) > 0 and not os.path.exists(os.path.dirname(checkpoint_dir)):
                    os.makedirs(os.path.dirname(checkpoint_dir))
                print ('\n***************CHECKPOINT EPOCH {}****************'.format(epoch))
                shutil.copyfile(checkpoint_best_model_path, checkpoint_dir+"checkpoint_best_epoch_{}".format(state["best_epoch"]))            
                shutil.copyfile(checkpoint_path, checkpoint_dir+"checkpoint_epoch_{}".format(epoch))            
                shutil.copyfile(checkpoint_path+'_best_model_on_Test', checkpoint_dir+"checkpoint_best_model_on_Test_epoch_{}".format(state["best_epoch"]))            
                
                shutil.copyfile(checkpoint_best_model_checkpoint_path, checkpoint_dir+"checkpoint_best_model_checkpoint_epoch_{}".format(best_epoch_checkpoint))
                
                shutil.copyfile("{}_training_metrics.csv".format(unique_name), checkpoint_dir+model.model_name+"_training_metrics_at_epoch_{}.csv".format(epoch))
                shutil.copyfile("{}_test_metrics.csv".format(unique_name), checkpoint_dir+model.model_name+"_test_metrics_at_epoch_{}.csv".format(epoch))
                
            ########################################################################################################################################################
    
    #LOAD BEST MODEL AND COMPUTE LOSS FOR ALL SETS
    print("TESTING")
    # Load best model based on validation loss
    state = load_model(model, None,  checkpoint_path+'_best_model_on_Test', args.use_cuda,device,scheduler)
    #compute loss on all set_output_size
    train_loss = evaluate(model, device, criterion_sed, criterion_doa, tr_data)
    val_loss = evaluate(model, device, criterion_sed, criterion_doa, val_data)
    test_loss = evaluate(model, device, criterion_sed, criterion_doa, test_data)

    #PRINT AND SAVE RESULTS
    results = {'train_loss': train_loss.cpu().detach().numpy(),
               'val_loss': val_loss.cpu().detach().numpy(),
               'test_loss': test_loss.cpu().detach().numpy(),
               'train_loss_hist': train_loss_hist,
               'val_loss_hist': val_loss_hist}

    print(model.model_name)
    print ('RESULTS')
    for i in results:
        if 'hist' not in i:
            print (i, results[i])
    out_path = os.path.join(args.results_path, 'results_dict.json')
    np.save(out_path, results)
    print('*********** TEST BEST MODEL (epoch {}) ************'.format(state['best_test_epoch']))
    test_best_results=evaluate_test(model,device, test_data,epoch=state['best_test_epoch'],max_loc_value=args.max_loc_value,num_frames=args.num_frames,spatial_threshold=args.spatial_threshold)
                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #saving/loading parameters
    parser.add_argument('--results_path', type=str, default='RESULTS/Task2',
                        help='Folder to write results dicts into')
    parser.add_argument('--checkpoint_dir', type=str, default='RESULTS/Task2',
                        help='Folder to write checkpoints into')
    parser.add_argument('--load_model', type=str, default=None,#'RESULTS/Task2/checkpoint',
                        help='Reload a previously trained model (whole task model)')
    #dataset parameters
    parser.add_argument('--training_predictors_path', type=str,default='/var/datasets/L3DAS21/processed/task2_predictors_train.pkl')
    parser.add_argument('--training_target_path', type=str,default='/var/datasets/L3DAS21/processed/task2_target_train.pkl')
    parser.add_argument('--validation_predictors_path', type=str, default='/var/datasets/L3DAS21/processed/task2_predictors_validation.pkl')
    parser.add_argument('--validation_target_path', type=str, default='/var/datasets/L3DAS21/processed/task2_target_validation.pkl')
    parser.add_argument('--test_predictors_path', type=str, default='/var/datasets/L3DAS21/processed/task2_predictors_test.pkl')
    parser.add_argument('--test_target_path', type=str, default='/var/datasets/L3DAS21/processed/task2_target_test.pkl')
    #training parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--early_stopping', type=str, default='True')
    parser.add_argument('--fixed_seed', type=str, default='True')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size")
    parser.add_argument('--sr', type=int, default=32000,
                        help="Sampling rate")
    parser.add_argument('--patience', type=int, default=250,
                        help="Patience for early stopping on validation set")

    #model parameters
    #the following parameters produce a prediction for each 100-msecs frame
    parser.add_argument('--architecture', type=str, default='DualQSELD-TCN',
                        help="model's architecture, can be seldnet_vanilla or seldnet_augmented")
    parser.add_argument('--input_channels', type=int, default=4,
                        help="4/8 for 1/2 mics, multiply x2 if using also phase information")
    parser.add_argument('--n_mics', type=int, default=1)
    parser.add_argument('--phase', type=str, default='False')
    parser.add_argument('--class_overlaps', type=int, default=3,
                        help= 'max number of simultaneous sounds of the same class')
    parser.add_argument('--time_dim', type=int, default=4800)
    parser.add_argument('--freq_dim', type=int, default=256)
    parser.add_argument('--output_classes', type=int, default=14)
    parser.add_argument('--pool_size', type=str, default='[[8,2],[8,2],[2,2],[1,1]]')
    parser.add_argument('--cnn_filters', type=str, default='[64,64,64]')
    parser.add_argument('--pool_time', type=str, default='True')
    parser.add_argument('--dropout_perc', type=float, default=0.3)
    parser.add_argument('--D', type=str, default='[10]')
    parser.add_argument('--G', type=int, default=128)
    parser.add_argument('--U', type=int, default=128)
    parser.add_argument('--V', type=str, default='[128,128]')
    parser.add_argument('--spatial_dropout_rate', type=float, default=0.5)
    parser.add_argument('--batch_norm', type=str, default='BN')
    parser.add_argument('--dilation_mode', type=str, default='fibonacci')
    parser.add_argument('--model_extra_name', type=str, default='')
    parser.add_argument('--test_mode', type=str, default='test_best')
    parser.add_argument('--use_lr_scheduler', type=str, default='True')
    parser.add_argument('--lr_scheduler_step_size', type=int, default=150)
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=0.000005) 
    parser.add_argument('--dataset_normalization', type=str, default='True') 
    parser.add_argument('--kernel_size_cnn_blocks', type=int, default=3) 
    parser.add_argument('--kernel_size_dilated_conv', type=int, default=3) 
    parser.add_argument('--use_tcn', type=str, default='True') 
    parser.add_argument('--use_bias_conv', type=str, default='True') 
    parser.add_argument('--use_bias_linear', type=str, default='True') 
    parser.add_argument('--verbose', type=str, default='False')
    parser.add_argument('--sed_loss_weight', type=float, default=1.)
    parser.add_argument('--doa_loss_weight', type=float, default=5.)
    parser.add_argument('--domain_classifier', type=str, default='same') 
    parser.add_argument('--domain', type=str, default='DQ') 
    parser.add_argument('--fc_activations', type=str, default='Linear') 
    parser.add_argument('--fc_dropout', type=str, default='Last') 
    parser.add_argument('--fc_layers', type=str, default='[128]') 
    parser.add_argument('--V_kernel_size', type=int, default=3) 
    parser.add_argument('--use_time_distributed', type=str, default='False') 
    parser.add_argument('--parallel_ConvTC_block', type=str, default='False') 

    '''parser.add_argument('--wandb_id', type=str, default='none')
    parser.add_argument('--wandb_project', type=str, default='')
    parser.add_argument('--wandb_entity', type=str, default='')'''
    ############## TEST  ###################
    parser.add_argument('--max_loc_value', type=float, default=2.,
                         help='max value of target loc labels (to rescale model\'s output since the models has tanh in the output loc layer)')
    parser.add_argument('--num_frames', type=int, default=600,
                        help='total number of time frames in the predicted seld matrices. (600 for 1-minute sounds with 100msecs frames)')
    parser.add_argument('--spatial_threshold', type=float, default=2.,
                        help='max cartesian distance withn consider a true positive')
    ########################################

    ######################### CHECKPOINT ####################################################
    parser.add_argument('--checkpoint_step', type=int, default=100,
                        help="Save and test models every checkpoint_step epochs")
    parser.add_argument('--test_step', type=int, default=10,
                        help="Save and test models every checkpoint_step epochs")
    parser.add_argument('--min_n_epochs', type=int, default=1000,
                        help="Save and test models every checkpoint_step epochs")
    parser.add_argument('--Dcase21_metrics_DOA_threshold', type=int, default=20) 
    parser.add_argument('--parallel_magphase', type=str, default='False') 

    parser.add_argument('--TextArgs', type=str, default='config/Test.txt', help='Path to text with training settings')#'config/PHC-SELD-TCN-S1_BN.txt'
    parse_list = readFile(parser.parse_args().TextArgs)
    args = parser.parse_args(parse_list)
    
    #eval string bools and lists
    args.use_cuda = eval(args.use_cuda)
    args.early_stopping = eval(args.early_stopping)
    args.fixed_seed = eval(args.fixed_seed)
    args.pool_size= eval(args.pool_size)
    args.cnn_filters = eval(args.cnn_filters)
    args.verbose = eval(args.verbose)
    args.D=eval(args.D)
    args.V=eval(args.V)
    args.use_lr_scheduler=eval(args.use_lr_scheduler)
    #args.dataset_normalization=eval(args.dataset_normalization)
    args.phase=eval(args.phase)
    args.use_tcn=eval(args.use_tcn)
    args.use_bias_conv=eval(args.use_bias_conv)
    args.use_bias_linear=eval(args.use_bias_linear)
    args.fc_layers = eval(args.fc_layers)
    args.parallel_magphase = eval(args.parallel_magphase)

    main(args)
