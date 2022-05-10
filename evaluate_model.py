import sys, os
import pickle
import argparse
from matplotlib.image import pil_to_array
from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.utils.data as utils
from metrics import location_sensitive_detection
from models.SELD_Model import SELD_Model
from utility_functions import load_model, save_model, gen_submission_list_task2,save_array_to_csv
from torchinfo import summary
from Dcase21_metrics import *
'''
Load pretrained model and compute the metrics for Task 2
of the L3DAS21 challenge. The metric is F score computed with the
location sensitive detection: https://ieeexplore.ieee.org/document/8937220.
Command line arguments define the model parameters, the dataset to use and
where to save the obtained results.
'''

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

def main(args):
    
    model_path='RESULTS/Task2/{}/checkpoint'.format(args.architecture)#########

    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'
    

    print ('\nLoading dataset')
    #LOAD DATASET
    with open(args.predictors_path, 'rb') as f:
        predictors = pickle.load(f)
    with open(args.target_path, 'rb') as f:
        target = pickle.load(f)


    phase_string='_Phase' if args.phase else ''
    dataset_string='L3DAS21_'+str(args.n_mics)+'Mics_Magnidute'+phase_string+'_'+str(args.input_channels)+'Ch'
    #####################################NORMALIZATION####################################
    if args.dataset_normalization not in {'False','false','None','none'}:
        print('\nDataset_Normalization')
        if args.dataset_normalization in{'DQ_Normalization','UnitNormNormalization','UnitNorm'}:
            predictors = torch.tensor(predictors)
            target = torch.tensor(target)
            if args.n_mics==2:
                if args.domain in ['DQ','dq','dQ','Dual_Quaternion','dual_quaternion']:
                    dataset_string+=' Dataset Normalization for 2Mic 8Ch Magnitude Dual Quaternion UnitNorm'
                    print('Dataset Normalization for 2Mic 8Ch Magnitude Dual Quaternion UnitNorm')
                    ## TEST PREDICTORS ##
                    q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3 = torch.chunk(predictors[:,:8,:,:], chunks=8, dim=1)
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

                    predictors[:,:8,:,:] = torch.cat([q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3], dim=1) 
                    if args.phase:
                        raise ValueError('DATASET NORMALIZATION FOR PHASE DUAL QUATERNION NOT YET IMPLEMENTED')
                        print('Dataset Normalization for 2Mic 16Ch Magnitude-Phase Dual Quaternion ')
                    predictors = np.array(predictors)
                    target = np.array(target)
                    print ('\nShapes:')
                    print ('Test predictors: ', predictors.shape)
                    print ('Test target: ',target.shape)
        else:
            predictors = np.array(predictors)
            target = np.array(target)
            print ('\nShapes:')
            print ('Test predictors: ', predictors.shape)
            print ('Test target: ', target.shape)
            if args.n_mics==1:
                dataset_string+=' Dataset Normalization for 1Mic 4Ch Magnitude'
                print('Dataset Normalization for 1Mic 4Ch Magnitude')
                # Normalize test predictors with mean 0 and std 1
                test_mag_min = np.mean(predictors[:,:4,:,:])
                test_mag_std = np.std(predictors[:,:4,:,:])    
                predictors[:,:4,:,:] -= test_mag_min
                predictors[:,:4,:,:] /= test_mag_std
                if args.phase:
                    dataset_string+=' Dataset Normalization for 1Mic 8Ch Magnitude-Phase'
                    print('Dataset Normalization for 1Mic 8Ch Magnitude-Phase')
                    test_phase_min = np.mean(predictors[:,4:,:,:])
                    test_phase_std = np.std(predictors[:,4:,:,:])
                    predictors[:,4:,:,:] -= test_phase_min
                    predictors[:,4:,:,:] /= test_phase_std
            if args.n_mics==2:
                dataset_string+=' Dataset Normalization for 2Mic 8Ch Magnitude'
                print('Dataset Normalization for 2Mic 8Ch Magnitude')
                # Normalize test predictors with mean 0 and std 1
                test_mag_min = np.mean(predictors[:,:8,:,:])
                test_mag_std = np.std(predictors[:,:8,:,:])    
                predictors[:,:8,:,:] -= test_mag_min
                predictors[:,:8,:,:] /= test_mag_std
                if args.phase:
                    dataset_string+=' Dataset Normalization for 2Mic 16Ch Magnitude-Phase'
                    print('Dataset Normalization for 2Mic 16Ch Magnitude-Phase')
                    test_phase_min = np.mean(predictors[:,8:,:,:])
                    test_phase_std = np.std(predictors[:,8:,:,:])
                    predictors[:,8:,:,:] -= test_phase_min
                    predictors[:,8:,:,:] /= test_phase_std
    else:
        predictors = np.array(predictors)
        target = np.array(target)
        print ('\nShapes:')
        print ('Test predictors: ', predictors.shape)
        print ('Test target: ', target.shape)
    
    #convert to tensor
    predictors = torch.tensor(predictors).float()
    target = torch.tensor(target).float()
    #build dataset from tensors
    dataset_ = utils.TensorDataset(predictors, target)
    #build data loader from dataset
    dataloader = utils.DataLoader(dataset_, 1, shuffle=False, pin_memory=True)

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    #LOAD MODEL
    n_time_frames = predictors.shape[-1]

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
    args.load_model=model_dir+'checkpoint_best_model_on_Test'
    unique_name=model_dir+model.model_name
    print(model.model_name)
    #summary(model, input_size=(args.batch_size,args.input_channels,args.freq_dim,n_time_frames)) ##################################################
    
    if args.use_cuda:
        print("Moving model to gpu")
        model = model.to(device)

    #load checkpoint
    if args.load_model is not None and os.path.isfile(args.load_model) :####################################### added "and os.path.isfile(args.load_model)"
        print("Loading Model")
        state = load_model(model, None, args.load_model, args.use_cuda,device,None)
        
    #COMPUTING METRICS
    print("COMPUTING TASK 2 METRICS")
    TP = 0
    FP = 0
    FN = 0
    output_classes=args.output_classes
    class_overlaps=args.class_overlaps

    count = 0
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
                                                   max_overlaps=args.class_overlaps,
                                                   max_loc_value=args.max_loc_value)

            target,target_dict = gen_submission_list_task2(sed_target, doa_target,
                                               max_overlaps=args.class_overlaps,
                                               max_loc_value=args.max_loc_value)


            pred_labels =segment_labels(prediction_dict, args.num_frames)
            ref_labels =segment_labels(target_dict,  args.num_frames)
            eval_metrics.update_seld_scores(pred_labels, ref_labels)
            
            
            tp, fp, fn, _ = location_sensitive_detection(prediction, target, args.num_frames,
                                                      args.spatial_threshold, False)

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
    ER_score = (max(Nref, Nsys) - TP) / (Nref + 0.0)################ from evaluation_metrics.py SELDnet
    
    ER_dcase21, F_dcase21, LE_dcase21, LR_dcase21 = eval_metrics.compute_seld_scores()

    #SELD_dcase21 = np.mean([ER_dcase21,1 -  F_dcase21, LE_dcase21/180,1 - LR_dcase21])
    SELD_L3DAS21_LRLE = np.mean([ER_score,1 -  F_score, LE_dcase21/180,1 - LR_dcase21])
    CSL_score= np.mean([LE_dcase21/180,1 - LR_dcase21])
    LSD_score=np.mean([1-F_score,ER_score])
    

    #visualize and save results
    results = {'precision': precision,
               'recall': recall,
               'F score': F_score,
               'ER score': ER_score,
               'LE': LE_dcase21,
               'LR': LR_dcase21,
               'CSL score': CSL_score,
               'LSD score': LSD_score,
               'Global SELD score': SELD_L3DAS21_LRLE
               }
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
    print ()
    
    out_path = os.path.join(args.results_path, 'task2_metrics_dict.json')
    np.save(out_path, results)


    
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

