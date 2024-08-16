import utils
import algo
import numpy as np
import pandas as pd
import os
import pickle


def pipe_att_or_unatt_LVO(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, TRAIN_WITH_ATT, eeg_ori_list=None, dim_list_EEG=None, dim_list_Stim=None, n_components=3, saccade_multisubj_list=None, V_eeg=None, V_Stim=None, PLOT=False, figure_dir=None, SAVERES=False, table_dir=None, OVERWRITE=False, feat_name='ObjFlow', COMBINE_ATT_UNATT=False):
    '''
    TASK: Perform CCA analysis for attended or unattended features and save forward models and correlations

    Inputs that need further explanation:
    TRAIN_WITH_ATT: True if training with attended features, False if training with unattended features
    eeg_ori_list: Original EEG data, necessary for calculating the forward model when the input eeg_multisubj_list is already hankelized (e.g., due to spatial-temporal regression)
    dim_list_EEG: If the input eeg_multisubj_list is actually a stack of EEG and other modalities, dim_list_EEG is a list of the dimensions of each modality. E.g., [64, 4] for stacked EEG and EOG. Always put EEG at the first place.
    dim_list_Stim: Similar to dim_list_EEG, but for the dimensions of the stimulus features.
    saccade_multisubj_list: Saccade data. A mask will be created if saccade data is provided to exclude the time points around saccades.
    V_eeg, V_Stim: If want to use pretrained filters trained from single object data, provide the filters here.
    PLOT: True if want to plot the forward models [only applicable for EEG], False otherwise
    SAVERES: True if want to save the results in a table, False otherwise
    OVERWRITE: True if want to overwrite the existing results in the table, False otherwise
    '''
    eeg_onesubj_list = [eeg[:,:,Subj_ID] for eeg in eeg_multisubj_list]
    eeg_ori_onesubj_list = [eeg_ori[:,:,Subj_ID] for eeg_ori in eeg_ori_list] if eeg_ori_list is not None else None 
    feat_att_list = [feat_att[:,:,Subj_ID] for feat_att in feat_att_list] if np.ndim(feat_att_list[0]) == 3 else feat_att_list
    feat_unatt_list = [feat_unatt[:,:,Subj_ID] for feat_unatt in feat_unatt_list] if np.ndim(feat_unatt_list[0]) == 3 else feat_unatt_list
    if saccade_multisubj_list is not None:
        saccade_onesubj_list = [saccade[:,:,Subj_ID] for saccade in saccade_multisubj_list]
        mask_list = utils.get_mask_list(saccade_onesubj_list, before=10, after=20)
    else:
        mask_list = None
    CCA = algo.CanonicalCorrelationAnalysis(eeg_onesubj_list, feat_att_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, dim_list_EEG=dim_list_EEG, dim_list_Stim=dim_list_Stim, n_components=n_components, mask_list=mask_list)
    corr_att_fold, corr_unatt_fold, sig_corr_fold, sig_corr_pool, forward_model_fold = CCA.att_or_unatt_LVO(feat_unatt_list, TRAIN_WITH_ATT, V_eeg=V_eeg, V_Stim=V_Stim, EEG_ori_list=eeg_ori_onesubj_list, COMBINE_ATT_UNATT=COMBINE_ATT_UNATT)
    train_type = 'SO' if V_eeg is not None else 'Att' if TRAIN_WITH_ATT else 'Unatt'
    ifmask = True if mask_list is not None else False
    if PLOT:
        figure_name = f"{figure_dir}{feat_name}_Subj_{Subj_ID+1}_Train_{train_type}_Mask_{ifmask}_Folds.png"
        # if FM_org is not None:
        #     forward_model_fold = [utils.F_organize(forward_model, FM_org[0], FM_org[1]) for forward_model in forward_model_fold]
        utils.plot_spatial_resp_fold(forward_model_fold, corr_att_fold, corr_unatt_fold, sig_corr_fold, figure_name, AVG=False)
        utils.plot_spatial_resp_fold(forward_model_fold, corr_att_fold, corr_unatt_fold, sig_corr_pool, figure_name, AVG=True)
    if SAVERES:
        table_name = table_dir + f'{feat_name}_Corr_Train_{train_type}_Mask_{ifmask}.csv'
        utils.save_corr_df(table_name, sig_corr_pool, corr_att_fold, corr_unatt_fold, Subj_ID, OVERWRITE)


def pipe_compete_trials_LVO(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dir, dim_list_EEG=None, dim_list_Stim=None, saccade_multisubj_list=None, BOOTSTRAP=True, V_eeg=None, V_Stim=None, n_components=3, nb_comp_into_account=2, signifi_level=False, message=True, OVERWRITE=False, feat_name='ObjFlow', COMBINE_ATT_UNATT=False):
    '''
    TASK: Determine the attended feature from the unattended feature using CCA and evaluate the performance

    Inputs that need further explanation:
    TRAIN_WITH_ATT: True if training with attended features, False if training with unattended features
    eeg_ori_list: Original EEG data, necessary for calculating the forward model when the input eeg_multisubj_list is already hankelized (e.g., due to spatial-temporal regression)
    dim_list_EEG: If the input eeg_multisubj_list is actually a stack of EEG and other modalities, dim_list_EEG is a list of the dimensions of each modality. E.g., [64, 4] for stacked EEG and EOG. Always put EEG at the first place.
    dim_list_Stim: Similar to dim_list_EEG, but for the dimensions of the stimulus features.
    saccade_multisubj_list: Saccade data. A mask will be created if saccade data is provided to exclude the time points around saccades.
    BOOTSTRAP: True if selecting trials with given length randomly (wiith overlap), False if dividing the trials without overlap
    V_eeg, V_Stim: If want to use pretrained filters trained from single object data, provide the filters here.
    nb_comp_into_account: Number of components into account when calculating the accuracy
    OVERWRITE: True if want to overwrite the existing results in the table, False otherwise
    '''
    res = []
    eeg_onesubj_list = [eeg[:,:,Subj_ID] for eeg in eeg_multisubj_list]
    feat_att_list = [feat_att[:,:,Subj_ID] for feat_att in feat_att_list] if np.ndim(feat_att_list[0]) == 3 else feat_att_list
    feat_unatt_list = [feat_unatt[:,:,Subj_ID] for feat_unatt in feat_unatt_list] if np.ndim(feat_unatt_list[0]) == 3 else feat_unatt_list
    if saccade_multisubj_list is not None:
        saccade_onesubj_list = [saccade[:,:,Subj_ID] for saccade in saccade_multisubj_list]
        mask_list = utils.get_mask_list(saccade_onesubj_list, before=10, after=20)
    else:
        mask_list = None
    ifmask = True if mask_list is not None else False
    CCA = algo.CanonicalCorrelationAnalysis(eeg_onesubj_list, feat_att_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, dim_list_EEG=dim_list_EEG, dim_list_Stim=dim_list_Stim, n_components=n_components, mask_list=mask_list, signifi_level=signifi_level, message=message)
    for trial_len in trial_len_list:
        print('Trial length: ', trial_len)
        corr_att_eeg, corr_unatt_eeg, corr_att_unatt = CCA.att_or_unatt_LVO_trials(feat_unatt_list, trial_len=trial_len, BOOTSTRAP=BOOTSTRAP, V_eeg=V_eeg, V_Stim=V_Stim, COMBINE_ATT_UNATT=COMBINE_ATT_UNATT)
        acc, _, _, _, _= utils.eval_compete(corr_att_eeg, corr_unatt_eeg, TRAIN_WITH_ATT=True, nb_comp_into_account=nb_comp_into_account)
        res.append(acc)
        if trial_len == 30:
            # save corr_att_eeg, corr_unatt_eeg, and corr_att_unatt in a dictionary
            corr_dict = {'EEG_Att': corr_att_eeg, 'EEG_Unatt': corr_unatt_eeg, 'Att_Unatt': corr_att_unatt}
            file_name = table_dir + f'{Subj_ID}_VAD_Corrdict_Len_{trial_len}.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(corr_dict, f)
    train_type = 'SO' if V_eeg is not None else 'Att'
    table_name = table_dir + f'{feat_name}_Acc_Train_{train_type}_Mask_{ifmask}.csv'
    utils.save_acc_df(table_name, Subj_ID, trial_len_list, res, OVERWRITE)


def pipe_discriminative_trials_LVO(Subj_ID, target_list, signal_list, compete_list, fs, para_signal, para_compete, para_target, trial_len_list, table_dir, BOOTSTRAP=True, n_components=3, nb_comp_into_account=1, OVERWRITE=False, feat_name='ObjFlow'):
    '''
    TASK: Determine the attended feature from the unattended feature using CCA and evaluate the performance

    Inputs that need further explanation:
    TRAIN_WITH_ATT: True if training with attended features, False if training with unattended features
    eeg_ori_list: Original EEG data, necessary for calculating the forward model when the input eeg_multisubj_list is already hankelized (e.g., due to spatial-temporal regression)
    dim_list_EEG: If the input eeg_multisubj_list is actually a stack of EEG and other modalities, dim_list_EEG is a list of the dimensions of each modality. E.g., [64, 4] for stacked EEG and EOG. Always put EEG at the first place.
    dim_list_Stim: Similar to dim_list_EEG, but for the dimensions of the stimulus features.
    saccade_multisubj_list: Saccade data. A mask will be created if saccade data is provided to exclude the time points around saccades.
    BOOTSTRAP: True if selecting trials with given length randomly (wiith overlap), False if dividing the trials without overlap
    V_eeg, V_Stim: If want to use pretrained filters trained from single object data, provide the filters here.
    nb_comp_into_account: Number of components into account when calculating the accuracy
    OVERWRITE: True if want to overwrite the existing results in the table, False otherwise
    '''
    res = []
    target_list = [target[:,:,Subj_ID] for target in target_list] if np.ndim(target_list[0]) == 3 else target_list
    signal_list = [signal[:,:,Subj_ID] for signal in signal_list] if np.ndim(signal_list[0]) == 3 else signal_list
    compete_list = [compete[:,:,Subj_ID] for compete in compete_list] if np.ndim(compete_list[0]) == 3 else compete_list
    DCCA = algo.DiscriminativeCCA(signal_list, compete_list, target_list, fs, para_signal, para_compete, para_target, n_components=n_components, regularization=None)
    for trial_len in trial_len_list:
        print('Trial length: ', trial_len)
        corr_signal, corr_compete = DCCA.att_or_unatt_trials(trial_len=trial_len, BOOTSTRAP=BOOTSTRAP)
        acc, _, _, _, _= utils.eval_compete(corr_signal, corr_compete, TRAIN_WITH_ATT=True, nb_comp_into_account=nb_comp_into_account)
        res.append(acc)
    table_name = table_dir + f'{feat_name}_Acc_Disc.csv'
    utils.save_acc_df(table_name, Subj_ID, trial_len_list, res, OVERWRITE)


def pipe_mm_trials_LVO(Subj_ID, eeg_multisubj_list, feat_match_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dir, MATCHATT, dim_list_EEG=None, dim_list_Stim=None, saccade_multisubj_list=None, V_eeg=None, V_Stim=None, n_components=3, nb_comp_into_account=2, signifi_level=False, message=True, OVERWRITE=False, SINGLEOBJ=False, feat_name='ObjFlow'):
    '''
    TASK: Determine the matched feature from a random feature sampled from a different time point using CCA and evaluate the performance
    '''
    res = []
    eeg_onesubj_list = [eeg[:,:,Subj_ID] for eeg in eeg_multisubj_list]
    feat_match_list = [feat_match[:,:,Subj_ID] for feat_match in feat_match_list] if np.ndim(feat_match_list[0]) == 3 else feat_match_list
    if saccade_multisubj_list is not None:
        saccade_onesubj_list = [saccade[:,:,Subj_ID] for saccade in saccade_multisubj_list]
        mask_list = utils.get_mask_list(saccade_onesubj_list, before=10, after=20)
    else:
        mask_list = None
    ifmask = True if mask_list is not None else False
    CCA = algo.CanonicalCorrelationAnalysis(eeg_onesubj_list, feat_match_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, dim_list_EEG=dim_list_EEG, dim_list_Stim=dim_list_Stim, n_components=n_components, mask_list=mask_list, signifi_level=signifi_level, message=message)
    for trial_len in trial_len_list:
        print('Trial length: ', trial_len)
        corr_match_eeg, corr_mismatch_eeg, corr_match_mm = CCA.match_mismatch_LVO(trial_len=trial_len, V_eeg=V_eeg, V_Stim=V_Stim)
        acc, _, _ = utils.eval_mm(corr_match_eeg, corr_mismatch_eeg, nb_comp_into_account)
        res.append(acc)
        if trial_len == 30:
            # save corr_att_eeg, corr_unatt_eeg, and corr_att_unatt in a dictionary
            corr_dict = {'EEG_Match': corr_match_eeg, 'EEG_MM': corr_mismatch_eeg, 'Match_MM': corr_match_mm}
            file_name = table_dir + f'{Subj_ID}_MM_Corrdict_Len_{trial_len}.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(corr_dict, f)
    if SINGLEOBJ:
        table_name = table_dir + f'{feat_name}_Acc_MM_SO_Mask_{ifmask}.csv'
    else:
        train_type = 'SO' if V_eeg is not None else 'Att' if MATCHATT else 'Unatt'
        table_name = table_dir + f'{feat_name}_Acc_MM_Train_{train_type}_Mask_{ifmask}.csv'
    utils.save_acc_df(table_name, Subj_ID, trial_len_list, res, OVERWRITE)


def pipe_vad_mm(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dir, dim_list_EEG=None, dim_list_Stim=None, saccade_multisubj_list=None, BOOTSTRAP=True, V_eeg=None, V_Stim=None, n_components=3, nb_comp_into_account=2, signifi_level=False, message=True, OVERWRITE=False, feat_name='ObjFlow', COMBINE_ATT_UNATT=False):
    res_vad = []
    res_mm = []
    eeg_onesubj_list = [eeg[:,:,Subj_ID] for eeg in eeg_multisubj_list]
    feat_att_list = [feat_att[:,:,Subj_ID] for feat_att in feat_att_list] if np.ndim(feat_att_list[0]) == 3 else feat_att_list
    feat_unatt_list = [feat_unatt[:,:,Subj_ID] for feat_unatt in feat_unatt_list] if np.ndim(feat_unatt_list[0]) == 3 else feat_unatt_list
    if saccade_multisubj_list is not None:
        saccade_onesubj_list = [saccade[:,:,Subj_ID] for saccade in saccade_multisubj_list]
        mask_list = utils.get_mask_list(saccade_onesubj_list, before=10, after=20)
    else:
        mask_list = None
    ifmask = True if mask_list is not None else False
    CCA = algo.CanonicalCorrelationAnalysis(eeg_onesubj_list, feat_att_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, dim_list_EEG=dim_list_EEG, dim_list_Stim=dim_list_Stim, n_components=n_components, mask_list=mask_list, signifi_level=signifi_level, message=message)
    for trial_len in trial_len_list:
        print('Trial length: ', trial_len)
        corr_att_eeg, corr_unatt_eeg, corr_mismatch_eeg, corr_att_unatt, corr_att_mismatch = CCA.VAD_MM_LVO(feat_unatt_list, trial_len, V_eeg=V_eeg, V_Stim=V_Stim)
        acc_vad, _, _, _, _= utils.eval_compete(corr_att_eeg, corr_unatt_eeg, TRAIN_WITH_ATT=True, nb_comp_into_account=nb_comp_into_account)
        acc_mm, _, _ = utils.eval_mm(corr_att_eeg, corr_mismatch_eeg, nb_comp_into_account)
        res_vad.append(acc_vad)
        res_mm.append(acc_mm)
        corr_dict = {'EEG_Att': corr_att_eeg, 'EEG_Unatt': corr_unatt_eeg, 'EEG_MM':corr_mismatch_eeg, 'Att_Unatt': corr_att_unatt, 'Att_MM': corr_att_mismatch}
        file_name = table_dir + f'{Subj_ID}_VAD_MM_Corrdict_Len_{trial_len}.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(corr_dict, f)
    train_type = 'SO' if V_eeg is not None else 'Att'
    table_name = table_dir + f'{feat_name}_Acc_Train_{train_type}_Mask_{ifmask}_CB.csv'
    utils.save_acc_df(table_name, Subj_ID, trial_len_list, res_vad, OVERWRITE)
    table_name = table_dir + f'{feat_name}_Acc_MM_Train_{train_type}_Mask_{ifmask}_CB.csv'
    utils.save_acc_df(table_name, Subj_ID, trial_len_list, res_mm, OVERWRITE)

def pipe_single_obj(Subj_ID, eeg_multisubj_list, feat_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, figure_dir, table_dir, dim_list_EEG=None, dim_list_Stim=None, saccade_multisubj_list=None, n_components=3, PLOT=False, OVERWRITE=False, feat_name='ObjFlow'):
    '''
    TASK: Train and test on the single-object data. Then use all single-object data to train the filters to be used in the overlaid-object data.
    '''
    eeg_onesubj_list = [eeg[:,:,Subj_ID] for eeg in eeg_multisubj_list]
    feat_list = [feat[:,:,Subj_ID] for feat in feat_list] if np.ndim(feat_list[0]) == 3 else feat_list
    if saccade_multisubj_list is not None:
        saccade_onesubj_list = [saccade[:,:,Subj_ID] for saccade in saccade_multisubj_list]
        mask_list = utils.get_mask_list(saccade_onesubj_list, before=10, after=20)
    else:
        mask_list = None
    ifmask = True if mask_list is not None else False
    CCA = algo.CanonicalCorrelationAnalysis(eeg_onesubj_list, feat_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, dim_list_EEG=dim_list_EEG, dim_list_Stim=dim_list_Stim, n_components=n_components, mask_list=mask_list)
    corr_train_fold, corr_test_fold, sig_corr_fold, sig_corr_pool, forward_model_fold = CCA.cross_val_LVO()
    if PLOT:
        figure_name = f"{figure_dir}{feat_name}_Subj_{Subj_ID+1}_Folds.png"
        utils.plot_spatial_resp_fold(forward_model_fold, corr_test_fold, None, sig_corr_fold, figure_name, AVG=False)
        utils.plot_spatial_resp_fold(forward_model_fold, corr_test_fold, None, sig_corr_pool, figure_name, AVG=True)
    table_name = table_dir + f'{feat_name}_Corr_SO_Mask_{ifmask}.csv'
    # check if the file exists
    if not os.path.isfile(table_name):
        # create a pandas dataframe that contains Subj_ID, Corr_Att, Corr_Unatt, Sig_Corr
        res_df = utils.create_corr_df(Subj_ID, sig_corr_pool, corr_train_fold, corr_test_fold)
        res_df.rename(columns={'Att': 'Train', 'Unatt': 'Test'}, inplace=True)
    else:
        # read the dataframe
        res_df = pd.read_csv(table_name, header=0, index_col=[0,1,2])
        if not 'Subj '+str(Subj_ID+1) in res_df.index.get_level_values('Subject ID'):
            res_add = utils.create_corr_df(Subj_ID, sig_corr_pool, corr_train_fold, corr_test_fold)
            res_add.rename(columns={'Att': 'Train', 'Unatt': 'Test'}, inplace=True)
            res_df = pd.concat([res_df, res_add], axis=0)
        elif OVERWRITE:
            res_df = res_df.drop('Subj '+str(Subj_ID+1), level='Subject ID')
            res_add = utils.create_corr_df(Subj_ID, sig_corr_pool, corr_train_fold, corr_test_fold)
            res_add.rename(columns={'Att': 'Train', 'Unatt': 'Test'}, inplace=True)
            res_df = pd.concat([res_df, res_add], axis=0)
        else:
            print(f"Results for Subj {Subj_ID+1} already exist in {table_name}")
    with open(table_name, 'w') as f:
        res_df.to_csv(f, header=True)
    EEG_all = np.concatenate(eeg_onesubj_list, axis=0)
    feat_all = np.concatenate(feat_list, axis=0)
    _, _, _, _, V_eeg_SO, V_stim_SO, _ = CCA.fit(EEG_all, feat_all)
    return V_eeg_SO, V_stim_SO


def pipe_att_or_unatt_aug(Subj_ID, nested_data, nested_aug_data, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list=None, BOOTSTRAP=True, n_components=3, nb_comp_into_account=2, table_dir=None, OVERWRITE=False, SYNMASK=False, feat_name='ObjFlow'):
    eeg_multisubj_list, feat_att_list, feat_unatt_list, saccade_multisubj_list = nested_data
    eeg_multisubj_list_aug, feat_att_list_aug, _, saccade_multisubj_list_aug = nested_aug_data
    eeg_onesubj_list = [eeg[:,:,Subj_ID] for eeg in eeg_multisubj_list]
    eeg_onesubj_list_aug = [eeg[:,:,Subj_ID] for eeg in eeg_multisubj_list_aug]
    feat_att_list = [feat_att[:,:,Subj_ID] for feat_att in feat_att_list] if np.ndim(feat_att_list[0]) == 3 else feat_att_list
    feat_unatt_list = [feat_unatt[:,:,Subj_ID] for feat_unatt in feat_unatt_list] if np.ndim(feat_unatt_list[0]) == 3 else feat_unatt_list
    feat_att_list_aug = [feat_att[:,:,Subj_ID] for feat_att in feat_att_list_aug] if np.ndim(feat_att_list_aug[0]) == 3 else feat_att_list_aug
    
    saccade_onesubj_list = [saccade[:,:,Subj_ID] for saccade in saccade_multisubj_list]
    saccade_onesubj_list_aug = [saccade[:,:,Subj_ID] for saccade in saccade_multisubj_list_aug]
    if SYNMASK:
        saccade_onesubj_list = utils.shuffle_datalist(saccade_onesubj_list, 1)
        saccade_onesubj_list_aug = utils.shuffle_datalist(saccade_onesubj_list_aug, 1)
    mask_list = utils.get_mask_list(saccade_onesubj_list, before=10, after=30)
    mask_list_aug = utils.get_mask_list(saccade_onesubj_list_aug, before=10, after=30)
    data_loss = utils.data_loss_due_to_mask(mask_list+mask_list_aug, L_EEG, offset_EEG)

    CCA = algo.CanonicalCorrelationAnalysis(eeg_onesubj_list, feat_att_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, n_components=n_components, mask_list=mask_list, leave_out=4)
    corr_att_fold, corr_unatt_fold, _, sig_corr_pool = CCA.att_or_unatt_aug(eeg_onesubj_list_aug, feat_att_list_aug, mask_list_aug, feat_unatt_list, trial_len=None)
    table_name = table_dir + f'{feat_name}_Corr_Train_Aug_Syn_{SYNMASK}.csv'
    utils.save_corr_df(table_name, sig_corr_pool, corr_att_fold, corr_unatt_fold, Subj_ID, OVERWRITE)
    
    res = []
    for trial_len in trial_len_list:
        print('Trial length: ', trial_len)
        corr_att_eeg, corr_unatt_eeg, _, _ = CCA.att_or_unatt_aug(eeg_onesubj_list_aug, feat_att_list_aug, mask_list_aug, feat_unatt_list, trial_len, BOOTSTRAP)
        acc, _, _, _, _= utils.eval_compete(corr_att_eeg, corr_unatt_eeg, TRAIN_WITH_ATT=True, nb_comp_into_account=nb_comp_into_account)
        res.append(acc)
    table_name = table_dir + f'{feat_name}_Acc_Train_Aug_Syn_{SYNMASK}.csv'
    utils.save_acc_df(table_name, Subj_ID, trial_len_list, res, OVERWRITE, data_loss)


def pipe_saccade(nested_data, nested_aug_data, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list=None, BOOTSTRAP=True, n_components=3, nb_comp_into_account=2, table_dir=None, OVERWRITE=False, SYNMASK=False, feat_name='ObjFlow'):
    eeg_multisubj_list, feat_att_list, feat_unatt_list, saccade_multisubj_list = nested_data
    eeg_multisubj_list_aug, feat_att_list_aug, _, saccade_multisubj_list_aug = nested_aug_data
    nb_subj = eeg_multisubj_list[0].shape[2]
    mask_list = None
    mask_list_aug = None
    data_loss = []
    for Subj_ID in range(nb_subj):
        saccade_onesubj_list = [saccade[:,:,Subj_ID] for saccade in saccade_multisubj_list]
        saccade_onesubj_list_aug = [saccade[:,:,Subj_ID] for saccade in saccade_multisubj_list_aug]
        if SYNMASK:
            saccade_onesubj_list = utils.shuffle_datalist(saccade_onesubj_list, 1)
            saccade_onesubj_list_aug = utils.shuffle_datalist(saccade_onesubj_list_aug, 1)
        mask_onesubj_list = utils.get_mask_list(saccade_onesubj_list, before=10, after=30, ThreeD=True)
        mask_onesubj_list_aug = utils.get_mask_list(saccade_onesubj_list_aug, before=10, after=30, ThreeD=True)
        if mask_list is None:
            mask_list = mask_onesubj_list
            mask_list_aug = mask_onesubj_list_aug
        else:
            mask_list = [np.concatenate((mask_prev_subjs, mask), axis=2) for mask_prev_subjs, mask in zip(mask_list, mask_onesubj_list)]
            mask_list_aug = [np.concatenate((mask_prev_subjs, mask), axis=2) for mask_prev_subjs, mask in zip(mask_list_aug, mask_onesubj_list_aug)]
        data_loss_onesubj = utils.data_loss_due_to_mask(mask_onesubj_list+mask_onesubj_list_aug, L_EEG, offset_EEG)
        data_loss.append(data_loss_onesubj)

    CCA = algo.CanonicalCorrelationAnalysis(eeg_multisubj_list, feat_att_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, n_components=n_components, mask_list=mask_list, leave_out=7)
    res = {Subj_ID: [] for Subj_ID in range(nb_subj)}
    table_name = table_dir + f'{feat_name}_Acc_Train_Aug_Multisubj_Syn_{SYNMASK}.csv'
    for trial_len in trial_len_list:
        print('Trial length: ', trial_len)
        corr_att_eeg, corr_unatt_eeg = CCA.att_or_unatt_aug_multisubj(eeg_multisubj_list_aug, feat_att_list_aug, mask_list_aug, feat_unatt_list, trial_len, BOOTSTRAP)
        for Subj_ID in range(nb_subj):
            acc, _, _, _, _= utils.eval_compete(corr_att_eeg[Subj_ID], corr_unatt_eeg[Subj_ID], TRAIN_WITH_ATT=True, nb_comp_into_account=nb_comp_into_account)
            res[Subj_ID].append(acc)
    for Subj_ID in range(nb_subj):
        utils.save_acc_df(table_name, Subj_ID, trial_len_list, res[Subj_ID], OVERWRITE, data_loss[Subj_ID])


def pipe_att_or_unatt_gcca(Subj_ID, nested_datalist, fs, L_list, offset_list, n_components=4, nb_comp_into_account=2, trial_len_list=[15,30,45,60], table_dir=None, OVERWRITE=False, feat_name='ObjFlow'):
    '''
    TASK: Perform GCCA analysis on given modalities (including stimulus features); Attended features and unattended features are discriminated based on ISC.
    nested_datalist: A list of lists of data for each modality. The last modality is the attended stimulus feature. E.g., [EEG_list, EOG_list, feat_att_list, feat_unatt_list]
    The last two lists are the attended and unattended stimulus features, respectively.
    '''
    data_mm_list = [[data[:,:,Subj_ID] for data in data_list] for data_list in nested_datalist[:-2]]
    feat_att_list, feat_unatt_list = nested_datalist[-2], nested_datalist[-1]
    feat_att_list = [feat_att[:,:,Subj_ID] for feat_att in feat_att_list] if np.ndim(feat_att_list[0]) == 3 else feat_att_list
    feat_unatt_list = [feat_unatt[:,:,Subj_ID] for feat_unatt in feat_unatt_list] if np.ndim(feat_unatt_list[0]) == 3 else feat_unatt_list
    GCCA_MM = algo.GeneralizedCCA_MultiMod(data_mm_list+[feat_att_list, feat_unatt_list], fs, L_list, offset_list, leave_out=2, n_components=n_components)
    res = []
    for trial_len in trial_len_list:
        print('Trial length: ', trial_len)
        isc_att, isc_unatt = GCCA_MM.att_or_unatt_LVO_trials(trial_len)
        acc, _, _, _, _= utils.eval_compete(isc_att, isc_unatt, TRAIN_WITH_ATT=True, nb_comp_into_account=nb_comp_into_account)
        res.append(acc)
    table_name = table_dir + f'{feat_name}_Acc_MM.csv'
    utils.save_acc_df(table_name, Subj_ID, trial_len_list, res, OVERWRITE)


def pipe_GCCA(nested_datalist, fs, L_list, offset_list, mod_name_list, W_list, nested_dimlist, figure_dir, table_dir, n_components=10, leave_out=2, SINGLEOBJ=False, OVERWRITE=False, FM_ORG=None):
    prefix = 'Single_' if SINGLEOBJ else 'Superimposed_'
    table_name = table_dir + f'{prefix}Folds.csv' if W_list[0] is None else table_dir + f'Folds_Pretrain_on_SO.csv'
    for datalist, L, offset, mod_name, W_data, dim_list in zip(nested_datalist, L_list, offset_list, mod_name_list, W_list, nested_dimlist):
        GCCA = algo.GeneralizedCCA(datalist, fs, L=L, offset=offset, dim_list=dim_list, n_components=n_components, signifi_level=True, leave_out=leave_out)
        _, corr_test_fold, _, cov_test_fold, _, _, sig_corr_fold, sig_corr_pool, forward_model_fold = GCCA.cross_val_LVO(W_eeg=W_data)
        if FM_ORG is not None:
            forward_model_fold = [utils.F_organize(forward_model, FM_ORG[0], FM_ORG[1]) for forward_model in forward_model_fold]
        if forward_model_fold[0].shape[0] == 64:
            figure_name = figure_dir + f"{prefix}{mod_name}_Folds.png" if W_data is None else figure_dir + f"{prefix}{mod_name}_Pretrain_on_SO_Folds.png"
            utils.plot_spatial_resp_fold(forward_model_fold, corr_test_fold, None, sig_corr_fold, figure_name, AVG=False)
            utils.plot_spatial_resp_fold(forward_model_fold, corr_test_fold, None, sig_corr_pool, figure_name, AVG=True)
        # check if the file exists
        if not os.path.isfile(table_name):
            # create a pandas dataframe that contains Subj_ID, Corr_Att, Corr_Unatt, Sig_Corr
            res_df = utils.create_ISC_df(corr_test_fold, cov_test_fold, sig_corr_pool, mod_name)
        else:
            # read the dataframe
            res_df = pd.read_csv(table_name, header=0, index_col=[0,1,2,3])
            if not mod_name in res_df.index.get_level_values('Modality'):
                res_add = utils.create_ISC_df(corr_test_fold, cov_test_fold, sig_corr_pool, mod_name)
                res_df = pd.concat([res_df, res_add], axis=0)
            elif OVERWRITE:
                res_df = res_df.drop(mod_name, level='Modality')
                res_add = utils.create_ISC_df(corr_test_fold, cov_test_fold, sig_corr_pool, mod_name)
                res_df = pd.concat([res_df, res_add], axis=0)
            else:
                print(f"Results for {mod_name} already exist in {table_name}")
        with open(table_name, 'w') as f:
            res_df.to_csv(f, header=True)


def pipe_GCCA_preprocessed(eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, para_gcca, para_cca_eeg, para_cca_stim, table_dir, figure_dir=None, n_components_GCCA=192, n_components_CCA=3, nb_comp_kept_grid=range(16, 200, 16), nb_comp_into_account=2, feat_name='ObjFlow', trial_len=30, OVERWRITE=False):
    nb_subj = eeg_multisubj_list[0].shape[2]
    GCCA_CCA = algo.GCCAPreprocessedCCA(0, eeg_multisubj_list, feat_att_list, fs, para_gcca, para_cca_eeg, para_cca_stim, leave_out=2, n_components_GCCA=n_components_GCCA, n_components_CCA=n_components_CCA)
    nb_comp_kept = GCCA_CCA.search_para(nb_comp_kept_grid)
    print(f'Number of components kept after GCCA: {nb_comp_kept}')
    acc_test_list = []
    sim_test_list = []
    for Subj_ID in range(nb_subj):
        GCCA_CCA.switch_subj(Subj_ID)
        corr_att_eeg_train, corr_att_eeg_test, corr_unatt_eeg_train, corr_unatt_eeg_test, sim_train, sim_test = GCCA_CCA.att_or_unatt_trials(feat_unatt_list, trial_len=trial_len)
        if figure_dir is not None:
            utils.plot_accuracy_percentage(corr_att_eeg_train, corr_unatt_eeg_train, np.sum(sim_train, axis=1), figure_dir, Subj_ID, nb_comp_into_account=nb_comp_into_account, name_prepend='Train_')
            utils.plot_accuracy_percentage(corr_att_eeg_test, corr_unatt_eeg_test, np.sum(sim_test, axis=1), figure_dir, Subj_ID, nb_comp_into_account=nb_comp_into_account)
        acc_test, _, _, _, _= utils.eval_compete(corr_att_eeg_test, corr_unatt_eeg_test, TRAIN_WITH_ATT=True, nb_comp_into_account=nb_comp_into_account)
        acc_test_list.append(acc_test)
        sim_test_list.append(np.median(np.sum(sim_test, axis=1)))
    
    # Convert the nested lists to NumPy arrays
    data_array = np.c_[np.array(acc_test_list), np.array(sim_test_list)]
    table_name = table_dir + f'{feat_name}_GCCA_Acc_len_{trial_len}.csv'
    if not os.path.isfile(table_name) or OVERWRITE:
        data_df = pd.DataFrame(data_array, index=['Subj '+str(i) for i in range(1, nb_subj+1)], columns=['Acc', 'Sim'])
        data_df.index.name = 'Subject ID'
        data_df.to_csv(table_name, header=True)
    else:
        print(f"Results already exist in {table_name}")