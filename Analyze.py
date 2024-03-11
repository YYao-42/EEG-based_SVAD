import numpy as np
import utils
import algo
import pickle

# Pipelines
#   1. pipe_att_or_unatt: Get the results of CCA when training with attended (or unattended) and testing with attended (or unattended) features [4 modes in total]
#   2. pipe_compete_trials: Discriminate the attended and unattended features under different trial lengths
#   3. pipe_mm_trials: Match the desired features under different trial lengths
#   4. pipe_single_obj: For single object case, get the results of CCA and match-mismatch under different trial lengths
    
def pipe_att_or_unatt(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, TRAIN_WITH_ATT, saccade_multisubj_list=None, nb_trials=None, V_eeg=None, V_Stim=None, figure_dir=None, trial_len=60, fold=5, n_components=5, nb_comp_into_account=2, PLOT=False, signifi_level=True, message=True):
    eeg_onesubj_list = [eeg[:,:,Subj_ID] for eeg in eeg_multisubj_list]
    if saccade_multisubj_list is not None:
        saccade_onesubj_list = [saccade[:,:,Subj_ID] for saccade in saccade_multisubj_list]
    else:
        saccade_onesubj_list = None
    CCA = algo.CanonicalCorrelationAnalysis(eeg_onesubj_list, feat_att_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, fold=fold, n_components=n_components, signifi_level=signifi_level, message=message)
    corr_att_fold, corr_unatt_fold, V_eeg_train, V_feat_train, sig_corr_att, sig_corr_unatt = CCA.att_or_unatt(feat_unatt_list, trial_len=trial_len, TRAIN_WITH_ATT=TRAIN_WITH_ATT, V_eeg=V_eeg, V_Stim=V_Stim, nb_trials=nb_trials, saccade_list=saccade_onesubj_list)
    acc, p_value, acc_sig, corr_att_cv, corr_unatt_cv= utils.eval_compete(corr_att_fold, corr_unatt_fold, TRAIN_WITH_ATT, nb_comp_into_account)
    if PLOT:
        # find the indices components with corr > sig_corr
        idx_sig_att = np.where(corr_att_cv > sig_corr_att)[0]
        idx_sig_unatt = np.where(corr_unatt_cv > sig_corr_unatt)[0]
        eeg_onesub = np.concatenate(tuple(eeg_onesubj_list), axis=0)
        forward_model = CCA.forward_model(eeg_onesub, V_eeg_train)
        train_type = 'SO' if V_eeg is not None else 'Att' if TRAIN_WITH_ATT else 'Unatt'
        if_saccade = 'Saccade' if saccade_multisubj_list is None else 'No_Saccade'
        figure_name_att = f"{figure_dir}Subj_{Subj_ID}_Trial_len_{trial_len}_Train_{train_type}_Test_Att_{if_saccade}.png"
        figure_name_unatt = f"{figure_dir}Subj_{Subj_ID}_Trial_len_{trial_len}_Train_{train_type}_Test_Unatt_{if_saccade}.png"
        utils.plot_spatial_resp(forward_model, corr_att_fold, figure_name_att, idx_sig=idx_sig_att)
        utils.plot_spatial_resp(forward_model, corr_unatt_fold, figure_name_unatt, idx_sig=idx_sig_unatt)
    return acc, p_value, acc_sig, corr_att_cv, corr_unatt_cv

def pipe_compete_trials(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dir, TRAIN_WITH_ATT, saccade_multisubj_list=None, nb_trials=200, V_eeg=None, V_Stim=None, fold=5, n_components=5, nb_comp_into_account=2):
    acc_list = []
    p_value_list = []
    acc_sig_list = []
    for trial_len in trial_len_list:
        print('Trial length: ', trial_len)
        acc, p_value, acc_sig, _, _ = pipe_att_or_unatt(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, TRAIN_WITH_ATT=TRAIN_WITH_ATT, saccade_multisubj_list=saccade_multisubj_list, nb_trials=nb_trials, V_eeg=V_eeg, V_Stim=V_Stim, trial_len=trial_len, fold=fold, n_components=n_components, nb_comp_into_account=nb_comp_into_account, signifi_level=False, message=False)
        acc_list.append(acc)
        p_value_list.append(p_value)
        acc_sig_list.append(acc_sig)
        # save the results
    results = {'acc': acc_list, 'pvalue': p_value_list, 'acc_sig': acc_sig_list, 'trial_len_list': trial_len_list}
    train_type = 'Att' if TRAIN_WITH_ATT else 'Unatt'
    if_saccade = 'Saccade' if saccade_multisubj_list is None else 'No_Saccade'
    table_name = f"{table_dir}Subj_{Subj_ID}_Compete_{train_type}_{if_saccade}.pkl"
    with open(table_name, 'wb') as f:
        pickle.dump(results, f)

def pipe_mm_trials(Subj_ID, eeg_multisubj_list, feat_match_list, feat_distract_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dir, MATCHATT, saccade_multisubj_list=None, nb_trials=200, V_eeg=None, V_Stim=None, fold=5, n_components=5, nb_comp_into_account=2):
    acc_list = []
    p_value_list = []
    acc_sig_list = []
    eeg_onesubj_list = [eeg[:,:,Subj_ID] for eeg in eeg_multisubj_list]
    if saccade_multisubj_list is not None:
        saccade_onesubj_list = [saccade[:,:,Subj_ID] for saccade in saccade_multisubj_list]
    else:
        saccade_onesubj_list = None
    for trial_len in trial_len_list:
        print('Trial length: ', trial_len)
        CCA = algo.CanonicalCorrelationAnalysis(eeg_onesubj_list, feat_match_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, fold=fold, n_components=n_components, signifi_level=False, message=False)
        corr_match_fold, corr_mismatch_fold = CCA.match_mismatch(trial_len, nb_trials, feat_distract_list, V_eeg=V_eeg, V_Stim=V_Stim, saccade_list=saccade_onesubj_list)
        acc, p_value, acc_sig = utils.eval_mm(corr_match_fold, corr_mismatch_fold, nb_comp_into_account)
        acc_list.append(acc)
        p_value_list.append(p_value)
        acc_sig_list.append(acc_sig)
        # save the results
    results = {'acc': acc_list, 'pvalue': p_value_list, 'acc_sig': acc_sig_list, 'trial_len_list': trial_len_list}
    train_type = 'Att' if MATCHATT else 'Unatt'
    if_saccade = 'Saccade' if saccade_multisubj_list is None else 'No_Saccade'
    table_name = f"{table_dir}Subj_{Subj_ID}_Match_{train_type}_{if_saccade}.pkl"
    with open(table_name, 'wb') as f:
        pickle.dump(results, f)

def pipe_single_obj(Subj_ID, eeg_multisubj_list, feat_match_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, figure_dir, table_dir, saccade_multisubj_list=None, nb_trials=200, fold=5, n_components=5, PLOT=False, trial_len=60, trial_len_list=None, nb_comp_into_account=2):
    eeg_onesubj_list = [eeg[:,:,Subj_ID] for eeg in eeg_multisubj_list]
    if saccade_multisubj_list is not None:
        saccade_onesubj_list = [saccade[:,:,Subj_ID] for saccade in saccade_multisubj_list]
    else:
        saccade_onesubj_list = None
    CCA = algo.CanonicalCorrelationAnalysis(eeg_onesubj_list, feat_match_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, fold=fold, n_components=n_components)
    _, corr_test, sig_corr, _, _, _, _, V_A_train, _ = CCA.cross_val(trial_len, saccade_list=saccade_onesubj_list)
    # find the indices of np.average(corr_test, axis=0) > sig_corr
    idx_sig = np.where(np.average(corr_test, axis=0) > sig_corr)[0]
    if PLOT:
        eeg_onesub = np.concatenate(tuple(eeg_onesubj_list), axis=0)
        forward_model = CCA.forward_model(eeg_onesub, V_A_train)
        figure_name = figure_dir + 'Subj_' + str(Subj_ID) + '_Trial_len_' + str(trial_len) +  '.png'
        utils.plot_spatial_resp(forward_model, corr_test, figure_name, idx_sig=idx_sig)
    if trial_len_list is not None:
        acc_list = []
        p_value_list = []
        acc_sig_list = []
        for trial_len in trial_len_list:
            corr_match_fold, corr_mismatch_fold = CCA.match_mismatch(trial_len, nb_trials, feat_distract_list=None)
            acc, p_value, acc_sig = utils.eval_mm(corr_match_fold, corr_mismatch_fold, nb_comp_into_account)
            acc_list.append(acc)
            p_value_list.append(p_value)
            acc_sig_list.append(acc_sig)
        # save the results
        results = {'acc': acc_list, 'pvalue': p_value_list, 'acc_sig': acc_sig_list, 'trial_len_list': trial_len_list}
        table_name = table_dir + 'Subj_' + str(Subj_ID) + '_mm.pkl'
        with open(table_name, 'wb') as f:
            pickle.dump(results, f)

# Data loading related parameters
# Subjected to change
subjects = ['Pilot_1', 'Pilot_2', 'Pilot_4', 'Pilot_5', 'Pilot_6']
bads = [['A30', 'B25'], ['B25'], ['B25'], [], ['A31', 'B31']] 
SINGLEOBJ = False
LOAD_ONLY = True
ALL_NEW = False
subj_to_analyze = 'Pilot_6'

# Not subjected to change
Subj_ID = subjects.index(subj_to_analyze)
PATTERN = 'Overlay'
subj_path = ['../../Experiments/data/Two_Obj/' + PATTERN + '/' + sub + '/' for sub in subjects]
nb_subj = len(subjects)
fsStim = 30
feats_path_folder = '../Feat_Multi/features/'
# Results saving related parameters
obj_type = 'SingleObj' if SINGLEOBJ else 'TwoObj'
data_types = ['EEG', 'EOG', 'ET']
figure_dirs = {}
table_dirs = {}
for data_type in data_types:
    figure_path = f'figures/{PATTERN}/{obj_type}/{data_type}/'
    utils.create_dir(figure_path)
    figure_dirs[data_type] = figure_path
    table_path = f'tables/{PATTERN}/{obj_type}/{data_type}/'
    utils.create_dir(table_path)
    table_dirs[data_type] = table_path


# Data loading and removing shot cuts
eeg_multisubj_list, eog_multisubj_list, feat_all_att_list, feat_all_unatt_list, gaze_multisubj_list, fs, len_seg_list = utils.load_data(subj_path, fsStim, bads, feats_path_folder, PATTERN, singleobj=False, LOAD_ONLY=LOAD_ONLY, ALL_NEW=ALL_NEW)
eeg_multisubj_list = [utils.remove_shot_cuts(eeg, fs) for eeg in eeg_multisubj_list]
eog_multisubj_list = [utils.remove_shot_cuts(eog, fs) for eog in eog_multisubj_list]
gaze_multisubj_list = [utils.remove_shot_cuts(gaze, fs) for gaze in gaze_multisubj_list]
feat_all_att_list = [utils.remove_shot_cuts(feat, fs) for feat in feat_all_att_list]
feat_all_unatt_list = [utils.remove_shot_cuts(feat, fs) for feat in feat_all_unatt_list]
saccade_multisubj_list = [np.expand_dims(gaze[:,2,:], axis=1) for gaze in gaze_multisubj_list]
gaze_multisubj_list = [gaze[:,:2,:] for gaze in gaze_multisubj_list]
objflow_att_list = [feats[:,8] for feats in feat_all_att_list]
objflow_unatt_list = [feats[:,8] for feats in feat_all_unatt_list]
if SINGLEOBJ:
    eeg_multisubj_SO_list, eog_multisubj_SO_list, feat_all_SO_list, _, gaze_multisubj_SO_list, fs, len_seg_SO_list = utils.load_data(subj_path, fsStim, bads, feats_path_folder, PATTERN, singleobj=True, LOAD_ONLY=LOAD_ONLY, ALL_NEW=ALL_NEW)
    eeg_multisubj_SO_list = [utils.remove_shot_cuts(eeg, fs) for eeg in eeg_multisubj_SO_list]
    eog_multisubj_SO_list = [utils.remove_shot_cuts(eog, fs) for eog in eog_multisubj_SO_list]
    gaze_multisubj_SO_list = [utils.remove_shot_cuts(gaze, fs) for gaze in gaze_multisubj_SO_list]
    feat_all_SO_list = [utils.remove_shot_cuts(feat, fs) for feat in feat_all_SO_list]
    saccade_multisubj_list_SO = [np.expand_dims(gaze[:,2,:], axis=1) for gaze in gaze_multisubj_SO_list]
    gaze_multisubj_SO_list = [gaze[:,:2,:] for gaze in gaze_multisubj_SO_list] 
    objflow_SO_list = [feats[:,8] for feats in feat_all_SO_list]

# optional: check the alignment of the data
utils.check_alignment(Subj_ID, eog_multisubj_list, gaze_multisubj_list, nb_points=500)

# Data processing related parameters, subjected to change
L_EEG = 3 
L_Stim = int(fsStim/2) 
offset_EEG = 1 
offset_Stim = 0 
trial_len_list = list(range(5, 95, 5))
feat_att_list = objflow_att_list
feat_unatt_list = objflow_unatt_list

if SINGLEOBJ:
    # Pipeline 0: Single object
    # Using EEG data
    pipe_single_obj(Subj_ID, eeg_multisubj_SO_list, objflow_SO_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, figure_dirs['EEG'], table_dirs['EEG'], fold=5, n_components=5, PLOT=True, trial_len=None, trial_len_list=None, nb_comp_into_account=2)
    pipe_single_obj(Subj_ID, eeg_multisubj_SO_list, objflow_SO_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, figure_dirs['EEG'], table_dirs['EEG'], saccade_multisubj_list=saccade_multisubj_list_SO, fold=5, n_components=5, PLOT=True, trial_len=None, trial_len_list=None, nb_comp_into_account=2)
    eeg_onesubj_SO_list = [eeg[:,:,Subj_ID] for eeg in eeg_multisubj_SO_list]
    CCA_SO = algo.CanonicalCorrelationAnalysis(eeg_onesubj_SO_list, objflow_SO_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, n_components=5)
    EEG_SO = np.concatenate(tuple(eeg_onesubj_SO_list), axis=0)
    feat_SO = np.concatenate(tuple(objflow_SO_list), axis=0)
    _, _, _, _, V_eeg_SO, V_Stim_SO, _ = CCA_SO.fit(EEG_SO, feat_SO)
    # Pipeline 1: Attended vs. Unattended
    acc, p_value, acc_sig, corr_att_cv, corr_unatt_cv= pipe_att_or_unatt(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, TRAIN_WITH_ATT=True, V_eeg=V_eeg_SO, V_Stim=V_Stim_SO, figure_dir=figure_dirs['EEG'], trial_len=None, PLOT=True)
    acc, p_value, acc_sig, corr_att_cv, corr_unatt_cv= pipe_att_or_unatt(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, TRAIN_WITH_ATT=True, saccade_multisubj_list=saccade_multisubj_list, V_eeg=V_eeg_SO, V_Stim=V_Stim_SO, figure_dir=figure_dirs['EEG'], trial_len=None, PLOT=True)
    # Pipeline 2: Compete trials
    pipe_compete_trials(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['EEG'], V_eeg=V_eeg_SO, V_Stim=V_Stim_SO, TRAIN_WITH_ATT=True, nb_trials=400, nb_comp_into_account=2) # Set TRAIN_WITH_ATT to True even use encoder and decoder trained with single object data, otherwise the results are 1-acc
    pipe_compete_trials(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['EEG'], saccade_multisubj_list=saccade_multisubj_list, V_eeg=V_eeg_SO, V_Stim=V_Stim_SO, TRAIN_WITH_ATT=True, nb_trials=400, nb_comp_into_account=2)
    # Same, but using EOG data
    eog_onesubj_SO_list = [eog[:,:,Subj_ID] for eog in eog_multisubj_SO_list]
    CCA_SO = algo.CanonicalCorrelationAnalysis(eog_onesubj_SO_list, objflow_SO_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, n_components=5)
    EOG_SO = np.concatenate(tuple(eog_onesubj_SO_list), axis=0)
    _, _, _, _, V_eog_SO, V_Stim_SO, _ = CCA_SO.fit(EOG_SO, feat_SO)
    pipe_compete_trials(Subj_ID, eog_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['EOG'], V_eeg=V_eog_SO, V_Stim=V_Stim_SO, TRAIN_WITH_ATT=True, nb_trials=400, nb_comp_into_account=2)
    pipe_compete_trials(Subj_ID, eog_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['EOG'], V_eeg=V_eog_SO, V_Stim=V_Stim_SO, TRAIN_WITH_ATT=True, saccade_multisubj_list=saccade_multisubj_list, nb_trials=400, nb_comp_into_account=2)
    # Same, but using Gaze data
    gaze_onesubj_SO_list = [gaze[:,:,Subj_ID] for gaze in gaze_multisubj_SO_list]
    CCA_SO = algo.CanonicalCorrelationAnalysis(gaze_onesubj_SO_list, objflow_SO_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, n_components=5)
    Gaze_SO = np.concatenate(tuple(gaze_onesubj_SO_list), axis=0)
    _, _, _, _, V_gaze_SO, V_Stim_SO, _ = CCA_SO.fit(Gaze_SO, feat_SO)
    pipe_compete_trials(Subj_ID, gaze_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['ET'], V_eeg=V_gaze_SO, V_Stim=V_Stim_SO, TRAIN_WITH_ATT=True, nb_trials=400, nb_comp_into_account=2)
    pipe_compete_trials(Subj_ID, gaze_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['ET'], V_eeg=V_gaze_SO, V_Stim=V_Stim_SO, TRAIN_WITH_ATT=True, saccade_multisubj_list=saccade_multisubj_list, nb_trials=400, nb_comp_into_account=2)
else:
    # Pipeline 1: Attended vs. Unattended
    # Using EEG data
    acc, p_value, acc_sig, corr_att_cv, corr_unatt_cv= pipe_att_or_unatt(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, TRAIN_WITH_ATT=True, figure_dir=figure_dirs['EEG'], trial_len=None, PLOT=True)
    acc, p_value, acc_sig, corr_att_cv, corr_unatt_cv= pipe_att_or_unatt(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, TRAIN_WITH_ATT=True, saccade_multisubj_list=saccade_multisubj_list, figure_dir=figure_dirs['EEG'], trial_len=None, PLOT=True)
    acc, p_value, acc_sig, corr_att_cv, corr_unatt_cv = pipe_att_or_unatt(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, TRAIN_WITH_ATT=False, figure_dir=figure_dirs['EEG'], trial_len=None, PLOT=True)
    # Using EOG data
    acc, p_value, acc_sig, corr_att_cv, corr_unatt_cv= pipe_att_or_unatt(Subj_ID, eog_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, TRAIN_WITH_ATT=True, figure_dir=figure_dirs['EOG'], trial_len=None, PLOT=False)
    # Using Gaze data
    acc, p_value, acc_sig, corr_att_cv, corr_unatt_cv= pipe_att_or_unatt(Subj_ID, gaze_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, TRAIN_WITH_ATT=True, figure_dir=figure_dirs['ET'], trial_len=None, PLOT=False)

    # Pipeline 2: Compete trials
    # Using EEG data
    pipe_compete_trials(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['EEG'], TRAIN_WITH_ATT=True, nb_trials=400)
    pipe_compete_trials(Subj_ID, eeg_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['EEG'], TRAIN_WITH_ATT=True, saccade_multisubj_list=saccade_multisubj_list, nb_trials=400)
    # Using EOG data
    pipe_compete_trials(Subj_ID, eog_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['EOG'], TRAIN_WITH_ATT=True, nb_trials=400)
    pipe_compete_trials(Subj_ID, eog_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['EOG'], TRAIN_WITH_ATT=True, saccade_multisubj_list=saccade_multisubj_list, nb_trials=400)
    # Using Gaze data
    pipe_compete_trials(Subj_ID, gaze_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['ET'], TRAIN_WITH_ATT=True, nb_trials=400)
    pipe_compete_trials(Subj_ID, gaze_multisubj_list, feat_att_list, feat_unatt_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['ET'], TRAIN_WITH_ATT=True, saccade_multisubj_list=saccade_multisubj_list, nb_trials=400)

    # Pipeline 3: Matched vs. Unmatched
    # Using EEG data
    feat_match_list = objflow_att_list
    feat_distract_list = objflow_unatt_list
    MATCHATT = True
    # with saccade
    pipe_mm_trials(Subj_ID, eeg_multisubj_list, feat_match_list, feat_distract_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['EEG'], MATCHATT, nb_trials=400, fold=5, n_components=5, nb_comp_into_account=2)
    # without saccade
    pipe_mm_trials(Subj_ID, eeg_multisubj_list, feat_match_list, feat_distract_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['EEG'], MATCHATT, saccade_multisubj_list=saccade_multisubj_list, nb_trials=400, fold=5, n_components=5, nb_comp_into_account=2)

    feat_match_list = objflow_unatt_list
    feat_distract_list = objflow_att_list
    MATCHATT = False
    pipe_mm_trials(Subj_ID, eeg_multisubj_list, feat_match_list, feat_distract_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['EEG'], MATCHATT, nb_trials=400, fold=5, n_components=5, nb_comp_into_account=2)
    pipe_mm_trials(Subj_ID, eeg_multisubj_list, feat_match_list, feat_distract_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, trial_len_list, table_dirs['EEG'], MATCHATT, saccade_multisubj_list=saccade_multisubj_list, nb_trials=400, fold=5, n_components=5, nb_comp_into_account=2)
