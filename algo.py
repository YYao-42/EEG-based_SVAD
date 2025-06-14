'''
Inherited from the previous project.
Class CanonicalCorrelationAnalysis and Class GeneralizedCCA are heavily modified.
Class DiscriminativeCCA, GeneralizedCCA_MultiMod and GCCAPreprocessedCCA are newly added.
Other classes are not used and modified, and thus do not have all the new features.
'''

import numpy as np
import copy
import random
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from numpy import linalg as LA
from scipy.linalg import eig, eigh, sqrtm, lstsq
from scipy.stats import pearsonr
import utils


class LeastSquares:
    '''
    Note:
    When working with a forward model, the encoder maps the stimuli (or the latent variables) to the EEG data. We need to take more past samples of the stimuli into account. Therefore the offset should be zero or a small number to compensate for the possible misalignment.
    When working with a backward model, the decoder maps the EEG data (or the latent variables) to the stimuli. We need to take more future samples of the EEG data into account. Therefore the offset should be L-1 or a slightly smaller number than L-1. 
    '''
    def __init__(self, EEG_list, Stim_list, fs, decoding, L_EEG=1, offset_EEG=0, L_Stim=1, offset_Stim=0, fold=10, message=True, signifi_level=True, pool=True, n_permu=500, p_value=0.05):
        self.EEG_list = EEG_list
        self.Stim_list = Stim_list
        self.fs = fs
        self.decoding = decoding
        self.L_EEG = L_EEG
        self.offset_EEG = offset_EEG
        self.L_Stim = L_Stim
        self.offset_Stim = offset_Stim
        self.fold = fold
        self.message = message
        self.signifi_level = signifi_level
        self.pool = pool
        self.n_permu = n_permu
        self.p_value = p_value
        if decoding:
            if np.ndim(Stim_list[0]) == 1:
                self.n_components = 1
            else:
                self.n_components = Stim_list[0].shape[1]
        else:
            self.n_components = EEG_list[0].shape[1]

    def encoder(self, EEG, Stim, W_f=None):
        '''
        Inputs:
        EEG: T(#sample)xD_eeg(#channel) array
        stim: T(#sample)xD_stim(#feature dim) array
        W_f: only used in test mode
        Output:
        W_f: (D_stim*L_stim)xD_eeg array
        '''
        self.n_components = EEG.shape[1] # in case functions in other classes call this function and the EEG is different from the one in the initialization
        Stim_Hankel = utils.block_Hankel(Stim, self.L_Stim, self.offset_Stim)
        if W_f is not None: # Test Mode
            pass
        else:
            W_f = lstsq(Stim_Hankel, EEG)[0]
        filtered_Stim = Stim_Hankel@W_f
        mse = np.mean((filtered_Stim-EEG)**2)
        return W_f, mse

    def decoder(self, EEG, Stim, W_b=None):
        '''
        Inputs:
        EEG: T(#sample)xD_eeg(#channel) array
        stim: T(#sample)xD_stim(#feature dim) array
        W_b: only used in test mode
        Output:
        W_b: (D_eeg*L_eeg)xD_stim array
        '''
        if np.ndim(Stim) == 1:
            Stim = np.expand_dims(Stim, axis=1)
        self.n_components = Stim.shape[1] # in case functions in other classes call this function and the Stim is different from the one in the initialization
        EEG_Hankel = utils.block_Hankel(EEG, self.L_EEG, self.offset_EEG)
        if W_b is not None: # Test Mode
            pass
        else:
            W_b = lstsq(EEG_Hankel, Stim)[0]
        filtered_EEG = EEG_Hankel@W_b
        return W_b, filtered_EEG


class CanonicalCorrelationAnalysis:
    def __init__(self, EEG_list, Stim_list, fs, L_EEG, L_Stim, offset_EEG=0, offset_Stim=0, mask_list=None, dim_list_EEG=None, dim_list_Stim=None, fold=10, leave_out=2, n_components=5, REGFEATS=False, regularization='lwcov', K_regu=None, message=True, signifi_level=True, n_permu=500, p_value=0.05, dim_subspace=2):
        '''
        EEG_list: list of EEG data, each element is a T(#sample)xDx(#channel) array corresponding to a video 
        Stim_list: list of stimulus, each element is a T(#sample)xDy(#feature dim) array corresponding to a video 
        fs: Sampling rate
        L_EEG/L_Stim: If use (spatial-) temporal filter, the number of taps
        offset_EEG/offset_Stim: If use (spatial-) temporal filter, the offset of time lags
        mask_list: list of mask. Certain time points (e.g., when having saccades) can be excluded from the analysis using masks.
        dim_list_EEG/dim_list_Stim: If 'EEG' is actually a stack of data from different modalities, dim_list_EEG should be a list of the dimensions of each modality. The same for 'Stim'.
        fold: Number of folds for cross-validation [Not used in the current version]
        leave_out: Number of videos left out for each fold
        n_components: Number of components to be returned
        REGFEATS: whether to control for the competing features in attention decoding and match mismatch task
        regularization: Regularization of the estimated covariance matrix
        K_regu: Number of eigenvalues to be kept. Others will be set to zero. Keep all if K_regu=None
        message: Print out the results if True
        signifi_level: Calculate the significance level if True
        n_permu: Number of permutations for the significance level
        p_value: Significance level
        dim_subspace: Number of components to be considered for metric TSC and ChDist
        '''
        self.EEG_list = EEG_list
        self.Stim_list = Stim_list
        self.fs = fs
        self.L_EEG = L_EEG
        self.L_Stim = L_Stim
        self.offset_EEG = offset_EEG
        self.offset_Stim = offset_Stim
        self.mask_list = mask_list
        self.dim_list_EEG = dim_list_EEG
        self.dim_list_Stim = dim_list_Stim
        self.fold = fold
        self.leave_out = leave_out
        self.n_components = n_components
        self.REGFEATS = REGFEATS
        self.regularization = regularization
        self.K_regu = K_regu
        self.message = message
        self.signifi_level = signifi_level
        self.n_permu = n_permu
        self.p_value = p_value
        self.dim_subspace = dim_subspace
        self.mask_train = None
        self.mask_test = None

    def select_mask(self, length):
        '''
        Select the mask (based on the length of the data)
        Expand the mask to cover the lags and correct for the offset
        '''
        if self.mask_list is not None:
            if len(self.mask_train) == length:
                mask = self.mask_train
            else:
                mask = self.mask_test
            mask_lags = utils.expand_mask(mask, self.L_EEG, self.offset_EEG)
        else:
            mask_lags = None
        return mask_lags

    def fit(self, X, Y, V_A=None, V_B=None, Lam=None):
        if np.ndim(Y) == 1:
            Y = np.expand_dims(Y, axis=1)
        T, Dx = X.shape
        _, Dy = Y.shape
        mask = self.select_mask(T)
        Lx = self.L_EEG
        Ly = self.L_Stim
        n_components = self.n_components
        mtx_X = utils.block_Hankel(X, Lx, self.offset_EEG, mask)
        dim_list_X = [d*Lx for d in self.dim_list_EEG] if self.dim_list_EEG is not None else [Dx*Lx]
        dim_list_Y = [d*Ly for d in self.dim_list_Stim] if self.dim_list_Stim is not None else [Dy*Ly]
        mtx_Y = utils.block_Hankel(Y, Ly, self.offset_Stim, mask)
        if V_A is not None: # Test Mode
            flag_test = True
        else: # Train mode
            flag_test = False
            # compute covariance matrices
            covXY = np.cov(mtx_X, mtx_Y, rowvar=False)
            if self.regularization=='lwcov':
                Rx, _ = utils.get_cov_mtx(mtx_X, dim_list_X, self.regularization)
                Ry, _ = utils.get_cov_mtx(mtx_Y, dim_list_Y, self.regularization)
            else:
                Rx = covXY[:Dx*Lx,:Dx*Lx]
                Ry = covXY[Dx*Lx:Dx*Lx+Dy*Ly,Dx*Lx:Dx*Lx+Dy*Ly]
            Rxy = covXY[:Dx*Lx,Dx*Lx:Dx*Lx+Dy*Ly]
            Ryx = covXY[Dx*Lx:Dx*Lx+Dy*Ly,:Dx*Lx]
            # PCA regularization (set K_regu<rank(Rx))
            # such that the small eigenvalues dominated by noise are discarded
            if self.K_regu is None:
                invRx = utils.PCAreg_inv(Rx, LA.matrix_rank(Rx))
                invRy = utils.PCAreg_inv(Ry, LA.matrix_rank(Ry))
            else:
                K_regu = min(LA.matrix_rank(Rx), LA.matrix_rank(Ry), self.K_regu)
                invRx = utils.PCAreg_inv(Rx, K_regu)
                invRy = utils.PCAreg_inv(Ry, K_regu)
            A = invRx@Rxy@invRy@Ryx
            B = invRy@Ryx@invRx@Rxy
            # lam of A and lam of B should be the same
            # can be used as a preliminary check for correctness
            # the correlation coefficients are already available by taking sqrt of the eigenvalues: corr_coe = np.sqrt(lam[:K_regu])
            # or we do the following to obtain transformed X and Y and calculate corr_coe from there
            Lam, V_A = utils.eig_sorted(A)
            _, V_B = utils.eig_sorted(B)
            Lam = np.real(Lam[:n_components])
            V_A = np.real(V_A[:,:n_components])
            V_B = np.real(V_B[:,:n_components])
        # mtx_X and mtx_Y should be centered according to the definition. But since we calculate the correlation coefficients, it does not matter.
        X_trans = mtx_X@V_A
        Y_trans = mtx_Y@V_B
        corr_pvalue = [pearsonr(X_trans[:,k], Y_trans[:,k]) for k in range(n_components)]
        corr_coe = np.array([corr_pvalue[k][0] for k in range(n_components)])
        # P-value-null hypothesis: the distributions underlying the samples are uncorrelated and normally distributed.
        p_value = np.array([corr_pvalue[k][1] for k in range(n_components)])
        if not flag_test:
            # to match filters v_a and v_b s.t. corr_coe is always positive
            V_A[:,corr_coe<0] = -1*V_A[:,corr_coe<0]
            corr_coe[corr_coe<0] = -1*corr_coe[corr_coe<0]
            # Note: Due to the regularization, the correlation coefficients are not exactly the same as the quare root of the Lam. 
        TSC = np.sum(np.square(corr_coe[:self.dim_subspace]))
        ChDist = np.sqrt(self.dim_subspace-TSC)
        return corr_coe, TSC, ChDist, p_value, V_A, V_B, Lam

    def get_transformed_data(self, X, Y, V_A, V_B, C=None):
        '''
        Get the transformed data
        X: EEG or other data modalities; V_A: filters for X
        Y: features; V_B: filters for Y
        C: competing features
        '''
        mask = self.select_mask(X.shape[0])
        mtx_X = utils.block_Hankel(X, self.L_EEG, self.offset_EEG, mask)
        mtx_Y = utils.block_Hankel(Y, self.L_Stim, self.offset_Stim, mask)
        mtx_X_centered = mtx_X - np.mean(mtx_X, axis=0, keepdims=True)
        mtx_Y_centered = mtx_Y - np.mean(mtx_Y, axis=0, keepdims=True)
        if C is not None:
            mtx_C = utils.block_Hankel(C, self.L_Stim, self.offset_Stim, mask)
            mtx_C_centered = mtx_C - np.mean(mtx_C, axis=0, keepdims=True)
            mtx_X_centered, mtx_Y_centered = utils.regress_out_2D_pair(mtx_X_centered, mtx_Y_centered, mtx_C_centered)
        X_trans = mtx_X_centered@V_A
        Y_trans = mtx_Y_centered@V_B
        return X_trans, Y_trans

    def get_corr_coe(self, X_trans, Y_trans):
        '''
        Get the statistics from the transformed data
        '''
        # corr_pvalue = [pearsonr(X_trans[:,k], Y_trans[:,k]) for k in range(self.n_components)]
        # corr_coe = np.array([corr_pvalue[k][0] for k in range(self.n_components)])
        # p_value = np.array([corr_pvalue[k][1] for k in range(self.n_components)])
        corr_coe = np.array([np.corrcoef(X_trans[:,k], Y_trans[:,k])[0,1] for k in range(self.n_components)])
        p_value = None
        TSC = np.sum(np.square(corr_coe[:self.dim_subspace]))
        ChDist = np.sqrt(self.dim_subspace-TSC)
        return corr_coe, TSC, ChDist, p_value

    def cal_corr_coe(self, X, Y, V_A=None, V_B=None, C=None):
        '''
        Same as get_corr_coe but with the input of the data and the filters
        '''
        if V_A is None:
            X_trans, Y_trans = X, Y
        else:
            X_trans, Y_trans = self.get_transformed_data(X, Y, V_A, V_B, C)
        corr_coe, TSC, ChDist, p_value = self.get_corr_coe(X_trans, Y_trans)
        return corr_coe, TSC, ChDist, p_value

    def cal_corr_coe_trials(self, X_trials, Y_trials, V_A=None, V_B=None, avg=True, C_trials=None):
        C_trials = [None]*len(X_trials) if C_trials is None else C_trials
        stats = [(self.cal_corr_coe(X, Y, V_A, V_B, C)) for X, Y, C in zip(X_trials, Y_trials, C_trials)]
        corr_coe = np.concatenate(tuple([np.expand_dims(stats[i][0],axis=0) for i in range(len(X_trials))]), axis=0)
        TSC = np.array([stats[i][1] for i in range(len(X_trials))])
        ChDist = np.array([stats[i][2] for i in range(len(X_trials))])
        if avg:
            corr_coe = np.mean(corr_coe, axis=0)
            TSC = np.mean(TSC)
            ChDist = np.mean(ChDist)
        return corr_coe, TSC, ChDist

    def cal_corr_compete_trials(self, X, Y_att, V_X, V_Y, BOOTSTRAP, trial_len, Y_unatt=None, given_start_points=None, BTfactor=3):
        '''
        This function versus next function:
        Must use this one if self.REGFEATS is True because the regression should be done segment-wise (then `get_transformed_data` shouldn't be called first)
        '''
        T = X.shape[0]
        if T-trial_len*self.fs >= 0:
            if given_start_points is None:
                if BOOTSTRAP:
                    nb_trials = min(T//self.fs//BTfactor, 1000)
                    start_points = np.random.randint(0, T-trial_len*self.fs, size=nb_trials)
                    start_points = np.sort(start_points)
                else:
                    start_points = np.array(range(0, T - T%(self.fs*trial_len), self.fs*trial_len))
            else:
                start_points = given_start_points
            X_trials = utils.into_trials(X, self.fs, trial_len, start_points=start_points)
            Y_att_trials = utils.into_trials(Y_att, self.fs, trial_len, start_points=start_points)
            Y_compete_trials = utils.into_trials(Y_unatt, self.fs, trial_len, start_points=start_points) if Y_unatt is not None else [utils.select_distractors([Y_att], self.fs, trial_len, start_point)[0] for start_point in start_points]
            C_trials = Y_compete_trials if self.REGFEATS else None
            corr_att_trials = self.cal_corr_coe_trials(X_trials, Y_att_trials, V_X, V_Y, avg=False, C_trials=C_trials)[0]
            corr_compete_trials = self.cal_corr_coe_trials(X_trials, Y_compete_trials, V_X, V_Y, avg=False, C_trials=C_trials)[0]
            corr_att_compete_trials = np.stack([np.corrcoef(Att[:,0], Compete[:,0])[0,1] for Att, Compete in zip(Y_att_trials, Y_compete_trials)])
        else:
            print('The length of the video is too short for the given trial length.')
            corr_att_trials = np.full((1, self.n_components), np.nan)
            corr_compete_trials = np.full((1, self.n_components), np.nan)
            corr_att_compete_trials = np.nan
        return corr_att_trials, corr_compete_trials, corr_att_compete_trials, start_points

    def cal_corr_compete_trials_mask(self, X, Y_att, V_X, V_Y, BOOTSTRAP, trial_len, Y_unatt=None, given_start_points=None, BTfactor=3):
        '''
        This function versus previous function:
        Must use this one if data need to be masked beforehand (then `get_transformed_data` should be called first)
        '''
        X_trans, Y_att_trans = self.get_transformed_data(X, Y_att, V_X, V_Y)
        _, Y_unatt_trans = self.get_transformed_data(X, Y_unatt, V_X, V_Y) if Y_unatt is not None else [None, None]
        T = X_trans.shape[0]
        if T-trial_len*self.fs >= 0:
            if given_start_points is None:
                if BOOTSTRAP:
                    nb_trials = min(T//self.fs//BTfactor, 1000)
                    start_points = np.random.randint(0, T-trial_len*self.fs, size=nb_trials)
                    start_points = np.sort(start_points)
                else:
                    start_points = np.array(range(0, T - T%(self.fs*trial_len), self.fs*trial_len))
            else:
                start_points = given_start_points
            Y_att_trials = utils.into_trials(Y_att, self.fs, trial_len, start_points=start_points)
            Y_compete_trials = utils.into_trials(Y_unatt, self.fs, trial_len, start_points=start_points) if Y_unatt is not None else [utils.select_distractors([Y_att], self.fs, trial_len, start_point)[0] for start_point in start_points]
            corr_att_compete_trials = np.stack([np.corrcoef(Att[:,0], Compete[:,0])[0,1] for Att, Compete in zip(Y_att_trials, Y_compete_trials)])
            # Note: the data has been transformed
            X_trans_trials = utils.into_trials(X_trans, self.fs, trial_len, start_points=start_points)
            Y_att_trans_trials = utils.into_trials(Y_att_trans, self.fs, trial_len, start_points=start_points)
            Y_compete_trans_trials = utils.into_trials(Y_unatt_trans, self.fs, trial_len, start_points=start_points) if Y_unatt_trans is not None else [utils.select_distractors([Y_att_trans], self.fs, trial_len, start_point)[0] for start_point in start_points]
            corr_att_trials = self.cal_corr_coe_trials(X_trans_trials, Y_att_trans_trials, avg=False)[0]
            corr_compete_trials = self.cal_corr_coe_trials(X_trans_trials, Y_compete_trans_trials, avg=False)[0]
        else:
            print('The length of the video is too short for the given trial length.')
            corr_att_trials = np.full((1, self.n_components), np.nan)
            corr_compete_trials = np.full((1, self.n_components), np.nan)
            corr_att_compete_trials = np.nan
        return corr_att_trials, corr_compete_trials, corr_att_compete_trials, start_points, X_trans_trials, Y_att_trans_trials, Y_compete_trans_trials

    def permutation_test(self, X, Y, V_A, V_B, PHASE_SCRAMBLE=True, block_len=None, X_trans=None, Y_trans=None, C=None):
        '''
        Permutation test for the correlation coefficients. Use phase scrambling or block shuffling.
        '''
        corr_coe_topK = np.zeros((self.n_permu, self.n_components))
        if X_trans is None and Y_trans is None:
            X_trans, Y_trans = self.get_transformed_data(X, Y, V_A, V_B, C) 
        for i in tqdm(range(self.n_permu)):
            X_shuffled = utils.shuffle_2D(X_trans, block_len) if not PHASE_SCRAMBLE else utils.phase_scramble_2D(X_trans)
            Y_shuffled = utils.shuffle_2D(Y_trans, block_len) if not PHASE_SCRAMBLE else utils.phase_scramble_2D(Y_trans)
            # corr_pvalue = [pearsonr(X_shuffled[:,k], Y_shuffled[:,k]) for k in range(self.n_components)]
            # corr_coe_topK[i,:] = np.array([corr_pvalue[k][0] for k in range(self.n_components)])
            corr_coe_topK[i,:] = np.array([np.corrcoef(X_shuffled[:,k], Y_shuffled[:,k])[0,1] for k in range(self.n_components)])
        return corr_coe_topK

    def permutation_test_acc(self, X_trials, Y1_trials, Y2_trials, nb_permu=100):
        acc_list = []
        corr_avg_list = []
        for i in tqdm(range(nb_permu)):
            # randomly shuffle X_trials
            X_trials_shifted = copy.deepcopy(X_trials)
            shift_amount = random.randint(len(X_trials)//4, len(X_trials)//4*3)
            X_trials_shifted = X_trials_shifted[shift_amount:] + X_trials_shifted[:shift_amount]
            corr_X1_trials = self.cal_corr_coe_trials(X_trials_shifted, Y1_trials, avg=False)[0]
            corr_X2_trials = self.cal_corr_coe_trials(X_trials_shifted, Y2_trials, avg=False)[0]
            acc, _, _, corr_X1_avg, _= utils.eval_compete(corr_X1_trials, corr_X2_trials, TRAIN_WITH_ATT=True, message=False)
            acc_list.append(acc)
            corr_avg_list.append(corr_X1_avg)
        return acc_list, corr_avg_list

    def forward_model(self, X, V_A, X_trans=None):
        '''
        Inputs:
        X: observations (one subject) TxD
        V_A: filters/backward models DLxK
        X_trans: transformed TxK
        (Do not consider REGFEATS here)
        Output:
        F: forward model
        '''
        if X_trans is not None: 
            # Calculate the forward model based on the original data
            F = (lstsq(X_trans, X)[0]).T
        else: 
            # Calculate the forward model based on the Hankelized data and extract the forward model corresponding to the correct time points
            mask = self.select_mask(X.shape[0])
            X_block_Hankel = utils.block_Hankel(X, self.L_EEG, self.offset_EEG, mask)
            Rxx = np.cov(X_block_Hankel, rowvar=False)
            if np.ndim(Rxx) == 0:
                Rxx = np.array([[Rxx]])
            F_redun = Rxx@V_A@LA.inv(V_A.T@Rxx@V_A)
            F = utils.F_organize(F_redun, self.L_EEG, self.offset_EEG)
        return F

    def calculate_sig_corr(self, corr_trials, nb_fold=1):
        '''
        Calculate the significance level based on the results of the permutation test
        '''
        assert self.n_components*self.n_permu*nb_fold == corr_trials.shape[0]*corr_trials.shape[1]
        sig_idx = -int(self.n_permu*self.p_value*self.n_components*nb_fold)
        corr_trials = np.sort(abs(corr_trials), axis=None)
        return corr_trials[sig_idx]
    
    def cross_val_LVO(self):
        '''
        Cross-validation with leave-one-pair-out; For single-object dataset only
        '''
        nb_videos = len(self.Stim_list)
        n_components = self.n_components
        nb_folds = nb_videos//self.leave_out
        assert nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
        nested_datalist = [self.EEG_list, self.Stim_list, self.mask_list] if self.mask_list is not None else [self.EEG_list, self.Stim_list]
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO(nested_datalist, self.leave_out)
        assert len(train_list_folds) == len(test_list_folds) == nb_folds, "The number of folds is not correct."
        corr_train_fold = np.zeros((nb_folds, n_components))
        corr_test_fold = np.zeros((nb_folds, n_components))
        corr_permu_fold = []
        forward_model_fold = []
        for idx in range(0, nb_folds):
            if self.mask_list is not None:
                [EEG_train, Sti_train, self.mask_train], [EEG_test, Sti_test, self.mask_test] = train_list_folds[idx], test_list_folds[idx]
            else:
                [EEG_train, Sti_train], [EEG_test, Sti_test] = train_list_folds[idx], test_list_folds[idx]
            corr_train_fold[idx,:], _, _, _, V_A_train, V_B_train, _ = self.fit(EEG_train, Sti_train)
            corr_test_fold[idx,:], _, _, _ = self.cal_corr_coe(EEG_test, Sti_test, V_A_train, V_B_train)
            forward_model = self.forward_model(EEG_test, V_A_train)
            forward_model_fold.append(forward_model)
            corr_permu_fold.append(self.permutation_test(EEG_test, Sti_test, V_A=V_A_train, V_B=V_B_train))
        sig_corr_fold = [self.calculate_sig_corr(corr_permu) for corr_permu in corr_permu_fold]
        corr_permu_all = np.concatenate(tuple(corr_permu_fold), axis=0)
        sig_corr_pool = self.calculate_sig_corr(corr_permu_all, nb_fold=nb_folds)
        if self.message:
            print('Average correlation coefficients of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train_fold, axis=0)))
            print('Average correlation coefficients of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test_fold, axis=0)))
            print('Significance level: {}'.format(sig_corr_pool))
        return corr_train_fold, corr_test_fold, sig_corr_fold, sig_corr_pool, forward_model_fold

    def match_mismatch_LVO(self, trial_len, BOOTSTRAP=True, V_eeg=None, V_Stim=None):
        '''
        Match-Mismatch task with leave-one-pair-out
        Always train on match and try to distinguish match from mismatch 
        Mismatch is a random segment that is not shown on the screen
        '''

        nb_videos = len(self.Stim_list)
        nb_folds = nb_videos//self.leave_out
        assert nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
        nested_datalist = [self.EEG_list, self.Stim_list, self.mask_list] if self.mask_list is not None else [self.EEG_list, self.Stim_list]
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO(nested_datalist, self.leave_out)
        assert len(train_list_folds) == len(test_list_folds) == nb_folds, "The number of folds is not correct."
        corr_match_eeg = []
        corr_mismatch_eeg = []
        corr_match_mm = []
        for idx in range(0, nb_folds):
            if self.mask_list is not None:
                [EEG_train, Sti_train, self.mask_train], [EEG_test, Sti_test, self.mask_test] = train_list_folds[idx], test_list_folds[idx]
            else:
                [EEG_train, Sti_train], [EEG_test, Sti_test] = train_list_folds[idx], test_list_folds[idx]
            if V_eeg is None:
                _, _, _, _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Sti_train)
            else:
                V_eeg_train, V_feat_train = V_eeg, V_Stim
            if self.REGFEATS:
                corr_match_eeg_i, corr_mismatch_eeg_i, corr_match_mm_i, _ = self.cal_corr_compete_trials(EEG_test, Sti_test, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len) 
            else: 
                corr_match_eeg_i, corr_mismatch_eeg_i, corr_match_mm_i, _, _, _ = self.cal_corr_compete_trials_mask(EEG_test, Sti_test, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len)
            corr_match_eeg.append(corr_match_eeg_i)
            corr_mismatch_eeg.append(corr_mismatch_eeg_i)
            corr_match_mm.append(corr_match_mm_i)
        
        corr_match_eeg = np.concatenate(tuple(corr_match_eeg), axis=0)
        corr_mismatch_eeg = np.concatenate(tuple(corr_mismatch_eeg), axis=0)
        corr_match_mm = np.concatenate(tuple(corr_match_mm), axis=0)
        return corr_match_eeg, corr_mismatch_eeg, corr_match_mm

    def att_or_unatt_LVO(self, feat_unatt_list, TRAIN_WITH_ATT, V_eeg=None, V_Stim=None, EEG_ori_list=None, COMBINE_ATT_UNATT=False):
        '''
        Get correlations with attended and unattended videos in leave-one-pair-out scheme
        Two modes:
            Train on attended data and try to decode the attended object
            Train on unattended data and try to decode the unattended object
        Attended and unattended data were both shown on the screen 
        '''
        feat_att_list = self.Stim_list
        nb_videos = len(feat_att_list)
        nb_folds = nb_videos//self.leave_out
        assert nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
        nested_datalist = [self.EEG_list, feat_att_list, feat_unatt_list, self.mask_list, EEG_ori_list]
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO(nested_datalist, self.leave_out)
        assert len(train_list_folds) == len(test_list_folds) == nb_folds, "The number of folds is not correct."
        corr_att_fold = []
        corr_unatt_fold = []
        corr_permu_fold = []
        forward_model_fold = []
        
        for idx in range(0, nb_folds):
            [EEG_train, Att_train, Unatt_train, self.mask_train, _], [EEG_test, Att_test, Unatt_test, self.mask_test, EEG_ori_test] = train_list_folds[idx], test_list_folds[idx]
            if V_eeg is None:
                if TRAIN_WITH_ATT:
                    if COMBINE_ATT_UNATT:
                        Att_train = np.expand_dims(Att_train, axis=1) if len(Att_train.shape) == 1 else Att_train
                        Unatt_train = np.expand_dims(Unatt_train, axis=1) if len(Unatt_train.shape) == 1 else Unatt_train
                        feat_cb = np.concatenate((Att_train, Unatt_train), axis=1)
                        _, _, _, _, V_eeg_train, V_feat_cb, _ = self.fit(EEG_train, feat_cb)
                        V_feat_train = V_feat_cb[:V_feat_cb.shape[0]//2,:]
                    else:
                        _, _, _, _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Att_train)
                else:
                    _, _, _, _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Unatt_train)
            else:
                V_eeg_train, V_feat_train = V_eeg, V_Stim
            corr_att, _, _, _ = self.cal_corr_coe(EEG_test, Att_test, V_eeg_train, V_feat_train) if not self.REGFEATS else self.cal_corr_coe(EEG_test, Att_test, V_eeg_train, V_feat_train, C=Unatt_test)
            corr_unatt, _, _, _ = self.cal_corr_coe(EEG_test, Unatt_test, V_eeg_train, V_feat_train) if not self.REGFEATS else self.cal_corr_coe(EEG_test, Unatt_test, V_eeg_train, V_feat_train, C=Att_test)
            corr_att = np.expand_dims(corr_att, axis=0)
            corr_unatt = np.expand_dims(corr_unatt, axis=0)
            if EEG_ori_test is not None:
                EEG_trans, _ = self.get_transformed_data(EEG_test, Att_test, V_eeg_train, V_feat_train)
                forward_model = self.forward_model(EEG_ori_test, V_eeg_train, X_trans=EEG_trans)
            else:
                forward_model = self.forward_model(EEG_test, V_eeg_train)

            corr_permu_fold.append(self.permutation_test(EEG_test, Att_test, V_A=V_eeg_train, V_B=V_feat_train)) if not self.REGFEATS else corr_permu_fold.append(self.permutation_test(EEG_test, Att_test, V_A=V_eeg_train, V_B=V_feat_train, C=Unatt_test))
            corr_att_fold.append(corr_att)
            corr_unatt_fold.append(corr_unatt)
            forward_model_fold.append(forward_model)
        corr_att_fold = np.concatenate(tuple(corr_att_fold), axis=0)
        corr_unatt_fold = np.concatenate(tuple(corr_unatt_fold), axis=0)
        sig_corr_fold = [self.calculate_sig_corr(corr_permu) for corr_permu in corr_permu_fold] # Checked: sig_corr_unatt is close to sig_corr_att          
        corr_permu_all = np.concatenate(tuple(corr_permu_fold), axis=0)
        sig_corr_pool = self.calculate_sig_corr(corr_permu_all, nb_fold=nb_folds)
        if self.message:
            modes = [
                ("Mode 1: train and test with attended features", corr_att_fold),
                ("Mode 2: train with attended features and test with unattended features", corr_unatt_fold),
                ("Mode 3: train and test with unattended features", corr_unatt_fold),
                ("Mode 4: train with unattended features and test with attended features", corr_att_fold),
                ("Mode 5: train with single object data and test with attended features", corr_att_fold),
                ("Mode 6: train with single object data and test with unattended features", corr_unatt_fold)
            ]
            print('Significance level: {}'.format(sig_corr_pool))
            for i, (mode, corr) in enumerate(modes):
                if V_eeg is not None and i < 4:
                    continue
                elif V_eeg is None and TRAIN_WITH_ATT and i > 1:
                    continue
                elif V_eeg is None and not TRAIN_WITH_ATT and (i < 2 or i > 3):
                    continue
                print(mode)
                print(np.average(corr, axis=0))
        return corr_att_fold, corr_unatt_fold, sig_corr_fold, sig_corr_pool, forward_model_fold

    def visual_attention_decoding_LVO(self, feat_unatt_list, trial_len, BOOTSTRAP=True, V_eeg=None, V_Stim=None, COMBINE_ATT_UNATT=False, PERMU_TEST=False):
        '''
        Visual attention decoding task with leave-one-pair-out
        Always train on attended data and try to decode the attended object
        '''
        feat_att_list = self.Stim_list
        nb_videos = len(feat_att_list)
        nb_folds = nb_videos//self.leave_out
        assert nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
        nested_datalist = [self.EEG_list, feat_att_list, feat_unatt_list, self.mask_list] if self.mask_list is not None else [self.EEG_list, feat_att_list, feat_unatt_list]
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO(nested_datalist, self.leave_out)
        assert len(train_list_folds) == len(test_list_folds) == nb_folds, "The number of folds is not correct."
        corr_att_eeg = []
        corr_unatt_eeg = []
        corr_att_unatt = []
        eeg_t_all_trials = []
        att_t_all_trials = []
        unatt_t_all_trials = []
        for idx in range(0, nb_folds):
            if self.mask_list is not None:
                [EEG_train, Att_train, Unatt_train, self.mask_train], [EEG_test, Att_test, Unatt_test, self.mask_test] = train_list_folds[idx], test_list_folds[idx]
            else:
                [EEG_train,  Att_train, Unatt_train], [EEG_test, Att_test, Unatt_test] = train_list_folds[idx], test_list_folds[idx]
            if V_eeg is None:
                    if COMBINE_ATT_UNATT:
                        Att_train = np.expand_dims(Att_train, axis=1) if len(Att_train.shape) == 1 else Att_train
                        Unatt_train = np.expand_dims(Unatt_train, axis=1) if len(Unatt_train.shape) == 1 else Unatt_train
                        feat_cb = np.concatenate((Att_train, Unatt_train), axis=1)
                        _, _, _, _, V_eeg_train, V_feat_cb, _ = self.fit(EEG_train, feat_cb)
                        V_feat_train = V_feat_cb[:V_feat_cb.shape[0]//2,:]
                    else:
                        _, _, _, _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Att_train)
            else:
                 V_eeg_train, V_feat_train = V_eeg, V_Stim
            if self.REGFEATS:
                corr_att_eeg_i, corr_unatt_eeg_i, corr_att_unatt_i, _ = self.cal_corr_compete_trials(EEG_test, Att_test, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len, Y_unatt=Unatt_test) 
            else: 
                corr_att_eeg_i, corr_unatt_eeg_i, corr_att_unatt_i, _, eeg_trans_trials, att_trans_trials, unatt_trans_trials = self.cal_corr_compete_trials_mask(EEG_test, Att_test, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len, Y_unatt=Unatt_test)
                eeg_t_all_trials = eeg_t_all_trials + eeg_trans_trials
                att_t_all_trials = att_t_all_trials + att_trans_trials
                unatt_t_all_trials = unatt_t_all_trials + unatt_trans_trials
            corr_att_eeg.append(corr_att_eeg_i)
            corr_unatt_eeg.append(corr_unatt_eeg_i)
            corr_att_unatt.append(corr_att_unatt_i)
        corr_att_eeg = np.concatenate(tuple(corr_att_eeg), axis=0)
        corr_unatt_eeg = np.concatenate(tuple(corr_unatt_eeg), axis=0)
        corr_att_unatt = np.concatenate(tuple(corr_att_unatt), axis=0)
        if PERMU_TEST:
            assert self.REGFEATS == False, "Permutation test is not defined for the regression case."
            acc_permu_list, avgcorr_permu_list = self.permutation_test_acc(eeg_t_all_trials, att_t_all_trials, unatt_t_all_trials)
        else:
            acc_permu_list, avgcorr_permu_list = None, None
        return corr_att_eeg, corr_unatt_eeg, corr_att_unatt, acc_permu_list, avgcorr_permu_list

    def VAD_MM_LVO(self, feat_unatt_list, trial_len, BOOTSTRAP=True, V_eeg=None, V_Stim=None, PERMU_TEST=False):
        '''
        Visual attention decoding task with leave-one-pair-out
        Always train on attended data and try to decode the attended object
        '''
        feat_att_list = self.Stim_list
        nb_videos = len(feat_att_list)
        nb_folds = nb_videos//self.leave_out
        assert nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
        nested_datalist = [self.EEG_list, feat_att_list, feat_unatt_list, self.mask_list] if self.mask_list is not None else [self.EEG_list, feat_att_list, feat_unatt_list]
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO(nested_datalist, self.leave_out)
        assert len(train_list_folds) == len(test_list_folds) == nb_folds, "The number of folds is not correct."
        corr_att_eeg_vad = []
        corr_unatt_eeg = []
        corr_att_eeg_mm = []
        corr_mismatch_eeg = []
        corr_att_unatt = []
        corr_att_mismatch = []
        eeg_t_all_trials = []
        att_t_all_trials = []
        unatt_t_all_trials = []
        for idx in range(0, nb_folds):
            if self.mask_list is not None:
                [EEG_train, Att_train, _, self.mask_train], [EEG_test, Att_test, Unatt_test, self.mask_test] = train_list_folds[idx], test_list_folds[idx]
            else:
                [EEG_train,  Att_train, _], [EEG_test, Att_test, Unatt_test] = train_list_folds[idx], test_list_folds[idx]
            if V_eeg is None:
                _, _, _, _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Att_train)
            else:
                V_eeg_train, V_feat_train = V_eeg, V_Stim
            if self.REGFEATS:
                corr_att_eeg_vad_i, corr_unatt_eeg_i, corr_att_unatt_i, start_points = self.cal_corr_compete_trials(EEG_test, Att_test, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len, Y_unatt=Unatt_test)
                corr_att_eeg_mm_i, corr_mismatch_eeg_i, corr_att_mm_i, _ = self.cal_corr_compete_trials(EEG_test, Att_test, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len, given_start_points=start_points)
            else:
                corr_att_eeg_vad_i, corr_unatt_eeg_i, corr_att_unatt_i, start_points, eeg_trans_trials, att_trans_trials, unatt_trans_trials = self.cal_corr_compete_trials_mask(EEG_test, Att_test, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len, Y_unatt=Unatt_test)
                corr_att_eeg_mm_i, corr_mismatch_eeg_i, corr_att_mm_i, _, _, _, _ = self.cal_corr_compete_trials_mask(EEG_test, Att_test, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len, given_start_points=start_points)
                eeg_t_all_trials = eeg_t_all_trials + eeg_trans_trials
                att_t_all_trials = att_t_all_trials + att_trans_trials
                unatt_t_all_trials = unatt_t_all_trials + unatt_trans_trials
            corr_att_eeg_vad.append(corr_att_eeg_vad_i)
            corr_unatt_eeg.append(corr_unatt_eeg_i)
            corr_att_unatt.append(corr_att_unatt_i)
            corr_att_eeg_mm.append(corr_att_eeg_mm_i)
            corr_mismatch_eeg.append(corr_mismatch_eeg_i)
            corr_att_mismatch.append(corr_att_mm_i)
        corr_att_eeg_vad = np.concatenate(tuple(corr_att_eeg_vad), axis=0)
        corr_unatt_eeg = np.concatenate(tuple(corr_unatt_eeg), axis=0)
        corr_att_unatt = np.concatenate(tuple(corr_att_unatt), axis=0)
        corr_att_eeg_mm = np.concatenate(tuple(corr_att_eeg_mm), axis=0)
        corr_mismatch_eeg = np.concatenate(tuple(corr_mismatch_eeg), axis=0)
        corr_att_mismatch = np.concatenate(tuple(corr_att_mismatch), axis=0)
        if PERMU_TEST:
            assert self.REGFEATS == False, "Permutation test is not defined for the regression case."
            acc_permu_list, avgcorr_permu_list = self.permutation_test_acc(eeg_t_all_trials, att_t_all_trials, unatt_t_all_trials)
        else:
            acc_permu_list, avgcorr_permu_list = None, None
        return corr_att_eeg_vad, corr_unatt_eeg, corr_att_unatt, corr_att_eeg_mm, corr_mismatch_eeg, corr_att_mismatch, acc_permu_list, avgcorr_permu_list

    def VAD_aug_subj_indpd(self, aug_data_list, aug_feat_att_list, aug_mask_list, feat_unatt_list, trial_len, BOOTSTRAP=True):
        '''
        Visual attention decoding task with leave-one-pair-out
        Augment dataset with single-object data
        Train in subject-independent manner by concatenating the data of all subjects
        '''
        # data are 3D (containing multiple subjects)
        nb_subj = aug_data_list[0].shape[2]
        aug_data = np.concatenate(tuple(aug_data_list), axis=0)
        aug_feat = np.concatenate(tuple(aug_feat_att_list), axis=0)
        aug_mask = np.concatenate(tuple(aug_mask_list), axis=0)
        test_data = np.concatenate(tuple(self.EEG_list), axis=0)
        test_att_feat = np.concatenate(tuple(self.Stim_list), axis=0)
        test_unatt_feat = np.concatenate(tuple(feat_unatt_list), axis=0)
        test_mask = np.concatenate(tuple(self.mask_list), axis=0)

        # Initialize dictionaries to store the correlation coefficients, with keys the subject index and values None
        corr_att_eeg ={subj: None for subj in range(nb_subj)}
        corr_unatt_eeg = {subj: None for subj in range(nb_subj)}

        EEG_train_aug = np.concatenate(tuple([aug_data[:,:,i] for i in range(nb_subj)]), axis=0)
        Att_train_aug = np.concatenate([aug_feat[:,:,i] for i in range(nb_subj)], axis=0) if aug_feat.ndim == 3 else np.concatenate([aug_feat for i in range(nb_subj)], axis=0)
        self.mask_train = np.concatenate([aug_mask[:,:,i] for i in range(nb_subj)], axis=0)
        _, _, _, _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train_aug, Att_train_aug)
        for subj in range(nb_subj):
            EEG_test_subj = test_data[:,:,subj]
            Att_test_subj = test_att_feat[:,:,subj] if test_att_feat.ndim == 3 else test_att_feat
            Unatt_test_subj = test_unatt_feat[:,:,subj] if test_unatt_feat.ndim == 3 else test_unatt_feat
            self.mask_test = test_mask[:,:,subj]
            corr_att_trials, corr_unatt_trials, _, _, _, _, _ = self.cal_corr_compete_trials_mask(EEG_test_subj, Att_test_subj, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len, Y_unatt=Unatt_test_subj, BTfactor=1)
            # change the values of the dictionaries
            corr_att_eeg[subj] = corr_att_trials if corr_att_eeg[subj] is None else np.concatenate((corr_att_eeg[subj], corr_att_trials), axis=0)
            corr_unatt_eeg[subj] = corr_unatt_trials if corr_unatt_eeg[subj] is None else np.concatenate((corr_unatt_eeg[subj], corr_unatt_trials), axis=0)
        return corr_att_eeg, corr_unatt_eeg


class DiscriminativeCCA:
    # Not used in analysis
    def __init__(self, signal_list, compete_list, target_list, fs, para_signal, para_compete, para_target, fold=None, leave_out=2, n_components=5, regularization='lwcov', message=True, signifi_level=True, n_permu=500, p_value=0.05):
        '''
        signal_list: list of signals to be maximally correlated with target signals, each element is a T(#sample)xDs(#channel) array
        compete_list: list of compete signals to be minimally correlated with target signals, each element is a T(#sample)xDc(#channel) array
        target_list: list of target signals, each element is a T(#sample)xDt(#channel) array
        fs: Sampling rate
        para_signal/para_compete/para_target: Parameters for spatial-temporal filter
        n_components: number of components to be returned
        regularization: regularization of the estimated covariance matrix
        message: If print message
        signifi_level: If calculate significance level
        n_permu: Number of permutations for significance level calculation in each fold
        p_value: P-value for significance level calculation
        '''
        self.signal_list = signal_list
        self.compete_list = compete_list
        self.target_list = target_list
        self.fs = fs
        self.para_signal = para_signal
        self.para_compete = para_compete
        self.para_target = para_target
        self.fold = fold
        self.leave_out = leave_out
        assert (leave_out is not None) ^ (fold is not None), "Only one of leave_out and fold should be not None"
        self.train_list_folds, self.test_list_folds = utils.split_mm_balance_folds([signal_list, compete_list, target_list], self.fold) if self.fold is not None else utils.split_multi_mod_LVO([signal_list, compete_list, target_list], self.leave_out)
        self.n_components = n_components
        self.regularization = regularization
        self.message = message
        self.signifi_level = signifi_level
        self.n_permu = n_permu
        self.p_value = p_value
        self.mask_train = None
        self.mask_test = None

    def fit(self, Xs, Xc, Xy):
        T = Xs.shape[0]
        mtx_Xs = utils.block_Hankel(Xs, self.para_signal[0], self.para_signal[1])
        mtx_Xc = utils.block_Hankel(Xc, self.para_compete[0], self.para_compete[1])
        mtx_Xd = mtx_Xs - mtx_Xc
        mtx_Xy = utils.block_Hankel(Xy, self.para_target[0], self.para_target[1])
        X = np.concatenate((mtx_Xd, mtx_Xy), axis=1)
        [Dd_H, Dy_H] = [mtx_Xd.shape[1], mtx_Xy.shape[1]]
        Rxx, _ = utils.get_cov_mtx(X, [Dd_H, Dy_H], self.regularization)
        Rss, _ = utils.get_cov_mtx(mtx_Xs, [Dd_H], self.regularization)
        Rcc, _ = utils.get_cov_mtx(mtx_Xc, [Dd_H], self.regularization)
        Dxx = np.zeros_like(Rxx)
        Dxx[:Dd_H,:Dd_H] = Rss - Rcc
        Dxx[Dd_H:,Dd_H:] = Rxx[Dd_H:,Dd_H:]
        # find the top-K eigenvalues
        lam, W = utils.geig_sorted(Rxx, Dxx, self.n_components, option='descending')
        scale_mtx = sqrtm(np.diag(lam)@LA.inv(W.T@Dxx@W*T))
        W = W @ scale_mtx
        Ws = W[:Dd_H,:]
        Wy = W[Dd_H:,:]
        Xs_trans, Xc_trans, Xy_trans = self.get_transformed_data(Xs, Xc, Xy, Ws, Ws, Wy)
        S = (Xs_trans + Xy_trans - Xc_trans) @ np.diag(1/lam)
        return Ws, Wy, S

    def fit_hetero(self, Xs, Xc, Xy):
        T = Xs.shape[0]
        mtx_Xs = utils.block_Hankel(Xs, self.para_signal[0], self.para_signal[1])
        mtx_Xc = utils.block_Hankel(Xc, self.para_compete[0], self.para_compete[1])
        mtx_Xy = utils.block_Hankel(Xy, self.para_target[0], self.para_target[1])
        X = np.concatenate((mtx_Xs, mtx_Xy, mtx_Xc), axis=1)
        [Ds_H, Dy_H, Dc_H] = [mtx_Xs.shape[1], mtx_Xy.shape[1], mtx_Xc.shape[1]]
        X_eqv = np.concatenate((mtx_Xs, mtx_Xy, mtx_Xc*(-1)), axis=1)
        Rxx, Dxx = utils.get_cov_mtx(X_eqv, [Ds_H, Dc_H, Dy_H], self.regularization)
        Dxx[Ds_H+Dy_H:,Ds_H+Dy_H:] = -1*Dxx[Ds_H+Dy_H:,Ds_H+Dy_H:]
        # find the top-K eigenvalues
        lam, W = utils.geig_sorted(Rxx, Dxx, self.n_components, option='descending')
        scale_mtx = sqrtm(np.diag(lam)@LA.inv(W.T@Dxx@W*T))
        W = W @ scale_mtx
        Ws = W[:Ds_H,:]
        Wy = W[Ds_H:Ds_H+Dy_H,:]
        Wc = W[Ds_H+Dy_H:,:]
        Xs_trans, Xc_trans, Xy_trans = self.get_transformed_data(Xs, Xc, Xy, Ws, Wc, Wy)
        S = (Xs_trans + Xy_trans - Xc_trans) @ np.diag(1/lam)
        return Ws, Wc, Wy, S

    def get_transformed_data(self, Xs, Xc, Xy, Ws, Wc, Wy):
        '''
        Get the transformed data
        '''
        mtx_Xs = utils.block_Hankel(Xs, self.para_signal[0], self.para_signal[1])
        mtx_Xc = utils.block_Hankel(Xc, self.para_compete[0], self.para_compete[1])
        mtx_Xy = utils.block_Hankel(Xy, self.para_target[0], self.para_target[1])
        mtx_Xs_centered = mtx_Xs - np.mean(mtx_Xs, axis=0, keepdims=True)
        mtx_Xc_centered = mtx_Xc - np.mean(mtx_Xc, axis=0, keepdims=True)
        mtx_Xy_centered = mtx_Xy - np.mean(mtx_Xy, axis=0, keepdims=True)
        Xs_trans = mtx_Xs_centered @ Ws
        Xc_trans = mtx_Xc_centered @ Wc
        Xy_trans = mtx_Xy_centered @ Wy
        return Xs_trans, Xc_trans, Xy_trans

    def get_corr_coe(self, X_trans, Y_trans):
        # corr_pvalue = [pearsonr(X_trans[:,k], Y_trans[:,k]) for k in range(self.n_components)]
        # corr_coe = np.array([corr_pvalue[k][0] for k in range(self.n_components)])
        # p_value = np.array([corr_pvalue[k][1] for k in range(self.n_components)])
        corr_coe = np.array(np.corrcoef(X_trans[:,k], Y_trans[:,k])[0,1] for k in range(self.n_components))
        p_value = None
        return corr_coe, p_value

    def cross_val(self):
        nb_folds = len(self.train_list_folds)
        corr_sy_train_fold = np.zeros((nb_folds, self.n_components))
        corr_cy_train_fold = np.zeros((nb_folds, self.n_components))
        corr_sy_test_fold = np.zeros((nb_folds, self.n_components))
        corr_cy_test_fold = np.zeros((nb_folds, self.n_components))
        for idx in range(nb_folds):
            [signal_train, compete_train, target_train], [signal_test, compete_test, target_test] = self.train_list_folds[idx], self.test_list_folds[idx]
            Ws, Wy, _ = self.fit(signal_train, compete_train, target_train)
            Wc = Ws
            # Ws, Wc, Wy, _ = self.fit_hetero(signal_train, compete_train, target_train)
            Xs_trans_train, Xc_trans_train, Xy_trans_train = self.get_transformed_data(signal_train, compete_train, target_train, Ws, Wc, Wy)
            corr_sy_train_fold[idx,:], _ = self.get_corr_coe(Xs_trans_train, Xy_trans_train)
            corr_cy_train_fold[idx,:], _ = self.get_corr_coe(Xc_trans_train, Xy_trans_train)
            # Use Ws for both Xs and Xc because we don't know which one is the attended signal in the test set
            Xs_trans_test, Xc_trans_test, Xy_trans_test = self.get_transformed_data(signal_test, compete_test, target_test, Ws, Ws, Wy) 
            corr_sy_test_fold[idx,:], _ = self.get_corr_coe(Xs_trans_test, Xy_trans_test)
            corr_cy_test_fold[idx,:], _ = self.get_corr_coe(Xc_trans_test, Xy_trans_test)
        if self.message:
            print('Average correlation coefficients of the top {} components on the training sets (signal): {}'.format(self.n_components, np.average(corr_sy_train_fold, axis=0)))
            print('Average correlation coefficients of the top {} components on the training sets (compete): {}'.format(self.n_components, np.average(corr_cy_train_fold, axis=0)))
            print('Average correlation coefficients of the top {} components on the test sets (signal): {}'.format(self.n_components, np.average(corr_sy_test_fold, axis=0)))
            print('Average correlation coefficients of the top {} components on the test sets (compete): {}'.format(self.n_components, np.average(corr_cy_test_fold, axis=0)))
        return corr_sy_train_fold, corr_cy_train_fold, corr_sy_test_fold, corr_cy_test_fold

    def att_or_unatt_trials(self, trial_len, BOOTSTRAP=True):
        nb_folds = len(self.train_list_folds)
        corr_signal = []
        corr_compete = []
        for idx in range(nb_folds):
            [signal_train, compete_train, target_train], [signal_test, compete_test, target_test] = self.train_list_folds[idx], self.test_list_folds[idx]
            Ws, Wy, _ = self.fit(signal_train, compete_train, target_train)
            # Ws, Wc, Wy, _ = self.fit_hetero(signal_train, compete_train, target_train)
            Xs_trans, Xc_trans, Xy_trans = self.get_transformed_data(signal_test, compete_test, target_test, Ws, Ws, Wy)
            if Xs_trans.shape[0]-trial_len*self.fs >= 0:
                if BOOTSTRAP:
                    nb_trials = Xs_trans.shape[0]//self.fs//3
                    start_points = np.random.randint(0, Xs_trans.shape[0]-trial_len*self.fs, size=nb_trials)
                    start_points = np.sort(start_points)
                else:
                    start_points = np.array(range(0, Xs_trans.shape[0] - Xs_trans.shape[0]%(self.fs*trial_len), self.fs*trial_len))
                Xy_trans_trials = utils.into_trials(Xy_trans, self.fs, trial_len, start_points=start_points)
                Xs_trans_trials = utils.into_trials(Xs_trans, self.fs, trial_len, start_points=start_points)
                Xc_trans_trials = utils.into_trials(Xc_trans, self.fs, trial_len, start_points=start_points)
                corr_signal_i = np.stack([self.get_corr_coe(Xy, Xs)[0] for Xy, Xs in zip(Xy_trans_trials, Xs_trans_trials)])
                corr_compete_i = np.stack([self.get_corr_coe(Xy, Xc)[0] for Xy, Xc in zip(Xy_trans_trials, Xc_trans_trials)])
                corr_signal.append(corr_signal_i)
                corr_compete.append(corr_compete_i)
            else:
                print('The length of the video is too short for the given trial length.')
                # return NaN values
                corr_signal.append(np.full((1, self.n_components), np.nan))
                corr_compete.append(np.full((1, self.n_components), np.nan))
        corr_signal = np.concatenate(tuple(corr_signal), axis=0)
        corr_compete = np.concatenate(tuple(corr_compete), axis=0)
        return corr_signal, corr_compete
    

class GeneralizedCCA:
    '''
    Perform GCCA on data of different subjects. If subjects have multi-modal data, then the data are concatenated along the channel axis.
    '''
    def __init__(self, EEG_list, fs, L, offset, hankelized=False, dim_list=None, fold=10, leave_out=2, n_components=5, regularization='lwcov', message=True, signifi_level=True, n_permu=500, p_value=0.05, dim_subspace=4, save_W_perfold=False, crs_val=True):
        '''
        EEG_list: list of EEG data, each element is a T(#sample)xDx(#channel)xN(#subj) array corresponding to a video 
        fs: Sampling rate
        L: If use (spatial-) temporal filter, the number of taps
        offset: If use (spatial-) temporal filter, the offset of the time lags
        hankelized: If the data is already hankelized because of regression
        dim_list: If 'EEG' contains data from multiple sources that have significantly different scales, then specify the dimensions of each source
        fold: Number of folds for cross-validation
        leave_out: Number of pairs to leave out for leave-one-pair-out cross-validation
        n_components: Number of components to be returned
        regularization: Regularization of the estimated covariance matrix
        message: If print message
        signifi_level: If calculate significance level
        pool: If pool the significance level of all components
        n_permu: Number of permutations for significance level calculation
        p_value: P-value for significance level calculation
        dim_subspace: Dimension of the subspace for calculating TSC
        save_W_perfold: If save the weights per fold
        crs_val: If perform cross-validation
        '''
        self.EEG_list = EEG_list
        self.fs = fs
        self.L = L
        self.offset = offset
        self.hankelized = hankelized
        self.dim_list = dim_list
        self.fold = fold
        self.leave_out = leave_out
        self.n_components = n_components
        self.regularization = regularization
        self.message = message
        self.signifi_level = signifi_level
        self.n_permu = n_permu
        self.p_value = p_value
        self.dim_subspace = dim_subspace
        self.save_W_perfold = save_W_perfold
        if self.save_W_perfold:
            self.test_list = []
            self.W_train_list = []
        self.crs_val = crs_val

    def fit(self, X_stack):
        '''
        Inputs:
        X_stack: stacked (along axis 2) data of different subjects
        Outputs:
        W_stack: weights with shape DLxNxn_components
        S: shared subspace with shape Txn_components
        F_stack: forward model with shape Dxn_components calculated from the shared subspace in the training set
        lam: eigenvalues, related to mean squared error (not used)
        '''
        T, D, N = X_stack.shape
        L = self.L
        dim_list_extended = [d*L for d in self.dim_list]*N if self.dim_list is not None else [D*L]*N
        # From [X1; X2; ... XN] to [X1 X2 ... XN]
        # each column represents a variable, while the rows contain observations
        X_list = [utils.block_Hankel(X_stack[:,:,n], L, self.offset) for n in range(N)]
        X = np.concatenate(tuple(X_list), axis=1)
        X_center = X - np.mean(X, axis=0, keepdims=True)
        Rxx, _ = utils.get_cov_mtx(X, dim_list_extended, self.regularization)
        Dxx = np.zeros_like(Rxx)
        for n in range(N):
            Dxx[n*D*L:(n+1)*D*L, n*D*L:(n+1)*D*L] = Rxx[n*D*L:(n+1)*D*L, n*D*L:(n+1)*D*L]
        lam, W = eigh(Dxx, Rxx, subset_by_index=[0,self.n_components-1]) # automatically ascend
        Lam = np.diag(lam)
        # Right scaling
        W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rxx * T @ W @ Lam))
        # Shared subspace
        S = X_center@W@Lam
        # Forward models
        F_redun = T * Dxx @ W
        # Reshape W as (DL*n_components*N)
        W_stack = np.reshape(W, (N,D*L,-1))
        W_stack = np.transpose(W_stack, [1,0,2])
        F_redun_stack = np.reshape(F_redun, (N,D*L,-1))
        F_redun_stack = np.transpose(F_redun_stack, [1,0,2])
        F_stack = utils.F_organize(F_redun_stack, L, self.offset, avg=True)
        return W_stack, S, F_stack, lam
    
    def forward_model(self, EEG, W_stack, S=None):
        '''
        Calculate the forward model from the transformed EEG data or shared subspace
        Input:
        EEG: EEG data with shape (T, D, N) [T: # sample, D: # channel, N: # subjects]
        W: weights with shape (DL, n_components)
        S: shared subspace with shape (T, n_components); if not None, then calculate forward model from the shared subspace
        Outputs:
        F: forward model (D, n_components)
        '''
        _, _, N = EEG.shape
        X_list = [utils.block_Hankel(EEG[:,:,n], self.L, self.offset) for n in range(N)]
        X_list_center = [X_list[n] - np.mean(X_list[n], axis=0, keepdims=True) for n in range(N)]
        X_stack = np.stack(X_list_center, axis=2)
        if S is not None:
            F_T = np.mean(np.einsum('kt, tdn -> kdn', S.T, X_stack), axis=2)
            F_redun = F_T.T
        else:
            X = np.concatenate(tuple(X_list_center), axis=0)
            X_list_trans = [X_stack[:,:,n]@W_stack[:,n,:] for n in range(N)]
            X_transformed = np.concatenate(tuple(X_list_trans), axis=0)
            F_redun = (lstsq(X_transformed, X)[0]).T
        F = utils.F_organize(F_redun, self.L, self.offset)
        return F

    def avg_stats(self, X_stack, W_stack):
        '''
        Calculate the pairwise average statistics.
        Inputs:
        X_stack: stacked (along axis 2) data of different subjects
        W_stack: weights 1) dim(W)=2: results of correlated component analysis 2) dim(W)=3: results of GCCA
        Output:
        avg_corr: pairwise average correlation
        avg_cov: pairwise average covariance
        avg_ChDist: pairwise average Chordal distance
        avg_TSC: pairwise average total squared correlation
        '''
        _, _, N = X_stack.shape
        n_components = self.n_components
        Hankellist = [np.expand_dims(utils.block_Hankel(X_stack[:,:,n], self.L, self.offset), axis=2) for n in range(N)]
        X_stack = np.concatenate(tuple(Hankellist), axis=2)
        corr_mtx_stack = np.zeros((N,N,n_components))
        cov_mtx_stack = np.zeros((N,N,n_components))
        avg_corr = np.zeros(n_components)
        avg_cov = np.zeros(n_components)
        if np.ndim (W_stack) == 2: # for correlated component analysis
            W_stack = np.expand_dims(W_stack, axis=1)
            W_stack = np.repeat(W_stack, N, axis=1)
        for component in range(n_components):
            w = W_stack[:,:,component]
            w = np.expand_dims(w, axis=1)
            X_trans = np.einsum('tdn,dln->tln', X_stack, w)
            X_trans = np.squeeze(X_trans, axis=1)
            corr_mtx_stack[:,:,component] = np.corrcoef(X_trans, rowvar=False)
            cov_mtx_stack[:,:,component] = np.cov(X_trans, rowvar=False)
            avg_corr[component] = np.sum(corr_mtx_stack[:,:,component]-np.eye(N))/N/(N-1)
            avg_cov[component] = (np.sum(cov_mtx_stack[:,:,component])-np.sum(np.diag(cov_mtx_stack[:,:,component])))/N/(N-1)
        Squared_corr = np.sum(np.square(corr_mtx_stack[:,:,:self.dim_subspace]), axis=2)
        avg_TSC = np.sum(Squared_corr-self.dim_subspace*np.eye(N))/N/(N-1)
        Chordal_dist = np.sqrt(self.dim_subspace-Squared_corr)
        avg_ChDist = np.sum(Chordal_dist)/N/(N-1)
        return avg_corr, avg_cov, avg_ChDist, avg_TSC

    def avg_stats_trials(self, X_trials, W_stack):
        stats = [(self.avg_stats(trial, W_stack)) for trial in X_trials]
        avg_corr = np.concatenate(tuple([np.expand_dims(stats[i][0],axis=0) for i in range(len(X_trials))]), axis=0).mean(axis=0)
        avg_cov = np.concatenate(tuple([np.expand_dims(stats[i][1],axis=0) for i in range(len(X_trials))]), axis=0).mean(axis=0)
        avg_ChDist = np.array([stats[i][2] for i in range(len(X_trials))]).mean()
        avg_TSC = np.array([stats[i][3] for i in range(len(X_trials))]).mean()
        return avg_corr, avg_cov, avg_ChDist, avg_TSC

    def get_transformed_data(self, X_stack, W_stack):
        '''
        Get the transformed data
        '''
        _, _, N = X_stack.shape
        if np.ndim (W_stack) == 2: # for correlated component analysis
            W_stack = np.expand_dims(W_stack, axis=1)
            W_stack = np.repeat(W_stack, N, axis=1)
        Hankellist = [np.expand_dims(utils.block_Hankel(X_stack[:,:,n], self.L, self.offset), axis=2) for n in range(N)]
        Hankel_center = [hankel - np.mean(hankel, axis=0, keepdims=True) for hankel in Hankellist]
        X_center = np.concatenate(tuple(Hankel_center), axis=2)
        X_trans = np.einsum('tdn,dkn->tkn', X_center, np.transpose(W_stack, (0,2,1)))
        return X_trans

    def get_avg_corr_coe(self, X_trans):
        '''
        Calculate the inter-subject correlation (average pairwise correlation)
        '''
        _, _, N = X_trans.shape
        n_components = self.n_components
        corr_mtx_stack = np.zeros((N,N,n_components))
        avg_corr = np.zeros(n_components)
        for component in range(n_components):
            corr_mtx_stack[:,:,component] = np.corrcoef(X_trans[:,component,:], rowvar=False)
            avg_corr[component] = np.sum(corr_mtx_stack[:,:,component]-np.eye(N))/N/(N-1)
        return avg_corr

    def permutation_test(self, X_stack, W_stack, PHASE_SCRAMBLE=True, block_len=None):
        corr_coe_topK = np.empty((0, self.n_components))
        X_trans = self.get_transformed_data(X_stack, W_stack)
        for i in tqdm(range(self.n_permu)):
            X_shuffled = utils.shuffle_3D(X_trans, block_len) if not PHASE_SCRAMBLE else utils.phase_scramble_3D(X_trans)
            corr_coe = self.get_avg_corr_coe(X_shuffled)
            corr_coe_topK = np.concatenate((corr_coe_topK, np.expand_dims(corr_coe, axis=0)), axis=0)
        return corr_coe_topK

    def permutation_test_trials(self, X_trials, W_stack, PHASE_SCRAMBLE=True, block_len=None):
        corr_coe_topK = np.empty((0, self.n_components))
        X_trans_trials = [self.get_transformed_data(X_stack, W_stack) for X_stack in X_trials]
        for i in tqdm(range(self.n_permu)):
            X_shuffled_trials = [utils.shuffle_3D(X_stack, block_len) if not PHASE_SCRAMBLE else utils.phase_scramble_3D(X_stack) for X_stack in X_trans_trials]
            corr_coe_trials_list = [self.get_avg_corr_coe(X_stack) for X_stack in X_shuffled_trials]
            corr_coe_trials = np.concatenate(tuple([np.expand_dims(corr_coe, axis=0) for corr_coe in corr_coe_trials_list]), axis=0)
            corr_coe_topK = np.concatenate((corr_coe_topK, np.mean(corr_coe_trials, axis=0, keepdims=True)), axis=0)
        return corr_coe_topK

    def calculate_sig_corr(self, corr_trials, nb_fold=1):
        assert self.n_components*self.n_permu*nb_fold == corr_trials.shape[0]*corr_trials.shape[1]
        sig_idx = -int(self.n_permu*self.p_value*self.n_components*nb_fold)
        corr_trials = np.sort(abs(corr_trials), axis=None)
        return corr_trials[sig_idx]

    def cross_val(self, trial_len=None):
        fold = self.fold
        n_components = self.n_components
        corr_train = np.zeros((fold, n_components))
        corr_test = np.zeros((fold, n_components))
        cov_train = np.zeros((fold, n_components))
        cov_test = np.zeros((fold, n_components))
        tsc_train = np.zeros((fold, 1))
        tsc_test = np.zeros((fold, 1))
        dist_train = np.zeros((fold, 1))
        dist_test = np.zeros((fold, 1))
        for idx in range(fold):
            train_list, test_list, _, _ = utils.split_mm_balance([self.EEG_list], fold=fold, fold_idx=idx+1)
            W_train, _, F_train, _ = self.fit(train_list[0])
            if self.save_W_perfold:
                self.test_list.append(test_list[0])
                self.W_train_list.append(W_train)
            corr_train[idx,:], cov_train[idx,:], dist_train[idx], tsc_train[idx] = self.avg_stats(train_list[0], W_train)
            if trial_len is not None:
                test_trials = utils.into_trials(test_list[0], self.fs, t=trial_len)
                corr_test[idx,:], cov_test[idx,:], dist_test[idx], tsc_test[idx] = self.avg_stats_trials(test_trials, W_train)
            else:
                corr_test[idx,:], cov_test[idx,:], dist_test[idx], tsc_test[idx] = self.avg_stats(test_list[0], W_train)
            if not self.crs_val:
                # fill the rest of the folds with the same results
                for i in range(idx+1, fold):
                    corr_train[i,:] = corr_train[idx,:]
                    cov_train[i,:] = cov_train[idx,:]
                    dist_train[i] = dist_train[idx]
                    tsc_train[i] = tsc_train[idx]
                    corr_test[i,:] = corr_test[idx,:]
                    cov_test[i,:] = cov_test[idx,:]
                    dist_test[i] = dist_test[idx]
                    tsc_test[i] = tsc_test[idx]
                break
        if self.signifi_level:
            if trial_len is not None:
                corr_trials = self.permutation_test_trials(test_trials, W_train)
            else:
                corr_trials = self.permutation_test(test_list[0], W_train)
            corr_trials = np.sort(abs(corr_trials), axis=None)
            sig_idx = -int(self.n_permu*self.p_value*n_components)
            print('Significance level: ISC={}'.format(corr_trials[sig_idx]))

        if self.message:
            print('Average ISC of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
            print('Average ISC of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
        return corr_train, corr_test, cov_train, cov_test, tsc_train, tsc_test, dist_train, dist_test, W_train, F_train
    
    def cross_val_LVO(self, W_eeg=None):
        nb_videos = len(self.EEG_list)
        n_components = self.n_components
        nb_folds = nb_videos//self.leave_out
        assert nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
        nested_datalist = [self.EEG_list]
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO(nested_datalist, self.leave_out)
        assert len(train_list_folds) == len(test_list_folds) == nb_folds, "The number of folds is not correct."
        corr_train_fold = np.zeros((nb_folds, n_components))
        corr_test_fold = np.zeros((nb_folds, n_components))
        corr_permu_fold = []
        forward_model_fold = []
        cov_train_fold = np.zeros((nb_folds, n_components))
        cov_test_fold = np.zeros((nb_folds, n_components))
        tsc_train_fold = np.zeros((nb_folds, 1))
        tsc_test_fold = np.zeros((nb_folds, 1))
        for idx in range(0, nb_folds):
            [EEG_train], [EEG_test] = train_list_folds[idx], test_list_folds[idx]
            if W_eeg is not None:
                W_train = W_eeg
            else:
                W_train, _, _, _ = self.fit(EEG_train)
            corr_train_fold[idx,:], cov_train_fold[idx,:], _, tsc_train_fold[idx] = self.avg_stats(EEG_train, W_train)
            corr_test_fold[idx,:], cov_test_fold[idx,:], _, tsc_test_fold[idx] = self.avg_stats(EEG_test, W_train)
            forward_model = self.forward_model(EEG_test, W_train)
            forward_model_fold.append(forward_model)
            if self.signifi_level:
                corr_permu_fold.append(self.permutation_test(EEG_test, W_train))
        if self.signifi_level:
            sig_corr_fold = [self.calculate_sig_corr(corr_permu) for corr_permu in corr_permu_fold]
            corr_permu_all = np.concatenate(tuple(corr_permu_fold), axis=0)
            sig_corr_pool = self.calculate_sig_corr(corr_permu_all, nb_fold=nb_folds)
        else:
            sig_corr_fold = None
            sig_corr_pool = None
        if self.message:
            print('Average ISC of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train_fold, axis=0)))
            print('Average ISC of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test_fold, axis=0)))
            print('Average ISCov of the top {} components on the test sets: {}'.format(n_components, np.average(cov_test_fold, axis=0)))
            print('Significance level: {}'.format(sig_corr_pool))
        return corr_train_fold, corr_test_fold, cov_train_fold, cov_test_fold, tsc_train_fold, tsc_test_fold, sig_corr_fold, sig_corr_pool, forward_model_fold
    

class GeneralizedCCA_MultiMod:
    '''
    Same as GeneralizedCCA, but accepts multi-modal data not only from different subjects but also from different sources. The first modality is the EEG data.
    '''
    def __init__(self, nested_datalist, fs, L_list, offset_list, fold=10, leave_out=2, n_components=10, regularization='lwcov', message=True, signifi_level=True, n_permu=500, p_value=0.05, dim_subspace=4):
        '''
        nested_datalist: [mod1_list, mod2_list, ...], where modi_list is a list of data from modality i. The elements in modi_list are the data from different videos with dimention T(#sample)xDx(#channel)xN(#subject) or T(#sample)xDx(#channel)
        fs: Sampling rate
        L_list: If use (spatial-) temporal filter, the number of taps for each modality
        offset_list: If use (spatial-) temporal filter, the offset of the time lags for each modality
        fold: Number of folds for cross-validation
        leave_out: Number of pairs to leave out for leave-one-pair-out cross-validation
        n_components: Number of components to be returned
        regularization: Regularization of the estimated covariance matrix
        message: If print message
        signifi_level: If calculate significance level
        n_permu: Number of permutations for significance level calculation
        p_value: P-value for significance level calculation
        dim_subspace: Dimension of the subspace for calculating TSC
        '''
        self.nested_datalist = [[data if np.ndim(data)>1 else np.expand_dims(data, axis=1) for data in datalist] for datalist in nested_datalist]
        self.nested_datalist = [[np.expand_dims(data, axis=2) if np.ndim(data)==2 else data for data in datalist] for datalist in self.nested_datalist]
        self.fs = fs
        self.L_list = L_list
        self.offset_list = offset_list
        self.fold = fold
        self.leave_out = leave_out
        self.n_components = n_components
        self.regularization = regularization
        self.message = message
        self.signifi_level = signifi_level
        self.n_permu = n_permu
        self.p_value = p_value
        self.dim_subspace = dim_subspace

    def fit(self, mm_data):
        '''
        Inputs:
        mm_data: multi-modal data, each element is a T(#sample)xDx(#channel)xN(#subject) array or a T(#sample)xDx(#channel) array 
        Outputs:
        W_list: list of weights corresponding to each modality 
        S: shared subspace with shape Txn_components
        lam: eigenvalues, related to mean squared error (not used in analysis)
        '''
        mm_data_3D = [np.expand_dims(data, axis=2) if np.ndim(data)==2 else data for data in mm_data]
        T, _, _ = mm_data_3D[0].shape
        N_list = [data.shape[2] for data in mm_data_3D]
        mm_hankel_3D = [utils.hankelize_data_multisub(data, L, offset) for data, L, offset in zip(mm_data_3D, self.L_list, self.offset_list)]
        dim_list = [[data.shape[1]]*N for (data,N) in zip(mm_hankel_3D, N_list)]
        dim_list = [dim for sublist in dim_list for dim in sublist]
        mm_hankel_2D = [np.reshape(data, (T,-1), order='F') for data in mm_hankel_3D]
        X = np.concatenate(tuple(mm_hankel_2D), axis=1)
        X_center = X - np.mean(X, axis=0, keepdims=True)
        Rxx, Dxx = utils.get_cov_mtx(X, dim_list, self.regularization)
        lam, W = eigh(Dxx, Rxx, subset_by_index=[0,self.n_components-1]) # automatically ascend
        Lam = np.diag(lam)
        # Right scaling
        W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rxx * T @ W @ Lam))
        # Shared subspace
        S = X_center@W@Lam
        # Reshape W as (DL*n_components*N)
        W_list = utils.W_organize(W, mm_data, self.L_list)
        return W_list, S, lam
    
    def forward_model(self, EEG_3D, W_EEG_3D):
        T, _, N = EEG_3D.shape
        EEG_trans_3D = self.get_transformed_data(EEG_3D, W_EEG_3D, self.L_list[0], self.offset_list[0])
        EEG_trans_2D = np.reshape(np.transpose(EEG_trans_3D, (0,2,1)), (T*N,-1), order='F') # concatenate multi-subject data along the time axis
        EEG_2D = np.reshape(np.transpose(EEG_3D, (0,2,1)), (T*N,-1), order='F')
        F = (lstsq(EEG_trans_2D, EEG_2D)[0]).T
        return F

    def avg_stats(self, mm_data, W_list):
        mm_trans_3D = [self.get_transformed_data(data, W, L, offset) for data, W, L, offset in zip(mm_data, W_list, self.L_list, self.offset_list)]
        mm_trans_3D_concat = np.concatenate(tuple(mm_trans_3D), axis=2)
        _, _, N = mm_trans_3D_concat.shape
        n_components = self.n_components
        corr_mtx_stack = np.zeros((N,N,n_components))
        cov_mtx_stack = np.zeros((N,N,n_components))
        avg_corr = np.zeros(n_components)
        avg_cov = np.zeros(n_components)
        for component in range(n_components):
            X_trans = mm_trans_3D_concat[:,component,:]
            corr_mtx_stack[:,:,component] = np.corrcoef(X_trans, rowvar=False)
            cov_mtx_stack[:,:,component] = np.cov(X_trans, rowvar=False)
            avg_corr[component] = np.sum(corr_mtx_stack[:,:,component]-np.eye(N))/N/(N-1)
            avg_cov[component] = (np.sum(cov_mtx_stack[:,:,component])-np.sum(np.diag(cov_mtx_stack[:,:,component])))/N/(N-1)
        Squared_corr = np.sum(np.square(corr_mtx_stack[:,:,:self.dim_subspace]), axis=2)
        avg_TSC = np.sum(Squared_corr-self.dim_subspace*np.eye(N))/N/(N-1)
        Chordal_dist = np.sqrt(self.dim_subspace-Squared_corr)
        avg_ChDist = np.sum(Chordal_dist)/N/(N-1)
        return avg_corr, avg_cov, avg_ChDist, avg_TSC

    def get_transformed_data(self, data_3D, W_3D, L, offset):
        data_hankel_3D = utils.hankelize_data_multisub(data_3D, L, offset)
        data_center_3D = data_hankel_3D - np.mean(data_hankel_3D, axis=0, keepdims=True)
        data_trans_3D = np.einsum('tdn,dkn->tkn', data_center_3D, np.transpose(W_3D, (0,2,1)))
        return data_trans_3D

    def get_avg_corr_coe(self, EEG_trans_3D):
        _, _, N = EEG_trans_3D.shape
        n_components = self.n_components
        corr_mtx_stack = np.zeros((N,N,n_components))
        avg_corr = np.zeros(n_components)
        for component in range(n_components):
            corr_mtx_stack[:,:,component] = np.corrcoef(EEG_trans_3D[:,component,:], rowvar=False)
            avg_corr[component] = np.sum(corr_mtx_stack[:,:,component]-np.eye(N))/N/(N-1)
        return avg_corr

    def permutation_test(self, EEG_3D, W_EEG_3D, PHASE_SCRAMBLE=True, block_len=None):
        corr_coe_topK = np.empty((0, self.n_components))
        EEG_trans = self.get_transformed_data(EEG_3D, W_EEG_3D, self.L_list[0], self.offset_list[0])
        for i in tqdm(range(self.n_permu)):
            X_shuffled = utils.shuffle_3D(EEG_trans, block_len) if not PHASE_SCRAMBLE else utils.phase_scramble_3D(EEG_trans)
            corr_coe = self.get_avg_corr_coe(X_shuffled)
            corr_coe_topK = np.concatenate((corr_coe_topK, np.expand_dims(corr_coe, axis=0)), axis=0)
        return corr_coe_topK

    def calculate_sig_corr(self, corr_trials, nb_fold=1):
        assert self.n_components*self.n_permu*nb_fold == corr_trials.shape[0]*corr_trials.shape[1]
        sig_idx = -int(self.n_permu*self.p_value*self.n_components*nb_fold)
        corr_trials = np.sort(abs(corr_trials), axis=None)
        return corr_trials[sig_idx]
    
    def cross_val(self):
        fold = self.fold
        n_components = self.n_components
        corr_all_train = np.zeros((fold, n_components))
        corr_all_test = np.zeros((fold, n_components))
        cov_all_test = np.zeros((fold, n_components))
        corr_eeg_train = np.zeros((fold, n_components))
        corr_eeg_test = np.zeros((fold, n_components))
        cov_eeg_test = np.zeros((fold, n_components))
        corr_permu_fold = []
        forward_model_fold = []
        for idx in range(fold):
            train_list, test_list, _, _ = utils.split_mm_balance(self.nested_datalist, fold=fold, fold_idx=idx+1)
            W_train_list, _, _ = self.fit(train_list)
            forward_model = self.forward_model(test_list[0], W_train_list[0])
            forward_model_fold.append(forward_model)
            corr_all_train[idx,:], _, _, _ = self.avg_stats(train_list, W_train_list)
            corr_eeg_train[idx,:], _, _, _ = self.avg_stats([train_list[0]], [W_train_list[0]])
            corr_all_test[idx,:], cov_all_test[idx,:], _, _ = self.avg_stats(test_list, W_train_list)
            corr_eeg_test[idx,:], cov_eeg_test[idx,:], _, _ = self.avg_stats([test_list[0]], [W_train_list[0]])
            corr_permu_fold.append(self.permutation_test(test_list[0], W_train_list[0]))
        sig_corr_fold = [self.calculate_sig_corr(corr_permu) for corr_permu in corr_permu_fold]
        corr_permu_all = np.concatenate(tuple(corr_permu_fold), axis=0)
        sig_corr_pool = self.calculate_sig_corr(corr_permu_all, nb_fold=fold)
        if self.message:
            print('Average IMC of the top {} components on the training sets: {}'.format(n_components, np.average(corr_all_train, axis=0)))
            print('Average IMC of the top {} components on the test sets: {}'.format(n_components, np.average(corr_all_test, axis=0)))
            print('Average ISC of the top {} components on the training sets: {}'.format(n_components, np.average(corr_eeg_train, axis=0)))
            print('Average ISC of the top {} components on the test sets: {}'.format(n_components, np.average(corr_eeg_test, axis=0)))
            print('Significance level: ISC={}'.format(sig_corr_pool))
        return corr_all_test, cov_all_test, corr_eeg_test, cov_eeg_test, sig_corr_fold, sig_corr_pool, forward_model_fold
    
    def cross_val_LVO(self):
        nb_videos = len(self.nested_datalist[0])
        n_components = self.n_components
        nb_folds = nb_videos//self.leave_out
        assert nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO(self.nested_datalist, self.leave_out)
        assert len(train_list_folds) == len(test_list_folds) == nb_folds, "The number of folds is not correct."
        corr_all_train = np.zeros((nb_folds, n_components))
        corr_all_test = np.zeros((nb_folds, n_components))
        cov_all_test = np.zeros((nb_folds, n_components))
        corr_eeg_train = np.zeros((nb_folds, n_components))
        corr_eeg_test = np.zeros((nb_folds, n_components))
        cov_eeg_test = np.zeros((nb_folds, n_components))
        corr_permu_fold = []
        forward_model_fold = []
        for idx in range(0, nb_folds):
            train_list, test_list = train_list_folds[idx], test_list_folds[idx]
            W_train_list, _, _ = self.fit(train_list)
            forward_model = self.forward_model(test_list[0], W_train_list[0])
            forward_model_fold.append(forward_model)
            corr_all_train[idx,:], _, _, _ = self.avg_stats(train_list, W_train_list)
            corr_eeg_train[idx,:], _, _, _ = self.avg_stats([train_list[0]], [W_train_list[0]])
            corr_all_test[idx,:], cov_all_test[idx,:], _, _ = self.avg_stats(test_list, W_train_list)
            corr_eeg_test[idx,:], cov_eeg_test[idx,:], _, _ = self.avg_stats([test_list[0]], [W_train_list[0]])
            corr_permu_fold.append(self.permutation_test(test_list[0], W_train_list[0]))
        sig_corr_fold = [self.calculate_sig_corr(corr_permu) for corr_permu in corr_permu_fold]
        corr_permu_all = np.concatenate(tuple(corr_permu_fold), axis=0)
        sig_corr_pool = self.calculate_sig_corr(corr_permu_all, nb_fold=nb_folds)
        if self.message:
            print('Average IMC of the top {} components on the training sets: {}'.format(n_components, np.average(corr_all_train, axis=0)))
            print('Average IMC of the top {} components on the test sets: {}'.format(n_components, np.average(corr_all_test, axis=0)))
            print('Average ISC of the top {} components on the training sets: {}'.format(n_components, np.average(corr_eeg_train, axis=0)))
            print('Average ISC of the top {} components on the test sets: {}'.format(n_components, np.average(corr_eeg_test, axis=0)))
            print('Significance level: ISC={}'.format(sig_corr_pool))
        return corr_all_test, cov_all_test, corr_eeg_test, cov_eeg_test, sig_corr_fold, sig_corr_pool, forward_model_fold

    def att_or_unatt_LVO_trials(self, trial_len):
        '''
        When nested_datalist contains EEG, attended features, and unattended features, this function performs the visual attention decoding task based on ISC values. [Not used in the paper]
        '''
        nb_videos = len(self.nested_datalist[0])
        nb_folds = nb_videos//self.leave_out
        assert nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO(self.nested_datalist, self.leave_out)
        assert len(train_list_folds) == len(test_list_folds) == nb_folds, "The number of folds is not correct."
        isc_att = []
        isc_unatt = []

        for idx in range(0, nb_folds):
            data_mm_train, Att_train, _ = train_list_folds[idx][:-2], train_list_folds[idx][-2], train_list_folds[idx][-1]
            data_mm_test, Att_test, Unatt_test = test_list_folds[idx][:-2], test_list_folds[idx][-2], test_list_folds[idx][-1]
            W_list, _, _ = self.fit(data_mm_train+[Att_train])

            if Att_test.shape[0]-trial_len*self.fs >= 0:
                nb_trials = Att_test.shape[0]//self.fs//3
                start_points = np.random.randint(0, Att_test.shape[0]-trial_len*self.fs, size=nb_trials)
                start_points = np.sort(start_points)
                data_mm_trials = [utils.into_trials(data, self.fs, trial_len, start_points=start_points) for data in data_mm_test]
                Att_trials = utils.into_trials(Att_test, self.fs, trial_len, start_points=start_points)
                Unatt_trials = utils.into_trials(Unatt_test, self.fs, trial_len, start_points=start_points)
                isc_att_i = [self.avg_stats([data[i] for data in data_mm_trials+[Att_trials]], W_list)[0] for i in range(len(Att_trials))]
                isc_unatt_i = [self.avg_stats([data[i] for data in data_mm_trials+[Unatt_trials]], W_list)[0] for i in range(len(Unatt_trials))]
                isc_att.append(isc_att_i)
                isc_unatt.append(isc_unatt_i)
            else:
                print('The length of the video is too short for the given trial length.')
                # return NaN values
                isc_att.append(np.full((1, self.n_components), np.nan))
                isc_unatt.append(np.full((1, self.n_components), np.nan))
        isc_att = np.concatenate(tuple(isc_att), axis=0)
        isc_unatt = np.concatenate(tuple(isc_unatt), axis=0)
        return isc_att, isc_unatt


class StimulusInformedGCCA:
    '''
    Stimulus-informed Generalized Canonical Correlation Analysis. Not used in the paper.
    '''
    def __init__(self, nested_datalist, fs, Llist, offsetlist, fold=10, n_components=5, regularization='lwcov', message=True, sweep_list=np.linspace(-2,2,9), ISC=True, signifi_level=True, pool=True, n_permu=500, p_value=0.05, trials=False, dim_subspace=4, crs_val=True):
        '''
        nested_datalist: [EEG_list, Stim_list], where
            EEG_list: list of EEG data, each element is a T(#sample)xDx(#channel) array
            Stim_list: list of stimulus, each element is a T(#sample)xDy(#feature dim) array
        fs: Sampling rate
        Llist: [L_EEG, L_Stim], where
            L_EEG/L_Stim: If use (spatial-) temporal filter, the number of taps
        offsetlist: [offset_EEG, offset_Stim], where
            offset_EEG/offset_Stim: If use (spatial-) temporal filter, the offset of the time lags
        fold: Number of folds for cross-validation
        n_components: Number of components to be returned
        regularization: Regularization of the estimated covariance matrix
        message: If print message
        signifi_level: If calculate significance level
        pool: If pool the significance level of all components
        n_permu: Number of permutations for significance level calculation
        p_value: P-value for significance level calculation
        '''
        self.nested_datalist = nested_datalist
        self.fs = fs
        self.Llist = Llist
        self.offsetlist = offsetlist
        self.n_components = n_components
        self.fold = fold
        self.regularization = regularization
        self.message = message
        self.sweep_list = sweep_list
        self.ISC = ISC
        self.signifi_level = signifi_level
        self.pool = pool
        self.n_permu = n_permu
        self.p_value = p_value
        self.trials = trials
        self.dim_subspace = dim_subspace
        self.crs_val = crs_val

    def fit(self, datalist, rho):
        EEG, Stim = datalist
        if np.ndim(EEG) == 2:
            EEG = np.expand_dims(EEG, axis=2)
        T, D_eeg, N = EEG.shape
        _, D_stim = Stim.shape
        L_EEG, L_Stim = self.Llist
        dim_list = [D_eeg*L_EEG]*N + [D_stim*L_Stim]
        offset_EEG, offset_Stim = self.offsetlist
        EEG_list = [utils.block_Hankel(EEG[:,:,n], L_EEG, offset_EEG) for n in range(N)]
        EEG_Hankel = np.concatenate(tuple(EEG_list), axis=1)
        Stim_Hankel = utils.block_Hankel(Stim, L_Stim, offset_Stim)
        X = np.concatenate((EEG_Hankel, Stim_Hankel), axis=1)
        Rxx = np.cov(X, rowvar=False)
        Dxx = np.zeros_like(Rxx)
        dim_accumu = 0
        for dim in dim_list:
            if self.regularization == 'lwcov':
                Rxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim] = LedoitWolf().fit(X[:,dim_accumu:dim_accumu+dim]).covariance_
            Dxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim] = Rxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim]
            dim_accumu = dim_accumu + dim
        try:
            lam, W = utils.transformed_GEVD(Dxx, Rxx, rho, dim_list[-1], self.n_components)
            Lam = np.diag(lam)
            Rxx[:,-D_stim*L_Stim:] = Rxx[:,-D_stim*L_Stim:]*rho
        except:
            print("Numerical issue exists for eigh. Use eig instead.")
            Rxx[:,-D_stim*L_Stim:] = Rxx[:,-D_stim*L_Stim:]*rho
            lam, W = eig(Dxx, Rxx)
            idx = np.argsort(lam)
            lam = np.real(lam[idx]) # rank eigenvalues
            W = np.real(W[:, idx]) # rearrange eigenvectors accordingly
            lam = lam[:self.n_components]
            Lam = np.diag(lam)
            W = W[:,:self.n_components]
        # Right scaling
        Rxx[-D_stim*L_Stim:, :] = Rxx[-D_stim*L_Stim:, :]*rho
        W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rxx * T @ W @ Lam))
        # Shared subspace
        S = self.shared_subspace(X, W, Lam, N, D_eeg*L_EEG, D_stim*L_Stim, rho)
        # Forward models
        F = T * Dxx @ W
        # Organize weights of different modalities
        Wlist = utils.W_organize(W, datalist, self.Llist)
        Flist = utils.W_organize(F, datalist, self.Llist)
        Fstack = utils.F_organize(Flist[0], L_EEG, offset_EEG, avg=True)
        return Wlist, S, Fstack, lam

    def shared_subspace(self, X, W, Lam, N, DL, DL_Stim, rho):
        W_rho = copy.deepcopy(W)
        W_rho[-DL_Stim:,:] = W[-DL_Stim:,:]*rho
        X_center = copy.deepcopy(X)
        for n in range(N):
            X_center[:,n*DL:(n+1)*DL] -= np.mean(X_center[:,n*DL:(n+1)*DL], axis=0, keepdims=True)
        X_center[:, -DL_Stim:] -= np.mean(X_center[:, -DL_Stim:], axis=0, keepdims=True)
        S = X_center@W_rho@Lam
        return S

    def forward_model(self, EEG, Wlist, S=None):
        '''
        Input:
        EEG: EEG data with shape (T, D, N) [T: # sample, D: # channel, N: # subjects]
        Wlist: [Weeg, Wstim]
        S: shared subspace with shape (T, n_components); if not None, then calculate forward model from the shared subspace
        Note: if S is not None, then EEG must be the one used in the training stage. So it is equivalent to the S generated by self.fit(). This can be used as a sanity check.
        Outputs:
        F: forward model (D, n_components)
        '''
        W = Wlist[0]
        if np.ndim(EEG) == 2:
            EEG = np.expand_dims(EEG, axis=2)
            W = np.expand_dims(W, axis=1)
        _, _, N = EEG.shape
        X_list = [utils.block_Hankel(EEG[:,:,n], self.Llist[0], self.offsetlist[0]) for n in range(N)]
        X_list_center = [X_list[n] - np.mean(X_list[n], axis=0, keepdims=True) for n in range(N)]
        X_stack = np.stack(X_list_center, axis=2)
        if S is not None:
            F_T = np.mean(np.einsum('kt, tdn -> kdn', S.T, X_stack), axis=2)
            F_redun = F_T.T
        else:
            X = np.concatenate(tuple(X_list_center), axis=0)
            X_list_trans = [X_stack[:,:,n]@W[:,n,:] for n in range(N)]
            X_transformed = np.concatenate(tuple(X_list_trans), axis=0)
            F_redun = (lstsq(X_transformed, X)[0]).T
        F = utils.F_organize(F_redun, self.Llist[0], self.offsetlist[0])
        return F

    def avg_corr_coe(self, datalist, Wlist):
        '''
        Calculate the pairwise average correlation.
        Inputs:
        datalist: data of different modalities (a list) E.g, [EEG_stack, Stim]
        Wlist: weights of different modalities (a list)
        Output:
        avg_corr: average pairwise correlation
        '''
        n_components = self.n_components
        Llist = self.Llist
        offsetlist = self.offsetlist
        if self.ISC and (np.ndim(datalist[0]) != 2): # calculate avg correlation across only EEG views, unless there is only one EEG subject (CCA)
            GCCA = GeneralizedCCA(datalist[0], self.fs, Llist[0], offsetlist[0], n_components=n_components)
            avg_corr, _, avg_ChDist, avg_TSC = GCCA.avg_stats(datalist[0], Wlist[0])
        else:
            avg_corr = np.zeros(n_components)
            corr_mtx_list = []
            n_mod = len(datalist)
            for component in range(n_components):
                X_trans_list = []
                for i in range(n_mod):
                    W = np.squeeze(Wlist[i])
                    rawdata = datalist[i]
                    L = Llist[i]
                    offset = offsetlist[i]
                    if np.ndim(W) == 3:
                        w = W[:,:,component]
                        w = np.expand_dims(w, axis=1)
                        data_trans = [np.expand_dims(utils.block_Hankel(rawdata[:,:,n],L,offset),axis=2) for n in range(rawdata.shape[2])]
                        data = np.concatenate(tuple(data_trans), axis=2)
                        X_trans = np.einsum('tdn,dln->tln', data, w)
                        X_trans = np.squeeze(X_trans, axis=1)
                    if np.ndim(W) == 2:
                        w = W[:,component]
                        data = utils.block_Hankel(rawdata,L,offset)
                        X_trans = data@w
                        X_trans = np.expand_dims(X_trans, axis=1)
                    X_trans_list.append(X_trans)
                X_trans_all = np.concatenate(tuple(X_trans_list), axis=1)
                corr_mtx = np.corrcoef(X_trans_all, rowvar=False)
                N = X_trans_all.shape[1]
                corr_mtx_list.append(corr_mtx)
                avg_corr[component] = np.sum(corr_mtx-np.eye(N))/N/(N-1)
            corr_mtx_stack = np.stack(corr_mtx_list, axis=2)
            Squared_corr = np.sum(np.square(corr_mtx_stack[:,:,:self.dim_subspace]), axis=2)
            avg_TSC = np.sum(Squared_corr-self.dim_subspace*np.eye(N))/N/(N-1)
            Chordal_dist = np.sqrt(self.dim_subspace-Squared_corr)
            avg_ChDist = np.sum(Chordal_dist)/N/(N-1)
        return avg_corr, avg_ChDist, avg_TSC

    def avg_corr_coe_trials(self, datalist_trials, Wlist):
        stats = [(self.avg_corr_coe(trial, Wlist)) for trial in datalist_trials]
        avg_corr = np.concatenate(tuple([np.expand_dims(stats[i][0],axis=0) for i in range(len(datalist_trials))]), axis=0).mean(axis=0)
        avg_ChDist = np.array([stats[i][1] for i in range(len(datalist_trials))]).mean()
        avg_TSC = np.array([stats[i][2] for i in range(len(datalist_trials))]).mean()
        return avg_corr, avg_ChDist, avg_TSC

    def rho_sweep(self):
        nested_train, _, train_list, val_list  = utils.get_val_set(self.nested_datalist, self.fold, fold_val=10, crs_val=self.crs_val)
        best = -np.inf
        for i in self.sweep_list:
            rho = 10**i
            Wlist_train, _, _, _ = self.fit(train_list, rho)
            if self.trials:
                mod_trials = [utils.into_trials(mod, self.fs) for mod in val_list]
                val_trails = [[mod[idx_trial] for mod in mod_trials] for idx_trial in range(len(mod_trials[0]))]
                corr_test, _, _= self.avg_corr_coe_trials(val_trails, Wlist_train)
            else:
                corr_test, _, _ = self.avg_corr_coe(val_list, Wlist_train)
            print('rho={}, corr={}'.format(rho, corr_test[0]))
            if corr_test[0] >= best:
                rho_best = rho
                best = corr_test[0]
        # Discard the part used for validation
        nested_update = nested_train
        print('Best rho={}, val ISC={}'.format(rho_best, best))
        return rho_best, nested_update

    def get_transformed_data(self, datalist, Wlist):
        nb_mod = len(datalist)
        data_trans_list = []
        for i in range(nb_mod):
            W = Wlist[i]
            X = datalist[i]
            L = self.Llist[i]
            offset = self.offsetlist[i]
            if np.ndim(X) == 3:
                _, _, N = X.shape
                Hankellist = [np.expand_dims(utils.block_Hankel(X[:,:,n], L, offset), axis=2) for n in range(N)]
                Hankel_center = [hankel - np.mean(hankel, axis=0, keepdims=True) for hankel in Hankellist]
                X_center = np.concatenate(tuple(Hankel_center), axis=2)
                X_trans = np.einsum('tdn,dkn->tkn', X_center, np.transpose(W, (0,2,1)))
            elif np.ndim(X) == 2:
                X_hankel = utils.block_Hankel(X, L, offset)
                X_trans = X_hankel@W
            else:
                raise ValueError('The dimension of X is incorrect')
            data_trans_list.append(X_trans)
        return data_trans_list

    def get_avg_corr_coe(self, data_trans_list):
        if self.ISC and (np.ndim(data_trans_list[0]) != 2):
            X_trans = data_trans_list[0]
        else:
            data_trans_list = [X_trans if np.ndim(X_trans)==3 else np.expand_dims(X_trans, axis=2) for X_trans in data_trans_list]
            X_trans = np.concatenate(tuple(data_trans_list), axis=2)
        _, _, N = X_trans.shape
        n_components = self.n_components
        corr_mtx_stack = np.zeros((N,N,n_components))
        avg_corr = np.zeros(n_components)
        for component in range(n_components):
            corr_mtx_stack[:,:,component] = np.corrcoef(X_trans[:,component,:], rowvar=False)
            avg_corr[component] = np.sum(corr_mtx_stack[:,:,component]-np.eye(N))/N/(N-1)
        return avg_corr

    def permutation_test(self, datalist, Wlist, block_len):
        n_components = self.n_components
        corr_coe_topK = np.empty((0, n_components))
        data_trans_list = self.get_transformed_data(datalist, Wlist)
        for i in tqdm(range(self.n_permu)):
            datalist_shuffled = utils.shuffle_datalist(data_trans_list, block_len)
            corr_coe = self.get_avg_corr_coe(datalist_shuffled)
            corr_coe_topK = np.concatenate((corr_coe_topK, np.expand_dims(corr_coe, axis=0)), axis=0)
        return corr_coe_topK

    def permutation_test_trials(self, datalist_trials, Wlist, block_len):
        corr_coe_topK = np.empty((0, self.n_components))
        data_trans_trials_list = [self.get_transformed_data(datalist, Wlist) for datalist in datalist_trials]
        for i in tqdm(range(self.n_permu)):
            datalist_shuffled_trials = [utils.shuffle_datalist(trial, block_len) for trial in data_trans_trials_list]
            corr_coe_trials_list = [self.get_avg_corr_coe(data_trans_list) for data_trans_list in datalist_shuffled_trials]
            corr_coe_trials = np.concatenate(tuple([np.expand_dims(corr_coe, axis=0) for corr_coe in corr_coe_trials_list]), axis=0)
            corr_coe_topK = np.concatenate((corr_coe_topK, np.mean(corr_coe_trials, axis=0, keepdims=True)), axis=0)
        return corr_coe_topK

    def cross_val(self, rho=None):
        n_components = self.n_components
        fold = self.fold
        corr_train = np.zeros((fold, n_components))
        corr_test = np.zeros((fold, n_components))
        tsc_train = np.zeros((fold, 1))
        tsc_test = np.zeros((fold, 1))
        dist_train = np.zeros((fold, 1))
        dist_test = np.zeros((fold, 1))
        if rho is None:
            rho, nested_update = self.rho_sweep()
        else:
            nested_update = self.nested_datalist
        for idx in range(fold):
            train_list, test_list, _, _ = utils.split_mm_balance(nested_update, fold=fold, fold_idx=idx+1)
            Wlist_train, _, F_train, _ = self.fit(train_list, rho)
            corr_train[idx,:], dist_train[idx], tsc_train[idx] = self.avg_corr_coe(train_list, Wlist_train)
            if self.trials:
                mod_trials = [utils.into_trials(mod, self.fs) for mod in test_list]
                test_trails = [[mod[idx_trial] for mod in mod_trials] for idx_trial in range(len(mod_trials[0]))]
                corr_test[idx,:], dist_test[idx], tsc_test[idx] = self.avg_corr_coe_trials(test_trails, Wlist_train)
            else:
                corr_test[idx,:], dist_test[idx], tsc_test[idx] = self.avg_corr_coe(test_list, Wlist_train)
            if not self.crs_val:
                # fill the rest of the folds with the same results
                for i in range(idx+1, fold):
                    corr_train[i,:] = corr_train[idx,:]
                    dist_train[i] = dist_train[idx]
                    tsc_train[i] = tsc_train[idx]
                    corr_test[i,:] = corr_test[idx,:]
                    dist_test[i] = dist_test[idx]
                    tsc_test[i] = tsc_test[idx]
                break
        if self.signifi_level:
            if self.pool:
                if self.trials:
                    corr_trials = self.permutation_test_trials(test_trails, Wlist_train, block_len=1)
                else:
                    corr_trials = self.permutation_test(test_list, Wlist_train, block_len=1)
                corr_trials = np.sort(abs(corr_trials), axis=None)
                sig_idx = -int(self.n_permu*self.p_value*n_components)
                print('Significance level: ISC={}'.format(corr_trials[sig_idx]))
            else:
                if self.trials:
                    corr_trials = self.permutation_test_trials(test_trails, Wlist_train, block_len=20*self.fs)
                else:
                    corr_trials = self.permutation_test(test_list, Wlist_train, block_len=20*self.fs)
                corr_trials = np.sort(abs(corr_trials), axis=0)
                sig_idx = -int(self.n_permu*self.p_value)
                print('Significance level: ISCs={}'.format(corr_trials[sig_idx,:]))
        if self.message:
            print('Average ISC of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
            print('Average ISC of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
            print('Average TSC on the test sets: {}'.format(np.average(tsc_test)))
        return corr_train, corr_test, tsc_train, tsc_test, dist_train, dist_test, Wlist_train, F_train, rho


class GCCAPreprocessedCCA:
    '''
    Using GCCA as preprocessing and CCA as the following step. Not used in the paper.
    '''
    def __init__(self, Subj_ID, eeg_multisubj_list, stim_list, fs, para_gcca, para_cca_eeg, para_cca_stim, W_GCCA_folds=None, preprocessed_train_folds=None, preprocessed_test_folds=None, leave_out=2, fold=None, n_components_GCCA=192, n_components_CCA=3, regularization='lwcov', K_regu=None, message=True, signifi_level=True, n_permu=500, p_value=0.05):
        '''
        Subj_ID: Subject ID
        eeg_multisubj_list: list of EEG data, each element is a T(#sample)xD(#channel)xN(#subject) array
        stim_list: list of stimulus, each element is a T(#sample)xDy(#feature dim) array
        fs: Sampling rate
        para_gcca: Parameters for GCCA. The content is [L_EEG, offset_EEG] used in GCCA
        para_cca_eeg: Parameters for CCA. The content is [L_EEG, offset_EEG] used in CCA for EEG
        para_cca_stim: Parameters for CCA. The content is [L_Stim, offset_Stim] used in CCA for stimulus
        leave_out: Number of subjects to be left out for leave-video-out cross-validation
        fold: Number of folds for K-fold cross-validation
        n_components: Number of components to be returned
        regularization: Regularization of the estimated covariance matrix
        K_regu: Number of eigenvalues to be kept. Others will be set to zero. Keep all if K_regu=None
        '''
        self.Subj_ID = Subj_ID
        self.eeg_multisubj_list = eeg_multisubj_list
        self.eeg_onesubj_list = [eeg[:,:,Subj_ID] for eeg in eeg_multisubj_list]
        self.stim_list = stim_list
        self.stim_onesubj_list = [stim[:,:,Subj_ID] for stim in stim_list] if np.ndim(stim_list[0]) == 3 else stim_list
        self.fs = fs
        self.para_gcca = para_gcca
        self.para_cca_eeg = para_cca_eeg
        self.para_cca_stim = para_cca_stim
        self.W_GCCA_folds = W_GCCA_folds
        self.preprocessed_train_folds = preprocessed_train_folds
        self.preprocessed_test_folds = preprocessed_test_folds
        self.leave_out = leave_out
        self.fold = fold
        # assert that only one of leave_out and fold is not None
        assert (leave_out is not None) ^ (fold is not None), "Only one of leave_out and fold should be not None"
        self.train_list_folds, self.test_list_folds = utils.split_mm_balance_folds([self.eeg_multisubj_list, self.stim_list], self.fold) if self.fold is not None else utils.split_multi_mod_LVO([self.eeg_multisubj_list, self.stim_list], self.leave_out)
        self.n_components_GCCA = n_components_GCCA
        self.n_components_CCA = n_components_CCA
        self.regularization = regularization
        self.K_regu = K_regu
        self.message = message
        self.signifi_level = signifi_level
        self.n_permu = n_permu
        self.p_value = p_value

    def switch_subj(self, Subj_ID):
        self.Subj_ID = Subj_ID
        self.eeg_onesubj_list = [eeg[:,:,Subj_ID] for eeg in self.eeg_multisubj_list]
        self.stim_onesubj_list = [stim[:,:,Subj_ID] for stim in self.stim_list] if np.ndim(self.stim_list[0]) == 3 else self.stim_list

    def get_GCCA_results(self):
        L_EEG, offset_EEG = self.para_gcca
        self.W_GCCA_folds = []
        self.preprocessed_train_folds = []
        self.preprocessed_test_folds = []
        GCCA = GeneralizedCCA(self.eeg_multisubj_list, self.fs, L_EEG, offset_EEG, n_components=self.n_components_GCCA, regularization=self.regularization)
        for train_mm, test_mm in tqdm(zip(self.train_list_folds, self.test_list_folds)):
            [EEG_train, Stim_train], [EEG_test, Stim_test] = train_mm, test_mm
            W, _, _, _ = GCCA.fit(EEG_train)
            EEG_trans_train = GCCA.get_transformed_data(EEG_train, W)
            EEG_trans_test = GCCA.get_transformed_data(EEG_test, W)
            self.W_GCCA_folds.append(W)
            self.preprocessed_train_folds.append([EEG_trans_train, Stim_train])
            self.preprocessed_test_folds.append([EEG_trans_test, Stim_test])
        return self.W_GCCA_folds, self.preprocessed_train_folds, self.preprocessed_test_folds

    def forward_model(self, EEG_trans, idx=None):
        if idx is not None:
            EEG_ori = self.test_list_folds[idx][0][:,:,self.Subj_ID]
        else:
            EEG_ori = np.concatenate(tuple(self.eeg_onesubj_list), axis=0)
        F = (lstsq(EEG_trans, EEG_ori)[0]).T
        return F

    def get_transformed_data(self, X, Y, V_A, V_B):
        L_EEG, offset_EEG = self.para_cca_eeg
        L_Stim, offset_Stim = self.para_cca_stim
        mtx_X = utils.block_Hankel(X, L_EEG, offset_EEG)
        mtx_Y = utils.block_Hankel(Y, L_Stim, offset_Stim)
        mtx_X_centered = mtx_X - np.mean(mtx_X, axis=0, keepdims=True)
        mtx_Y_centered = mtx_Y - np.mean(mtx_Y, axis=0, keepdims=True)
        X_trans = mtx_X_centered@V_A
        Y_trans = mtx_Y_centered@V_B
        return X_trans, Y_trans

    def get_corr_coe(self, X_trans, Y_trans):
        # corr_pvalue = [pearsonr(X_trans[:,k], Y_trans[:,k]) for k in range(self.n_components_CCA)]
        # corr_coe = np.array([corr_pvalue[k][0] for k in range(self.n_components_CCA)])
        corr_coe = np.array([np.corrcoef(X_trans[:,k], Y_trans[:,k])[0,1] for k in range(self.n_components_CCA)])
        return corr_coe

    def get_sim(self, EEG_indiv, EEG_avg, dim_subspace):
        # Calulate TSC between the individual EEG components and the average EEG components (as a measure of the similarity between the individual EEG components and the average EEG components)
        # CCA_EEG = CanonicalCorrelationAnalysis([EEG_indiv], [EEG_avg], self.fs, L_EEG=1, L_Stim=1, leave_out=None, regularization=self.regularization, signifi_level=False, n_components=dim_subspace, dim_subspace=dim_subspace)
        # _, tsc, _, _, _, _, _ = CCA_EEG.fit(EEG_indiv, EEG_avg)
        # sim = np.array(tsc)
        corr_comp = [np.corrcoef(EEG_indiv[:,k], EEG_avg[:,k])[0,1] for k in range(dim_subspace)]
        sim = np.array(corr_comp)
        return sim

    def get_corr_sim_trials(self, EEG_indiv, EEG_avg, Att, Unatt, V_eeg, V_feat, BOOTSTRAP, trial_len):
        X_trans, Y_att_trans = self.get_transformed_data(EEG_indiv, Att, V_eeg, V_feat)
        _, Y_unatt_trans = self.get_transformed_data(EEG_indiv, Unatt, V_eeg, V_feat)
        dim_subspace = 2
        if X_trans.shape[0]-trial_len*self.fs >= 0:
            if BOOTSTRAP:
                nb_trials = min(X_trans.shape[0]//self.fs//3, 1000)
                start_points = np.random.randint(0, X_trans.shape[0]-trial_len*self.fs, size=nb_trials)
                start_points = np.sort(start_points)
            else:
                start_points = np.array(range(0, X_trans.shape[0] - X_trans.shape[0]%(self.fs*trial_len), self.fs*trial_len))
            X_trials = utils.into_trials(X_trans, self.fs, trial_len, start_points=start_points)
            Y_att_trials = utils.into_trials(Y_att_trans, self.fs, trial_len, start_points=start_points)
            Y_unatt_trials = utils.into_trials(Y_unatt_trans, self.fs, trial_len, start_points=start_points)
            EEG_avg_trials = utils.into_trials(EEG_avg, self.fs, trial_len, start_points=start_points)
            EEG_indiv_trials = utils.into_trials(EEG_indiv, self.fs, trial_len, start_points=start_points)
            corr_att_trials = np.stack([self.get_corr_coe(eeg, att) for eeg, att in zip(X_trials, Y_att_trials)])
            corr_unatt_trials = np.stack([self.get_corr_coe(eeg, unatt) for eeg, unatt in zip(X_trials, Y_unatt_trials)])
            sim_avg_indiv_trials = np.stack([self.get_sim(indiv, avg, dim_subspace) for indiv, avg in zip(EEG_indiv_trials, EEG_avg_trials)])
            sim_avg_indiv_trials = np.expand_dims(sim_avg_indiv_trials, axis=1) if np.ndim(sim_avg_indiv_trials) == 1 else sim_avg_indiv_trials
        else:
            print('The length of the video is too short for the given trial length.')
            # return NaN values
            corr_att_trials = np.full((1, self.n_components_CCA), np.nan)
            corr_att_trials = np.full((1, self.n_components_CCA), np.nan)
            sim_avg_indiv_trials = np.full((1, dim_subspace), np.nan)
            # sim_avg_indiv_trials = np.full((1, 1), np.nan)
        return corr_att_trials, corr_unatt_trials, sim_avg_indiv_trials

    def search_para(self, nb_comp_kept_grid=range(16, 200, 16)):
        print('Getting GCCA results...')
        if self.W_GCCA_folds is None:
            _, _, _ = self.get_GCCA_results()
        data_for_search = self.preprocessed_train_folds
        nb_folds = len(data_for_search)
        L_EEG, offset_EEG = self.para_cca_eeg
        L_Stim, offset_Stim = self.para_cca_stim
        corr_grid = []
        print('Searching for the optimal number of components...')
        for nb_comp in nb_comp_kept_grid:
            corr_val_fold = np.zeros(nb_folds)
            for idx in range(nb_folds):
                [EEG, Stim] = data_for_search[idx]
                EEG = np.mean(EEG[:,:nb_comp,:], axis=-1)
                Stim = np.mean(Stim, axis=-1) if np.ndim(Stim) == 3 else Stim
                [eeg_train, stim_train], [eeg_val, stim_val] = utils.split_multi_mod([EEG, Stim], fold=10, fold_idx=idx+1) # nb_folds should be smaller than 10
                CCA = CanonicalCorrelationAnalysis(self.eeg_onesubj_list, self.stim_onesubj_list, self.fs, L_EEG, L_Stim, offset_EEG, offset_Stim, fold=self.fold, leave_out=self.leave_out, n_components=1, regularization=self.regularization, signifi_level=False)
                _, _, _, _, V_A_train, V_B_train, _ = CCA.fit(eeg_train, stim_train)
                corr_val_fold[idx], _, _, _ = CCA.cal_corr_coe(eeg_val, stim_val, V_A_train, V_B_train)
            # print('Number of components kept: {}, averaged CC1: {}'.format(nb_comp, np.mean(corr_val_fold)))
            corr_grid.append(np.mean(corr_val_fold))
        best_nb_comp = nb_comp_kept_grid[np.argmax(corr_grid)]
        print('Best number of components: {}'.format(best_nb_comp))
        self.preprocessed_train_folds = [[datalist[0][:,:best_nb_comp,:], datalist[1]] for datalist in self.preprocessed_train_folds]
        self.preprocessed_test_folds = [[datalist[0][:,:best_nb_comp,:], datalist[1]] for datalist in self.preprocessed_test_folds]
        return best_nb_comp

    def cross_val(self):
        print('Getting GCCA results...')
        if self.W_GCCA_folds is None:
            _, _, _ = self.get_GCCA_results()
        L_EEG, offset_EEG = self.para_cca_eeg
        L_Stim, offset_Stim = self.para_cca_stim
        nb_folds = len(self.preprocessed_train_folds)
        n_components = self.n_components_CCA
        corr_train_fold = np.zeros((nb_folds, n_components))
        corr_test_fold = np.zeros((nb_folds, n_components))
        forward_model_fold = []
        EEG_comp_fold = []
        print('After GCCA preprocessing, getting CCA cross-validation results...')
        CCA = CanonicalCorrelationAnalysis(self.eeg_onesubj_list, self.stim_onesubj_list, self.fs, L_EEG, L_Stim, offset_EEG, offset_Stim, fold=self.fold, leave_out=self.leave_out, n_components=n_components, regularization=self.regularization, K_regu=self.K_regu, message=self.message, signifi_level=self.signifi_level, n_permu=self.n_permu, p_value=self.p_value)
        idx = 0
        for train_mm, test_mm in tqdm(zip(self.preprocessed_train_folds, self.preprocessed_test_folds)):
            [EEG_trans_train, Stim_train], [EEG_trans_test, Stim_test] = train_mm, test_mm
            EEG_train = EEG_trans_train[:,:,self.Subj_ID]
            EEG_test = EEG_trans_test[:,:,self.Subj_ID]
            Stim_train = Stim_train[:,:,self.Subj_ID] if np.ndim(Stim_train) == 3 else Stim_train
            Stim_test = Stim_test[:,:,self.Subj_ID] if np.ndim(Stim_test) == 3 else Stim_test
            corr_train_fold[idx,:], _, _, _, V_A_train, V_B_train, _ = CCA.fit(EEG_train, Stim_train)
            corr_test_fold[idx,:], _, _, _ = CCA.cal_corr_coe(EEG_test, Stim_test, V_A_train, V_B_train)
            EEG_comp, _ = CCA.get_transformed_data(EEG_test, Stim_test, V_A_train, V_B_train)
            EEG_comp_fold.append(EEG_comp)
            forward_model = self.forward_model(EEG_comp, idx)
            forward_model_fold.append(forward_model)
            idx += 1
        EEG_comp_all = np.concatenate(tuple(EEG_comp_fold), axis=0)
        forward_model_all = self.forward_model(EEG_comp_all, idx=None)
        if self.message:
            print('Average correlation coefficients of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train_fold, axis=0)))
            print('Average correlation coefficients of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test_fold, axis=0)))
        return corr_train_fold, corr_test_fold, forward_model_fold, forward_model_all

    def att_or_unatt_trials(self, feat_unatt_list, trial_len, BOOTSTRAP=True, COMP_FROM_AVG_COMP=False):
        print('Getting GCCA results...')
        if self.W_GCCA_folds is None:
            _, _, _ = self.get_GCCA_results()
        L_EEG, offset_EEG = self.para_cca_eeg
        L_Stim, offset_Stim = self.para_cca_stim
        nb_folds = len(self.preprocessed_train_folds)
        train_unatt_folds, test_unatt_folds = utils.split_mm_balance_folds([feat_unatt_list], self.fold) if self.fold is not None else utils.split_multi_mod_LVO([feat_unatt_list], self.leave_out)
        print('After GCCA preprocessing, getting CCA results for attended and unattended trials...')
        CCA = CanonicalCorrelationAnalysis(self.eeg_onesubj_list, self.stim_onesubj_list, self.fs, L_EEG, L_Stim, offset_EEG, offset_Stim, fold=self.fold, leave_out=self.leave_out, n_components=self.n_components_CCA, regularization=self.regularization, K_regu=self.K_regu, message=self.message, signifi_level=self.signifi_level, n_permu=self.n_permu, p_value=self.p_value)
        corr_att_eeg_train = []
        corr_unatt_eeg_train = []
        corr_att_eeg_test = []
        corr_unatt_eeg_test = []
        sim_train= [] # holds the similarity measurements between the averaged GCCA components and individual GCCA components
        sim_test = []
        for idx in range(nb_folds):
            [EEG_trans_train, Att_train], [EEG_trans_test, Att_test] = self.preprocessed_train_folds[idx], self.preprocessed_test_folds[idx]
            [Unatt_train], [Unatt_test] = train_unatt_folds[idx], test_unatt_folds[idx]
            EEG_train = np.mean(EEG_trans_train, axis=-1) if COMP_FROM_AVG_COMP else EEG_trans_train[:,:,self.Subj_ID]
            EEG_train_avg = np.mean(EEG_trans_train, axis=-1)
            EEG_test = np.mean(EEG_trans_test, axis=-1) if COMP_FROM_AVG_COMP else EEG_trans_test[:,:,self.Subj_ID]
            EEG_test_avg = np.mean(EEG_trans_test, axis=-1)
            # EEG_test_avg = (np.sum(EEG_trans_test, axis=-1)-EEG_trans_test[:,:,self.Subj_ID])/(EEG_trans_test.shape[-1]-1)
            Att_train = Att_train[:,:,self.Subj_ID] if np.ndim(Att_train) == 3 else Att_train
            Att_test = Att_test[:,:,self.Subj_ID] if np.ndim(Att_test) == 3 else Att_test
            Unatt_train = Unatt_train[:,:,self.Subj_ID] if np.ndim(Unatt_train) == 3 else Unatt_train
            Unatt_test = Unatt_test[:,:,self.Subj_ID] if np.ndim(Unatt_test) == 3 else Unatt_test
            _, _, _, _, V_eeg_train, V_feat_train, _ = CCA.fit(EEG_train, Att_train)
            corr_att_train, corr_unatt_train, sim_avg_indiv_train = self.get_corr_sim_trials(EEG_train, EEG_train_avg, Att_train, Unatt_train, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len)
            corr_att_eeg_train.append(corr_att_train)
            corr_unatt_eeg_train.append(corr_unatt_train)
            corr_att_test, corr_unatt_test, sim_avg_indiv_test = self.get_corr_sim_trials(EEG_test, EEG_test_avg, Att_test, Unatt_test, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len)
            corr_att_eeg_test.append(corr_att_test)
            corr_unatt_eeg_test.append(corr_unatt_test)
            sim_train.append(sim_avg_indiv_train)
            sim_test.append(sim_avg_indiv_test)
        corr_att_eeg_train = np.concatenate(tuple(corr_att_eeg_train), axis=0)
        corr_unatt_eeg_train = np.concatenate(tuple(corr_unatt_eeg_train), axis=0)
        corr_att_eeg_test = np.concatenate(tuple(corr_att_eeg_test), axis=0)
        corr_unatt_eeg_test = np.concatenate(tuple(corr_unatt_eeg_test), axis=0)
        sim_train = np.concatenate(tuple(sim_train), axis=0)
        sim_test = np.concatenate(tuple(sim_test), axis=0)
        return corr_att_eeg_train, corr_att_eeg_test, corr_unatt_eeg_train, corr_unatt_eeg_test, sim_train, sim_test


    def get_transformed_EEG(self, X_stack, W_stack):
        _, _, N = X_stack.shape
        if np.ndim (W_stack) == 2: # for correlated component analysis
            W_stack = np.expand_dims(W_stack, axis=1)
            W_stack = np.repeat(W_stack, N, axis=1)
        Hankellist = [np.expand_dims(utils.block_Hankel(X_stack[:,:,n], self.L_EEG, self.offset_EEG), axis=2) for n in range(N)]
        Hankel_center = [hankel - np.mean(hankel, axis=0, keepdims=True) for hankel in Hankellist]
        X_center = np.concatenate(tuple(Hankel_center), axis=2)
        X_trans = np.einsum('tdn,dkn->tkn', X_center, np.transpose(W_stack, (0,2,1)))
        X_trans_sum = np.sum(X_trans, axis=2)
        return X_trans_sum

    def cross_val(self):
        fold = self.fold
        n_components = self.n_components
        corr_train = np.zeros((fold, n_components))
        corr_test = np.zeros((fold, n_components))
        for idx in range(fold):
            train_list = self.nested_train[idx]
            test_list = self.nested_test[idx]
            We_train = self.nested_We_train[idx]
            S = self.nested_S[idx]
            F_train = self.nested_F_train[idx]
            # obtain the stimulus filters by least square regression
            LS = LeastSquares(self.EEG_list, self.Stim_list, self.fs, decoding=False, L_Stim=self.L_Stim, offset_Stim=self.offset_Stim)
            Ws_train, _, = LS.encoder(S, train_list[1])
            corr_train[idx,:], _ = self.correlation(train_list[0], train_list[1], We_train, Ws_train)
            corr_test[idx,:], _ = self.correlation(test_list[0], test_list[1], We_train, Ws_train)
        if self.signifi_level:
            if self.pool:
                corr_trials = self.permutation_test(test_list[0], test_list[1], We_train, Ws_train, block_len=1)
                corr_trials = np.sort(abs(corr_trials), axis=None)
                sig_idx = -int(self.n_permu*self.p_value*n_components)
                sig_corr = corr_trials[sig_idx]
                print('Significance level: {}'.format(sig_corr))
            else:
                corr_trials = self.permutation_test(test_list[0], test_list[1], We_train, Ws_train, block_len=1)
                corr_trials = np.sort(abs(corr_trials), axis=0)
                sig_idx = -int(self.n_permu*self.p_value)
                sig_corr = corr_trials[sig_idx,:]
                print('Significance level of each component: {}'.format(sig_corr))
        else:
            sig_corr = None
        if self.message:
            print('Average correlation coefficients of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
            print('Average correlation coefficients of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
        return corr_train, corr_test, sig_corr, We_train, Ws_train, F_train