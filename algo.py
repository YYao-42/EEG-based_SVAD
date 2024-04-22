'''
Inherited from the previous project.
Class CanonicalCorrelationAnalysis and Class GeneralizedCCA are heavily modified.
Class GeneralizedCCA_MultiMod and Class GCCAPreprocessedCCA are newly added.
Other classes are not modified, and thus do not have all the new features.
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
    def __init__(self, EEG_list, Stim_list, fs, L_EEG, L_Stim, offset_EEG=0, offset_Stim=0, mask_list=None, dim_list_EEG=None, dim_list_Stim=None, fold=10, leave_out=2, n_components=5, regularization='lwcov', K_regu=None, message=True, signifi_level=True, n_permu=500, p_value=0.05, dim_subspace=2):
        '''
        EEG_list: list of EEG data, each element is a T(#sample)xDx(#channel) array
        Stim_list: list of stimulus, each element is a T(#sample)xDy(#feature dim) array
        fs: Sampling rate
        L_EEG/L_Stim: If use (spatial-) temporal filter, the number of taps
        offset_EEG/offset_Stim: If use (spatial-) temporal filter, the offset of time lags
        n_components: Number of components to be returned
        regularization: Regularization of the estimated covariance matrix
        K_regu: Number of eigenvalues to be kept. Others will be set to zero. Keep all if K_regu=None
        V_A/V_B: Filters of X and Y. Use only in test mode.
        Lam: Eigenvalues. Use only in test mode.
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
        if self.mask_list is not None:
            # select the mask that matches X.shape[0]
            if len(self.mask_train) == length:
                mask = list(np.squeeze(self.mask_train))
            else:
                mask = list(np.squeeze(self.mask_test))
            mask_EEG = copy.deepcopy(mask)
            mask_Stim = copy.deepcopy(mask)
            mask_EEG = (mask_EEG + self.offset_EEG*[True])[self.offset_EEG:]
            mask_Stim = (mask_Stim + self.offset_Stim*[True])[self.offset_Stim:]
            mask_lags_EEG = copy.deepcopy(mask_EEG)
            mask_lags_Stim = copy.deepcopy(mask_Stim)
            for i in range(len(mask)):
                if mask_EEG[i] == False:
                    end = min(i+self.L_EEG, len(mask))
                    mask_lags_EEG[i:end] = (end-i)*[False]
                if mask_Stim[i] == False:
                    end = min(i+self.L_Stim, len(mask))
                    mask_lags_Stim[i:end] = (end-i)*[False]
            mask = np.logical_and(mask_lags_EEG, mask_lags_Stim)
        else:
            mask = None
        return mask

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

    def get_transformed_data(self, X, Y, V_A, V_B):
        '''
        Get the transformed data
        '''
        mask = self.select_mask(X.shape[0])
        mtx_X = utils.block_Hankel(X, self.L_EEG, self.offset_EEG, mask)
        mtx_Y = utils.block_Hankel(Y, self.L_Stim, self.offset_Stim, mask)
        mtx_X_centered = mtx_X - np.mean(mtx_X, axis=0, keepdims=True)
        mtx_Y_centered = mtx_Y - np.mean(mtx_Y, axis=0, keepdims=True)
        X_trans = mtx_X_centered@V_A
        Y_trans = mtx_Y_centered@V_B
        return X_trans, Y_trans

    def get_corr_coe(self, X_trans, Y_trans):
        corr_pvalue = [pearsonr(X_trans[:,k], Y_trans[:,k]) for k in range(self.n_components)]
        corr_coe = np.array([corr_pvalue[k][0] for k in range(self.n_components)])
        p_value = np.array([corr_pvalue[k][1] for k in range(self.n_components)])
        TSC = np.sum(np.square(corr_coe[:self.dim_subspace]))
        ChDist = np.sqrt(self.dim_subspace-TSC)
        return corr_coe, TSC, ChDist, p_value

    def cal_corr_coe(self, X, Y, V_A, V_B):
        X_trans, Y_trans = self.get_transformed_data(X, Y, V_A, V_B)
        corr_coe, TSC, ChDist, p_value = self.get_corr_coe(X_trans, Y_trans)
        return corr_coe, TSC, ChDist, p_value

    def cal_corr_coe_trials(self, X_trials, Y_trials, V_A, V_B, avg=True):
        stats = [(self.cal_corr_coe(X, Y, V_A, V_B)) for X, Y in zip(X_trials, Y_trials)]
        corr_coe = np.concatenate(tuple([np.expand_dims(stats[i][0],axis=0) for i in range(len(X_trials))]), axis=0)
        TSC = np.array([stats[i][1] for i in range(len(X_trials))])
        ChDist = np.array([stats[i][2] for i in range(len(X_trials))])
        if avg:
            corr_coe = np.mean(corr_coe, axis=0)
            TSC = np.mean(TSC)
            ChDist = np.mean(ChDist)
        return corr_coe, TSC, ChDist

    def permutation_test(self, X, Y, V_A, V_B, PHASE_SCRAMBLE=True, block_len=None, X_trans=None, Y_trans=None):
        corr_coe_topK = np.zeros((self.n_permu, self.n_components))
        if X_trans is None and Y_trans is None:
            X_trans, Y_trans = self.get_transformed_data(X, Y, V_A, V_B) 
        for i in tqdm(range(self.n_permu)):
            X_shuffled = utils.shuffle_2D(X_trans, block_len) if not PHASE_SCRAMBLE else utils.phase_scramble_2D(X_trans)
            Y_shuffled = utils.shuffle_2D(Y_trans, block_len) if not PHASE_SCRAMBLE else utils.phase_scramble_2D(Y_trans)
            corr_pvalue = [pearsonr(X_shuffled[:,k], Y_shuffled[:,k]) for k in range(self.n_components)]
            corr_coe_topK[i,:] = np.array([corr_pvalue[k][0] for k in range(self.n_components)])
        return corr_coe_topK

    def permutation_test_trials(self, X_trials, Y_trials, V_A, V_B, PHASE_SCRAMBLE=True, block_len=None):
        corr_coe_topK = np.empty((0, self.n_components))
        transformed_data_list = [self.get_transformed_data(X, Y, V_A, V_B) for X, Y in zip(X_trials, Y_trials)]
        for i in tqdm(range(self.n_permu)):
            X_shuffled_trials = [utils.shuffle_2D(trans[0], block_len) if not PHASE_SCRAMBLE else utils.phase_scramble_2D(trans[0]) for trans in transformed_data_list]
            Y_shuffled_trials = [utils.shuffle_2D(trans[1], block_len) if not PHASE_SCRAMBLE else utils.phase_scramble_2D(trans[1]) for trans in transformed_data_list]
            stat_trials_list = [self.get_corr_coe(X_trans, Y_trans) for X_trans, Y_trans in zip(X_shuffled_trials, Y_shuffled_trials)]
            corr_coe_trials = np.concatenate(tuple([np.expand_dims(corr_coe[0], axis=0) for corr_coe in stat_trials_list]), axis=0)
            corr_coe_topK = np.concatenate((corr_coe_topK, np.mean(corr_coe_trials, axis=0, keepdims=True)), axis=0)
        # X_trans_trials = [trans[0] for trans in transformed_data_list]
        # Y_trans_trials = [trans[1] for trans in transformed_data_list]
        # for i in tqdm(range(self.n_permu)):
        #     random.shuffle(X_trans_trials)
        #     random.shuffle(Y_trans_trials)
        #     stat_trials_list = [self.get_corr_coe(X_trans, Y_trans) for X_trans, Y_trans in zip(X_trans_trials, Y_trans_trials)]
        #     corr_coe_trials = np.concatenate(tuple([np.expand_dims(corr_coe[0], axis=0) for corr_coe in stat_trials_list]), axis=0)
        #     corr_coe_topK = np.concatenate((corr_coe_topK, np.mean(corr_coe_trials, axis=0, keepdims=True)), axis=0)
        return corr_coe_topK

    def forward_model(self, X, V_A, X_trans=None):
        '''
        Inputs:
        X: observations (one subject) TxD
        V_A: filters/backward models DLxK
        L: time lag (if temporal-spatial)
        Output:
        F: forward model
        '''
        if X_trans is not None:
            F = (lstsq(X_trans, X)[0]).T
        else:
            mask = self.select_mask(X.shape[0])
            X_block_Hankel = utils.block_Hankel(X, self.L_EEG, self.offset_EEG, mask)
            Rxx = np.cov(X_block_Hankel, rowvar=False)
            if np.ndim(Rxx) == 0:
                Rxx = np.array([[Rxx]])
            F_redun = Rxx@V_A@LA.inv(V_A.T@Rxx@V_A)
            F = utils.F_organize(F_redun, self.L_EEG, self.offset_EEG)
        return F

    def calculate_sig_corr(self, corr_trials, nb_fold=1):
        assert self.n_components*self.n_permu*nb_fold == corr_trials.shape[0]*corr_trials.shape[1]
        sig_idx = -int(self.n_permu*self.p_value*self.n_components*nb_fold)
        corr_trials = np.sort(abs(corr_trials), axis=None)
        return corr_trials[sig_idx]

    def cross_val(self, trial_len=None, V_eeg=None, V_Stim=None, saccade_list=None):
        fold = self.fold
        n_components = self.n_components
        corr_train = np.zeros((fold, n_components))
        corr_test = np.zeros((fold, n_components))
        tsc_train = np.zeros((fold, 1))
        tsc_test = np.zeros((fold, 1))
        dist_train = np.zeros((fold, 1))
        dist_test = np.zeros((fold, 1))
        for idx in range(fold):
            # EEG_train, EEG_test, Sti_train, Sti_test = split(EEG_list, feature_list, fold=fold, fold_idx=idx+1)
            EEG_train, EEG_test, Sti_train, Sti_test = utils.split_balance(self.EEG_list, self.Stim_list, fold=fold, fold_idx=idx+1)
            if saccade_list is not None:
                _, _, _, Sacc_test = utils.split_balance(self.EEG_list, saccade_list, fold=fold, fold_idx=idx+1)
                [EEG_test, Sti_test] = utils.remove_saccade([EEG_test, Sti_test], Sacc_test)
            if V_eeg is not None:
                V_A_train, V_B_train = V_eeg, V_Stim
                corr_train[idx,:], tsc_train[idx], dist_test[idx], _ = self.cal_corr_coe(EEG_train, Sti_train, V_A_train, V_B_train)
            else:
                corr_train[idx,:], tsc_train[idx], dist_train[idx], _, V_A_train, V_B_train, _ = self.fit(EEG_train, Sti_train)
            if trial_len is not None:
                EEG_trials = utils.into_trials(EEG_test, self.fs, t=trial_len)
                Sti_trials = utils.into_trials(Sti_test, self.fs, t=trial_len)
                corr_test[idx,:], tsc_test[idx], dist_test[idx] = self.cal_corr_coe_trials(EEG_trials, Sti_trials, V_A_train, V_B_train)
            else:
                corr_test[idx,:], tsc_test[idx], dist_test[idx], _ = self.cal_corr_coe(EEG_test, Sti_test, V_A_train, V_B_train)
        if self.signifi_level:
            if trial_len is not None:
                corr_trials = self.permutation_test_trials(EEG_trials, Sti_trials, V_A=V_A_train, V_B=V_B_train)
            else:
                corr_trials = self.permutation_test(EEG_test, Sti_test, V_A=V_A_train, V_B=V_B_train)
            sig_corr = self.calculate_sig_corr(corr_trials)
            print('Significance level: {}'.format(sig_corr))
        else:
            sig_corr = None
        if self.message:
            print('Average correlation coefficients of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
            print('Average correlation coefficients of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
        return corr_train, corr_test, sig_corr, tsc_train, tsc_test, dist_train, dist_test, V_A_train, V_B_train

    def match_mismatch(self, trial_len, nb_trials, feat_distract_list=None, V_eeg=None, V_Stim=None, saccade_list=None):
        # always train on match and try to distinguish match from mismatch 
        # mismatch is a random segment that is not shown on the screen
        corr_match_fold = []
        corr_mismatch_fold = []
        fold = self.fold
        for idx in range(fold):
            EEG_train, EEG_test, Sti_train, Sti_test = utils.split_balance(self.EEG_list, self.Stim_list, fold=fold, fold_idx=idx+1)
            if V_eeg is None:
                _, _, _, _, V_A_train, V_B_train, _ = self.fit(EEG_train, Sti_train)
            else:
                V_A_train, V_B_train = V_eeg, V_Stim
            if feat_distract_list is not None:
                _, _, _, Distract_test = utils.split_balance(self.EEG_list, feat_distract_list, fold=fold, fold_idx=idx+1)
            else:
                Distract_test = Sti_test
            if saccade_list is not None:
                _, _, _, Sacc_test = utils.split_balance(self.EEG_list, saccade_list, fold=fold, fold_idx=idx+1)
                [EEG_test, Sti_test, Distract_test] = utils.remove_saccade([EEG_test, Sti_test, Distract_test], Sacc_test)
            start_points = np.random.randint(0, EEG_test.shape[0]-trial_len*self.fs, size=nb_trials)
            EEG_trials = utils.into_trials(EEG_test, self.fs, trial_len, start_points=start_points)
            Match_trials = utils.into_trials(Sti_test, self.fs, trial_len, start_points=start_points)

            Mismatch_trials = [utils.select_distractors(Distract_test, self.fs, trial_len, start_point) for start_point in start_points]
            corr_match, _, _ = self.cal_corr_coe_trials(EEG_trials, Match_trials, V_A_train, V_B_train, avg=False)
            corr_mismatch, _, _ = self.cal_corr_coe_trials(EEG_trials, Mismatch_trials, V_A_train, V_B_train, avg=False)
            corr_match_fold.append(corr_match)
            corr_mismatch_fold.append(corr_mismatch)
        corr_match_fold = np.concatenate(tuple(corr_match_fold), axis=0)
        corr_mismatch_fold = np.concatenate(tuple(corr_mismatch_fold), axis=0)
        return corr_match_fold, corr_mismatch_fold

    def att_or_unatt(self, feat_unatt_list, trial_len, TRAIN_WITH_ATT, V_eeg=None, V_Stim=None, nb_trials=None, saccade_list=None):
        # train on attended data and try to decode the attended object
        # train on unattended data and try to decode the unattended object
        # attended and unattended data are both shown on the screen
        corr_att_fold = []
        corr_unatt_fold = []
        fold = self.fold
        for idx in range(fold):
            EEG_train, EEG_test, Att_train, Att_test = utils.split_balance(self.EEG_list, self.Stim_list, fold=fold, fold_idx=idx+1)
            _, _, Unatt_train, Unatt_test = utils.split_balance(self.EEG_list, feat_unatt_list, fold=fold, fold_idx=idx+1)
            if saccade_list is not None:
                _, _, _, Sacc_test = utils.split_balance(self.EEG_list, saccade_list, fold=fold, fold_idx=idx+1)
                [EEG_test, Att_test, Unatt_test] = utils.remove_saccade([EEG_test, Att_test, Unatt_test], Sacc_test)
            if V_eeg is None:
                if TRAIN_WITH_ATT:
                    _, _, _, _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Att_train)
                else:
                    _, _, _, _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Unatt_train)
            else:
                V_eeg_train, V_feat_train = V_eeg, V_Stim
            if trial_len is not None:
                if nb_trials is not None:
                    start_points = np.random.randint(0, EEG_test.shape[0]-trial_len*self.fs, size=nb_trials)
                else:
                    start_points = None
                EEG_trials = utils.into_trials(EEG_test, self.fs, trial_len, start_points=start_points)
                Att_trials = utils.into_trials(Att_test, self.fs, trial_len, start_points=start_points)
                Unatt_trials = utils.into_trials(Unatt_test, self.fs, trial_len, start_points=start_points)
                corr_att, _, _ = self.cal_corr_coe_trials(EEG_trials, Att_trials, V_eeg_train, V_feat_train, avg=False)
                corr_unatt, _, _ = self.cal_corr_coe_trials(EEG_trials, Unatt_trials, V_eeg_train, V_feat_train, avg=False)
                if idx == fold-1 and self.signifi_level and nb_trials is None:
                    corr_trials_att = self.permutation_test_trials(EEG_trials, Att_trials, V_A=V_eeg_train, V_B=V_feat_train)
                    corr_trials_unatt = self.permutation_test_trials(EEG_trials, Unatt_trials, V_A=V_eeg_train, V_B=V_feat_train)
            else:
                corr_att, _, _, _ = self.cal_corr_coe(EEG_test, Att_test, V_eeg_train, V_feat_train)
                corr_unatt, _, _, _ = self.cal_corr_coe(EEG_test, Unatt_test, V_eeg_train, V_feat_train)
                corr_att = np.expand_dims(corr_att, axis=0)
                corr_unatt = np.expand_dims(corr_unatt, axis=0)
                if idx == fold-1 and self.signifi_level and nb_trials is None:
                    corr_trials_att = self.permutation_test(EEG_test, Att_test, V_A=V_eeg_train, V_B=V_feat_train)
                    corr_trials_unatt = self.permutation_test(EEG_test, Unatt_test, V_A=V_eeg_train, V_B=V_feat_train)
            if idx == fold-1 and self.signifi_level and nb_trials is None:
                sig_corr_att = self.calculate_sig_corr(corr_trials_att)
                sig_corr_unatt = self.calculate_sig_corr(corr_trials_unatt)
                print('Significance level (tested on Att): {}'.format(sig_corr_att))
                print('Significance level (tested on Unatt): {}'.format(sig_corr_unatt))
            else:
                sig_corr_att = None
                sig_corr_unatt = None
            corr_att_fold.append(corr_att)
            corr_unatt_fold.append(corr_unatt)
        corr_att_fold = np.concatenate(tuple(corr_att_fold), axis=0)
        corr_unatt_fold = np.concatenate(tuple(corr_unatt_fold), axis=0)
        if self.message:
            modes = [
                ("Mode 1: train and test with attended features", corr_att_fold),
                ("Mode 2: train with attended features and test with unattended features", corr_unatt_fold),
                ("Mode 3: train and test with unattended features", corr_unatt_fold),
                ("Mode 4: train with unattended features and test with attended features", corr_att_fold),
                ("Mode 5: train with single object data and test with attended features", corr_att_fold),
                ("Mode 6: train with single object data and test with unattended features", corr_unatt_fold)
            ]
            for i, (mode, corr) in enumerate(modes):
                if V_eeg is not None and i < 4:
                    continue
                elif V_eeg is None and TRAIN_WITH_ATT and i > 1:
                    continue
                elif V_eeg is None and not TRAIN_WITH_ATT and (i < 2 or i > 3):
                    continue
                print(mode)
                print(np.average(corr, axis=0))
        return corr_att_fold, corr_unatt_fold, V_eeg_train, V_feat_train, sig_corr_att, sig_corr_unatt
    
    def cross_val_LVO(self, V_eeg=None, V_Stim=None):
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
            if V_eeg is not None:
                V_A_train, V_B_train = V_eeg, V_Stim
                corr_train_fold[idx,:], _, _, _ = self.cal_corr_coe(EEG_train, Sti_train, V_A_train, V_B_train)
            else:
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

    def match_mismatch_LVO(self, trial_len, V_eeg=None, V_Stim=None):
        # always train on match and try to distinguish match from mismatch 
        # mismatch is a random segment that is not shown on the screen
        nb_videos = len(self.Stim_list)
        nb_folds = nb_videos//self.leave_out
        assert nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
        nested_datalist = [self.EEG_list, self.Stim_list, self.mask_list] if self.mask_list is not None else [self.EEG_list, self.Stim_list]
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO(nested_datalist, self.leave_out)
        assert len(train_list_folds) == len(test_list_folds) == nb_folds, "The number of folds is not correct."
        corr_match_eeg = []
        corr_mismatch_eeg = []
        for idx in range(0, nb_folds):
            if self.mask_list is not None:
                [EEG_train, Sti_train, self.mask_train], [EEG_test, Sti_test, self.mask_test] = train_list_folds[idx], test_list_folds[idx]
            else:
                [EEG_train, Sti_train], [EEG_test, Sti_test] = train_list_folds[idx], test_list_folds[idx]
            if V_eeg is None:
                _, _, _, _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Sti_train)
            else:
                V_eeg_train, V_feat_train = V_eeg, V_Stim
            EEG_trans, Sti_trans = self.get_transformed_data(EEG_test, Sti_test, V_eeg_train, V_feat_train)

            if EEG_trans.shape[0]-trial_len*self.fs >= 0:
                nb_trials = EEG_trans.shape[0]//self.fs*2
                start_points = np.random.randint(0, EEG_trans.shape[0]-trial_len*self.fs, size=nb_trials)
                EEG_trans_trials = utils.into_trials(EEG_trans, self.fs, trial_len, start_points=start_points)
                Match_trans_trials = utils.into_trials(Sti_trans, self.fs, trial_len, start_points=start_points)
                Mismatch_trans_trials = [utils.select_distractors(Sti_trans, self.fs, trial_len, start_point) for start_point in start_points]
                corr_match_eeg_i = np.stack([self.get_corr_coe(EEG, Att)[0] for EEG, Att in zip(EEG_trans_trials, Match_trans_trials)])
                corr_mismatch_eeg_i = np.stack([self.get_corr_coe(EEG, Unatt)[0] for EEG, Unatt in zip(EEG_trans_trials, Mismatch_trans_trials)])
                corr_match_eeg.append(corr_match_eeg_i)
                corr_mismatch_eeg.append(corr_mismatch_eeg_i)
            else:
                print('The length of the video is too short for the given trial length.')
                # return NaN values
                corr_match_eeg.append(np.full((1, self.n_components), np.nan))
                corr_mismatch_eeg.append(np.full((1, self.n_components), np.nan))
        
        corr_match_eeg = np.concatenate(tuple(corr_match_eeg), axis=0)
        corr_mismatch_eeg = np.concatenate(tuple(corr_mismatch_eeg), axis=0)
        return corr_match_eeg, corr_mismatch_eeg

    def att_or_unatt_LVO(self, feat_unatt_list, TRAIN_WITH_ATT, V_eeg=None, V_Stim=None, EEG_ori_list=None):
        # train on attended data and try to decode the attended object
        # train on unattended data and try to decode the unattended object
        # attended and unattended data are both shown on the screen
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
                    _, _, _, _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Att_train)
                else:
                    _, _, _, _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Unatt_train)
            else:
                V_eeg_train, V_feat_train = V_eeg, V_Stim

            corr_att, _, _, _ = self.cal_corr_coe(EEG_test, Att_test, V_eeg_train, V_feat_train)
            corr_unatt, _, _, _ = self.cal_corr_coe(EEG_test, Unatt_test, V_eeg_train, V_feat_train)
            corr_att = np.expand_dims(corr_att, axis=0)
            corr_unatt = np.expand_dims(corr_unatt, axis=0)
            if EEG_ori_test is not None:
                EEG_trans, _ = self.get_transformed_data(EEG_test, Att_test, V_eeg_train, V_feat_train)
                forward_model = self.forward_model(EEG_ori_test, V_eeg_train, X_trans=EEG_trans)
            else:
                forward_model = self.forward_model(EEG_test, V_eeg_train)

            corr_permu_fold.append(self.permutation_test(EEG_test, Att_test, V_A=V_eeg_train, V_B=V_feat_train))
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

    def att_or_unatt_LVO_trials(self, feat_unatt_list, trial_len, BOOTSTRAP=True, V_eeg=None, V_Stim=None):
        feat_att_list = self.Stim_list
        nb_videos = len(feat_att_list)
        nb_folds = nb_videos//self.leave_out
        assert nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
        nested_datalist = [self.EEG_list, feat_att_list, feat_unatt_list, self.mask_list] if self.mask_list is not None else [self.EEG_list, feat_att_list, feat_unatt_list]
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO(nested_datalist, self.leave_out)
        assert len(train_list_folds) == len(test_list_folds) == nb_folds, "The number of folds is not correct."
        corr_att_eeg = []
        corr_unatt_eeg = []
        for idx in range(0, nb_folds):
            if self.mask_list is not None:
                [EEG_train, Att_train, _, self.mask_train], [EEG_test, Att_test, Unatt_test, self.mask_test] = train_list_folds[idx], test_list_folds[idx]
            else:
                [EEG_train,  Att_train, _], [EEG_test, Att_test, Unatt_test] = train_list_folds[idx], test_list_folds[idx]
            if V_eeg is None:
                _, _, _, _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Att_train)
            else:
                 V_eeg_train, V_feat_train = V_eeg, V_Stim
            EEG_trans, Att_trans = self.get_transformed_data(EEG_test, Att_test, V_eeg_train, V_feat_train)
            _, Unatt_trans = self.get_transformed_data(EEG_test, Unatt_test, V_eeg_train, V_feat_train)
            if EEG_trans.shape[0]-trial_len*self.fs >= 0:
                if BOOTSTRAP:
                    nb_trials = EEG_trans.shape[0]//self.fs * 2
                    start_points = np.random.randint(0, EEG_trans.shape[0]-trial_len*self.fs, size=nb_trials)
                else:
                    start_points = None
                EEG_trans_trials = utils.into_trials(EEG_trans, self.fs, trial_len, start_points=start_points)
                Att_trans_trials = utils.into_trials(Att_trans, self.fs, trial_len, start_points=start_points)
                Unatt_trans_trials = utils.into_trials(Unatt_trans, self.fs, trial_len, start_points=start_points)
                corr_att_eeg_i = np.stack([self.get_corr_coe(EEG, Att)[0] for EEG, Att in zip(EEG_trans_trials, Att_trans_trials)])
                corr_unatt_eeg_i = np.stack([self.get_corr_coe(EEG, Unatt)[0] for EEG, Unatt in zip(EEG_trans_trials, Unatt_trans_trials)])
                corr_att_eeg.append(corr_att_eeg_i)
                corr_unatt_eeg.append(corr_unatt_eeg_i)
            else:
                print('The length of the video is too short for the given trial length.')
                # return NaN values
                corr_att_eeg.append(np.full((1, self.n_components), np.nan))
                corr_unatt_eeg.append(np.full((1, self.n_components), np.nan))
        corr_att_eeg = np.concatenate(tuple(corr_att_eeg), axis=0)
        corr_unatt_eeg = np.concatenate(tuple(corr_unatt_eeg), axis=0)
        return corr_att_eeg, corr_unatt_eeg


class GeneralizedCCA:
    def __init__(self, EEG_list, fs, L, offset, hankelized=False, dim_list=None, fold=10, leave_out=2, n_components=5, regularization='lwcov', message=True, signifi_level=True, n_permu=500, p_value=0.05, dim_subspace=4, save_W_perfold=False, crs_val=True):
        '''
        EEG_list: list of EEG data, each element is a T(#sample)xDx(#channel) array
        fs: Sampling rate
        L: If use (spatial-) temporal filter, the number of taps
        offset: If use (spatial-) temporal filter, the offset of the time lags
        fold: Number of folds for cross-validation
        n_components: Number of components to be returned
        regularization: Regularization of the estimated covariance matrix
        message: If print message
        signifi_level: If calculate significance level
        pool: If pool the significance level of all components
        n_permu: Number of permutations for significance level calculation
        p_value: P-value for significance level calculation
        trials: whether segment the test data into trials
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
        lam: eigenvalues, related to mean squared error (not used in analysis)
        W_stack: (Squared_Error_SIGCCAd) weights with shape (D*N*n_components)
        avg_corr: average pairwise correlation
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
    def __init__(self, nested_datalist, fs, L_list, offset_list, fold=10, leave_out=2, n_components=10, regularization='lwcov', message=True, signifi_level=True, n_permu=500, p_value=0.05, dim_subspace=4):
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
        lam: eigenvalues, related to mean squared error (not used in analysis)
        W_stack: (Squared_Error_SIGCCAd) weights with shape (D*N*n_components)
        avg_corr: average pairwise correlation
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


class CorrelatedComponentAnalysis(GeneralizedCCA):
    def fit(self, EEG, W_train=None):
        '''
        Input:
        EEG: EEG data with shape (T, D, N) [T: # sample, D: # channel, N: # subjects]
        W_train: If not None, then goes to test mode.
        Outputs:
        ISC: inter-subject correlation defined in Parra's work, assuming w^T X_n^T X_n w is equal for all subjects
        W: weights
        '''
        T, _, N = EEG.shape
        X_list = [utils.block_Hankel(EEG[:,:,n], self.L, self.offset) for n in range(N)]
        X = np.stack(X_list, axis=2)
        X_center = X - np.mean(X, axis=0, keepdims=True)
        _, D, _ = X.shape
        Rw = np.zeros([D,D])
        for n in range(N):
            if self.regularization == 'lwcov':
                Rw += LedoitWolf().fit(X[:,:,n]).covariance_
            else:
                Rw += np.cov(np.transpose(X[:,:,n])) # Inside np.cov: observations in the columns
        if self.regularization == 'lwcov':
            Rt = N**2*LedoitWolf().fit(np.average(X, axis=2)).covariance_
        else:
            Rt = N**2*np.cov(np.transpose(np.average(X, axis=2)))
        Rb = (Rt - Rw)/(N-1)
        if W_train is not None:
            W = W_train
            ISC = np.diag((np.transpose(W)@Rb@W)/(np.transpose(W)@Rw@W))
            S = None
            F = None
        else:
            ISC, W = eigh(Rb, Rw, subset_by_index=[D-self.n_components,D-1])
            ISC = np.squeeze(np.fliplr(np.expand_dims(ISC, axis=0)))
            W = np.fliplr(W)
            # right scaling
            Lam = np.diag(1/(ISC*(N-1)+1))
            W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rt * T @ W @ Lam))
            # shared subspace
            S = np.sum(X_center, axis=2) @ W @ Lam
            # Forward models
            F_redun = T * Rw @ W / N
            F = utils.F_organize(F_redun, self.L, self.offset)
        return ISC, W, S, F

    def forward_model(self, EEG, W, S=None):
        '''
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
        if S is not None:
            X_stack = np.stack(X_list_center, axis=2)
            F_T = np.mean(np.einsum('kt, tdn -> kdn', S.T, X_stack), axis=2)
            F_redun = F_T.T
        else:
            X = np.concatenate(tuple(X_list_center), axis=0)
            X_transformed = X @ W
            F_redun = (lstsq(X_transformed, X)[0]).T
        F = utils.F_organize(F_redun, self.L, self.offset)
        return F

    def cross_val(self):
        fold = self.fold
        n_components = self.n_components
        corr_train = np.zeros((fold, n_components))
        corr_test = np.zeros((fold, n_components))
        cov_train = np.zeros((fold, n_components))
        cov_test = np.zeros((fold, n_components))
        tsc_train = np.zeros((fold, 1))
        tsc_test = np.zeros((fold, 1))
        isc_train = np.zeros((fold, n_components))
        isc_test = np.zeros((fold, n_components))
        for idx in range(fold):
            train_list, test_list, _, _ = utils.split_mm_balance([self.EEG_list], fold=fold, fold_idx=idx+1)
            isc_train[idx,:], W_train, _, F_train = self.fit(train_list[0])
            isc_test[idx, :], _, _, _ = self.fit(test_list[0], W_train) # Does not have a trial version
            corr_train[idx,:], cov_train[idx,:], _, tsc_train[idx] = self.avg_stats(train_list[0], W_train)
            if self.trials:
                test_trials = utils.into_trials(test_list[0], self.fs)
                corr_test[idx,:], cov_test[idx,:], _, tsc_test[idx] = self.avg_stats_trials(test_trials, W_train)
            else:
                corr_test[idx,:], cov_test[idx,:], _, tsc_test[idx] = self.avg_stats(test_list[0], W_train)
            if not self.crs_val:
                # fill the rest of the folds with the same results
                for i in range(idx+1, fold):
                    corr_train[i,:] = corr_train[idx,:]
                    cov_train[i,:] = cov_train[idx,:]
                    isc_train[i,:] = isc_train[idx,:]
                    tsc_train[i] = tsc_train[idx]
                    corr_test[i,:] = corr_test[idx,:]
                    cov_test[i,:] = cov_test[idx,:]
                    isc_test[i,:] = isc_test[idx,:]
                    tsc_test[i] = tsc_test[idx]
                break
        if self.signifi_level:
            if self.pool:
                if self.trials:
                    corr_trials = self.permutation_test_trials(test_trials, W_train, block_len=1)
                else:
                    corr_trials = self.permutation_test(test_list[0], W_train, block_len=1)
                corr_trials = np.sort(abs(corr_trials), axis=None)
                sig_idx = -int(self.n_permu*self.p_value*n_components)
                print('Significance level: ISC={}'.format(corr_trials[sig_idx]))
            else:
                if self.trials:
                    corr_trials = self.permutation_test_trials(test_trials, W_train, block_len=20*self.fs)
                else:
                    corr_trials = self.permutation_test(test_list[0], W_train, block_len=20*self.fs)
                corr_trials = np.sort(abs(corr_trials), axis=0)
                sig_idx = -int(self.n_permu*self.p_value)
                print('Significance level: ISCs={}'.format(corr_trials[sig_idx,:]))
        if self.message:
            print('Average ISC of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
            print('Average ISC of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
        return corr_train, corr_test, cov_train, cov_test, tsc_train, tsc_test, isc_train, isc_test, W_train, F_train


class StimulusInformedGCCA:
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


class StimulusInformedCorrCA(StimulusInformedGCCA):
    def fit(self, datalist, rho):
        EEG, Stim = datalist
        if np.ndim(EEG) == 2:
            EEG = np.expand_dims(EEG, axis=2)
        T, D_eeg, N = EEG.shape
        _, D_stim = Stim.shape
        L_EEG, L_Stim = self.Llist
        dim_list = [D_eeg*L_EEG] + [D_stim*L_Stim]
        offset_EEG, offset_Stim = self.offsetlist
        EEG_list = [utils.block_Hankel(EEG[:,:,n], L_EEG, offset_EEG) for n in range(N)]
        EEG_Hankel = np.stack(EEG_list, axis=2)
        EEG_center = EEG_Hankel - np.mean(EEG_Hankel, axis=0, keepdims=True)
        Stim_Hankel = utils.block_Hankel(Stim, L_Stim, offset_Stim)
        Stim_center = Stim_Hankel - np.mean(Stim_Hankel, axis=0, keepdims=True)
        Rxx = np.zeros((dim_list[0]+dim_list[1], dim_list[0]+dim_list[1]))
        Rw = np.zeros((dim_list[0], dim_list[0]))
        for n in range(N):
            if self.regularization == 'lwcov':
                cov_eeg = LedoitWolf().fit(EEG_Hankel[:,:,n]).covariance_
            else:
                cov_eeg = np.cov(np.transpose(EEG_Hankel[:,:,n]))
            Rw += cov_eeg
            cov_es = EEG_center[:,:,n].T @ Stim_center / T
            Rxx[0:dim_list[0], dim_list[0]:] += cov_es
            Rxx[dim_list[0]:, 0:dim_list[0]] += cov_es.T
        if self.regularization == 'lwcov':
            Rt = N**2*LedoitWolf().fit(np.average(EEG_Hankel, axis=2)).covariance_
            cov_stim = LedoitWolf().fit(Stim_Hankel).covariance_
        else:
            Rt = N**2*np.cov(np.transpose(np.average(EEG_Hankel, axis=2)))
            cov_stim = np.cov(np.transpose(Stim_Hankel))
        Rxx[0:dim_list[0], 0:dim_list[0]] = Rt
        Rxx[dim_list[0]:, dim_list[0]:] = cov_stim
        Dxx = np.zeros_like(Rxx)
        Dxx[0:dim_list[0], 0:dim_list[0]] = Rw
        Dxx[dim_list[0]:, dim_list[0]:] = cov_stim
        try:
            lam, W = utils.transformed_GEVD(Dxx, Rxx, rho, dim_list[-1], self.n_components)
            Lam = np.diag(lam)
            Rxx[:,dim_list[0]:] = Rxx[:,dim_list[0]:]*rho
        except:
            print("Numerical issue exists for eigh. Use eig instead.")
            Rxx[:,dim_list[0]:] = Rxx[:,dim_list[0]:]*rho
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
        Weeg = W[0:dim_list[0],:]
        Weeg_expand = np.repeat(np.expand_dims(Weeg, axis=1), N, axis=1)
        Wstim = W[dim_list[0]:,:]
        # Forward models
        F_redun = T * Dxx @ W / N
        F = utils.F_organize(F_redun[0:dim_list[0],:], L_EEG, offset_EEG)
        # Shared subspace
        S = self.shared_subspace(EEG_center, Stim_center, Weeg, Wstim, Lam, rho)
        Wlist = [Weeg_expand, Wstim]
        return Wlist, S, F, lam

    def shared_subspace(self, EEG_center, Stim_center, Weeg, Wstim, Lam, rho):
        S = (np.sum(np.einsum('tdn,dk->tkn', EEG_center, Weeg), axis=2) + rho*Stim_center@Wstim) @ Lam
        return S

    def forward_model(self, EEG, Wlist, S=None):
        '''
        Input:
        EEG: EEG data with shape (T, D, N) [T: # sample, D: # channel, N: # subjects]
        Wlist: [Weeg_expand, Wstim]
        S: shared subspace with shape (T, n_components); if not None, then calculate forward model from the shared subspace
        Note: if S is not None, then EEG must be the one used in the training stage. So it is equivalent to the S generated by self.fit(). This can be used as a sanity check.
        Outputs:
        F: forward model (D, n_components)
        '''
        if np.ndim(EEG) == 2:
            EEG = np.expand_dims(EEG, axis=2)
        W = Wlist[0][:,0,:]
        _, _, N = EEG.shape
        X_list = [utils.block_Hankel(EEG[:,:,n], self.Llist[0], self.offsetlist[0]) for n in range(N)]
        X_list_center = [X_list[n] - np.mean(X_list[n], axis=0, keepdims=True) for n in range(N)]
        if S is not None:
            X_stack = np.stack(X_list_center, axis=2)
            F_T = np.mean(np.einsum('kt, tdn -> kdn', S.T, X_stack), axis=2)
            F_redun = F_T.T
        else:
            X = np.concatenate(tuple(X_list_center), axis=0)
            X_transformed = X @ W
            F_redun = (lstsq(X_transformed, X)[0]).T
        F = utils.F_organize(F_redun, self.Llist[0], self.offsetlist[0])
        return F


class GCCAPreprocessedCCA:
    def __init__(self, Subj_ID, eeg_multisubj_list, stim_list, fs, para_gcca, para_cca_eeg, para_cca_stim, W_GCCA_folds=None, preprocessed_train_folds=None, preprocessed_test_folds = None, leave_out=2, fold=None, n_components=5, regularization='lwcov', K_regu=None, message=True, signifi_level=True, n_permu=500, p_value=0.05):
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
        self.n_components = n_components
        self.regularization = regularization
        self.K_regu = K_regu
        self.message = message
        self.signifi_level = signifi_level
        self.n_permu = n_permu
        self.p_value = p_value

    def get_GCCA_results(self):
        L_EEG, offset_EEG = self.para_gcca
        self.W_GCCA_folds = []
        self.preprocessed_train_folds = []
        self.preprocessed_test_folds = []
        GCCA = GeneralizedCCA(self.eeg_multisubj_list, self.fs, L_EEG, offset_EEG, n_components=self.n_components, regularization=self.regularization)
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

    def cross_val(self):
        print('Getting GCCA results...')
        if self.W_GCCA_folds is None:
            _, _, _ = self.get_GCCA_results()
        L_EEG, offset_EEG = self.para_gcca
        L_Stim, offset_Stim = self.para_cca_stim
        nb_folds = len(self.preprocessed_train_folds)
        n_components = self.n_components
        corr_train_fold = np.zeros((nb_folds, n_components))
        corr_test_fold = np.zeros((nb_folds, n_components))
        forward_model_fold = []
        EEG_comp_fold = []
        print('After GCCA preprocessing, getting CCA cross-validation results...')
        CCA = CanonicalCorrelationAnalysis(self.eeg_onesubj_list, self.stim_list, self.fs, L_EEG, L_Stim, offset_EEG, offset_Stim, fold=self.fold, leave_out=self.leave_out, n_components=self.n_components, regularization=self.K_regu, K_regu=self.K_regu, message=self.message, signifi_level=self.signifi_level, n_permu=self.n_permu, p_value=self.p_value)
        idx = 0
        for train_mm, test_mm in tqdm(zip(self.preprocessed_train_folds, self.preprocessed_test_folds)):
            [EEG_trans_train, Stim_train], [EEG_trans_test, Stim_test] = train_mm, test_mm
            EEG_train = EEG_trans_train[:,:,self.Subj_ID]
            EEG_test = EEG_trans_test[:,:,self.Subj_ID]
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

    def att_or_unatt_trials(self, feat_unatt_list, trial_len, BOOTSTRAP=True):
        print('Getting GCCA results...')
        if self.W_GCCA_folds is None:
            _, _, _ = self.get_GCCA_results()
        L_EEG, offset_EEG = self.para_gcca
        L_Stim, offset_Stim = self.para_cca_stim
        nb_folds = len(self.preprocessed_train_folds)
        train_unatt_folds, test_unatt_folds = utils.split_mm_balance_folds([feat_unatt_list], self.fold) if self.fold is not None else utils.split_multi_mod_LVO([feat_unatt_list], self.leave_out)
        print('After GCCA preprocessing, getting CCA results for attended and unattended trials...')
        CCA = CanonicalCorrelationAnalysis(self.eeg_onesubj_list, self.stim_list, self.fs, L_EEG, L_Stim, offset_EEG, offset_Stim, fold=self.fold, leave_out=self.leave_out, n_components=self.n_components, regularization=self.K_regu, K_regu=self.K_regu, message=self.message, signifi_level=self.signifi_level, n_permu=self.n_permu, p_value=self.p_value)
        corr_att_eeg = []
        corr_unatt_eeg = []
        for idx in range(nb_folds):
            [EEG_trans_train, Att_train], [EEG_trans_test, Att_test] = self.preprocessed_train_folds[idx], self.preprocessed_test_folds[idx]
            _, [Unatt_test] = train_unatt_folds[idx], test_unatt_folds[idx]
            EEG_train = EEG_trans_train[:,:,self.Subj_ID]
            EEG_test = EEG_trans_test[:,:,self.Subj_ID]
            _, _, _, _, V_eeg_train, V_feat_train, _ = CCA.fit(EEG_train, Att_train)
            EEG_comp, Att_comp = CCA.get_transformed_data(EEG_test, Att_test, V_eeg_train, V_feat_train)
            _, Unatt_comp = CCA.get_transformed_data(EEG_test, Unatt_test, V_eeg_train, V_feat_train)
            if EEG_comp.shape[0]-trial_len*self.fs >= 0:
                if BOOTSTRAP:
                    nb_trials = EEG_comp.shape[0]//self.fs * 2
                    start_points = np.random.randint(0, EEG_comp.shape[0]-trial_len*self.fs, size=nb_trials)
                else:
                    start_points = None
                EEG_comp_trials = utils.into_trials(EEG_comp, self.fs, trial_len, start_points=start_points)
                Att_comp_trials = utils.into_trials(Att_comp, self.fs, trial_len, start_points=start_points)
                Unatt_comp_trials = utils.into_trials(Unatt_comp, self.fs, trial_len, start_points=start_points)
                corr_att_eeg_i = np.stack([CCA.get_corr_coe(EEG, Att)[0] for EEG, Att in zip(EEG_comp_trials, Att_comp_trials)])
                corr_unatt_eeg_i = np.stack([CCA.get_corr_coe(EEG, Unatt)[0] for EEG, Unatt in zip(EEG_comp_trials, Unatt_comp_trials)])
                corr_att_eeg.append(corr_att_eeg_i)
                corr_unatt_eeg.append(corr_unatt_eeg_i)
            else:
                print('The length of the video is too short for the given trial length.')
                # return NaN values
                corr_att_eeg.append(np.full((1, self.n_components), np.nan))
                corr_unatt_eeg.append(np.full((1, self.n_components), np.nan))
        corr_att_eeg = np.concatenate(tuple(corr_att_eeg), axis=0)
        corr_unatt_eeg = np.concatenate(tuple(corr_unatt_eeg), axis=0)
        return corr_att_eeg, corr_unatt_eeg


class LSGCCA:
    def __init__(self, EEG_list, Stim_list, fs, L_EEG, L_Stim, offset_EEG, offset_Stim, id_sub, corrca=False, fold=10, n_components=5, regularization='lwcov', message=True, signifi_level=True, pool=True, n_permu=500, p_value=0.05):
        self.EEG_list = EEG_list
        self.Stim_list = Stim_list
        self.fs = fs
        self.L_EEG = L_EEG
        self.offset_EEG = offset_EEG
        self.L_Stim = L_Stim
        self.offset_Stim = offset_Stim
        self.id_sub = id_sub
        self.corrca = corrca
        self.fold = fold
        self.n_components = n_components
        self.regularization = regularization
        self.message = message
        self.signifi_level = signifi_level
        self.pool = pool
        self.n_permu = n_permu
        self.p_value = p_value

    def get_transformed_EEG(self, EEG, We):
        EEG_Hankel = utils.block_Hankel(EEG, self.L_EEG, self.offset_EEG)
        EEG_Hankel_center = EEG_Hankel - np.mean(EEG_Hankel, axis=0, keepdims=True)
        filtered_EEG = EEG_Hankel_center @ We
        return filtered_EEG

    def get_transformed_data(self, EEG, Stim, We, Ws):
        filtered_EEG = self.get_transformed_EEG(EEG, We)
        Stim_Hankel = utils.block_Hankel(Stim, self.L_Stim, self.offset_Stim)
        filtered_Stim = Stim_Hankel @ Ws
        return filtered_EEG, filtered_Stim

    def correlation(self, EEG, Stim, We, Ws):
        filtered_EEG, filtered_Stim = self.get_transformed_data(EEG, Stim, We, Ws)
        corr_pvalue = [pearsonr(filtered_EEG[:,k], filtered_Stim[:,k]) for k in range(self.n_components)]
        corr_coe = np.array([corr_pvalue[k][0] for k in range(self.n_components)])
        p_value = np.array([corr_pvalue[k][1] for k in range(self.n_components)])
        return corr_coe, p_value

    def permutation_test(self, EEG, Stim, We, Ws, block_len):
        corr_coe_topK = np.zeros((self.n_permu, self.n_components))
        filtered_EEG, filtered_Stim = self.get_transformed_data(EEG, Stim, We, Ws)
        for i in tqdm(range(self.n_permu)):
            EEG_shuffled = utils.shuffle_2D(filtered_EEG, block_len)
            Stim_shuffled = utils.shuffle_2D(filtered_Stim, block_len)
            corr_pvalue = [pearsonr(EEG_shuffled[:,k], Stim_shuffled[:,k]) for k in range(self.n_components)]
            corr_coe_topK[i,:] = np.array([corr_pvalue[k][0] for k in range(self.n_components)])
        return corr_coe_topK

    def to_latent_space(self):
        self.nested_train = []
        self.nested_test = []
        self.nested_We_train = []
        self.nested_S = []
        self.nested_F_train = []
        for idx in range(self.fold):
            train_list, test_list, _, _ = utils.split_mm_balance([self.EEG_list, self.Stim_list], fold=self.fold, fold_idx=idx+1)
            if self.corrca:
                CorrCA = CorrelatedComponentAnalysis(self.EEG_list, self.fs, self.L_EEG, self.offset_EEG, n_components=self.n_components, regularization=self.regularization)
                _, We_train, S, F_train = CorrCA.fit(train_list[0])
                We_train = np.expand_dims(We_train, axis=1)
                We_train = np.repeat(We_train, train_list[0].shape[2], axis=1)
            else:
                GCCA = GeneralizedCCA(self.EEG_list, self.fs,self.L_EEG, self.offset_EEG, n_components=self.n_components, regularization=self.regularization)
                We_train, S, F_train, _ = GCCA.fit(train_list[0])
            self.nested_train.append(train_list)
            self.nested_test.append(test_list)
            self.nested_We_train.append(We_train)
            self.nested_S.append(S)
            self.nested_F_train.append(F_train)

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
            EEG_trans = self.get_transformed_EEG(train_list[0][:,:,self.id_sub], We_train[:,self.id_sub,:])
            Ws_train, _, = LS.encoder(EEG_trans, train_list[1])
            # Ws_train, _, = LS.encoder(S, train_list[1])
            corr_train[idx,:], _ = self.correlation(train_list[0][:,:,self.id_sub], train_list[1], We_train[:,self.id_sub,:], Ws_train)
            corr_test[idx,:], _ = self.correlation(test_list[0][:,:,self.id_sub], test_list[1], We_train[:,self.id_sub,:], Ws_train)
        if self.signifi_level:
            if self.pool:
                corr_trials = self.permutation_test(test_list[0][:,:,self.id_sub], test_list[1], We_train[:,self.id_sub,:], Ws_train, block_len=1)
                corr_trials = np.sort(abs(corr_trials), axis=None)
                sig_idx = -int(self.n_permu*self.p_value*n_components)
                sig_corr = corr_trials[sig_idx]
                print('Significance level: {}'.format(sig_corr))
            else:
                corr_trials = self.permutation_test(test_list[0][:,:,self.id_sub], test_list[1], We_train[:,self.id_sub,:], Ws_train, block_len=1)
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
    

class LSGCCA_Group(LSGCCA):
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