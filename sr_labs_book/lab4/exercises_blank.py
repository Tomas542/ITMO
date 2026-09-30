# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow


def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram
    
    tar_scores = []
    imp_scores = []

    ###########################################################
    for score, label in zip(all_scores, all_labels):
        if label == 1:
            tar_scores.append(score)
        else:
            imp_scores.append(score)
    ###########################################################
    
    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)
    
    return tar_scores, imp_scores

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio
    
    tar_scores_mean = np.mean(tar_scores)
    tar_scores_std  = np.std(tar_scores)
    imp_scores_mean = np.mean(imp_scores)
    imp_scores_std  = np.std(imp_scores)
    
    all_scores_sort   = np.zeros(len(all_scores))
    ground_truth_sort = np.zeros(len(all_scores), dtype='bool')
    
    ###########################################################
    order = np.argsort(all_scores)
    all_scores_arr = np.asarray(all_scores)
    all_labels_arr = np.asarray(all_labels)
    all_scores_sort = all_scores_arr[order]
    ground_truth_sort = all_labels_arr[order].astype(bool)
    ###########################################################
    
    tar_gauss_pdf = np.zeros(len(all_scores))
    imp_gauss_pdf = np.zeros(len(all_scores))
    LLR           = np.zeros(len(all_scores))
    
    ###########################################################
    tar_gauss_pdf = gauss_pdf(all_scores_sort, tar_scores_mean, tar_scores_std)
    imp_gauss_pdf = gauss_pdf(all_scores_sort, imp_scores_mean, imp_scores_std)
    LLR = np.log(tar_gauss_pdf) - np.log(imp_gauss_pdf)
    ###########################################################
    
    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test
    
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        err = (solution != ground_truth_sort)                          # error vector
        
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        
        P_err[idx]   = fnr_thr[idx]*P_Htar + fpr_thr[idx]*(1 - P_Htar) # prob. of error
    
    # Plot error's prob.
    plot(LLR, P_err, color='blue')
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()
        
    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = fnr_thr[P_err_idx]*P_Htar + fpr_thr[P_err_idx]*(1 - P_Htar)
    
    return LLR[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test
    
    thr   = 0.0
    fpr   = 0.0
    
    ###########################################################
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)

    for idx in range(len_thr):
        solution = LLR > LLR[idx]
        err = (solution != ground_truth_sort)
        fnr_thr[idx] = np.sum(err[ ground_truth_sort]) / len(tar_scores)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort]) / len(imp_scores)

    best = np.argmin(np.abs(fnr_thr - fnr))
    thr = LLR[best]
    fpr = fpr_thr[best]
    ###########################################################
    
    return thr, fpr

def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):
    # Function to perform Bayes' test
    
    thr   = 0.0
    fnr   = 0.0
    fpr   = 0.0
    AC    = 0.0
    
    ###########################################################
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    AC_thr  = np.zeros(len_thr)

    for idx in range(len_thr):
        solution = LLR > LLR[idx]
        err = (solution != ground_truth_sort)
        fnr_thr[idx] = np.sum(err[ ground_truth_sort]) / len(tar_scores)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort]) / len(imp_scores)
        AC_thr[idx] = (C00 * (1 - fnr_thr[idx]) + C10 * fnr_thr[idx]) * P_Htar + \
                      (C11 * (1 - fpr_thr[idx]) + C01 * fpr_thr[idx]) * (1 - P_Htar)

    best = np.argmin(AC_thr)
    thr = LLR[best]
    fnr = fnr_thr[best]
    fpr = fpr_thr[best]
    AC  = AC_thr[best]
    ###########################################################
    
    return thr, fnr, fpr, AC

def minmax_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    # Function to perform minimax test
    
    thr    = 0.0
    fnr    = 0.0
    fpr    = 0.0
    AC     = 0.0
    P_Htar = 0.0
    
    ###########################################################
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)

    for idx in range(len_thr):
        solution = LLR > LLR[idx]
        err = (solution != ground_truth_sort)
        fnr_thr[idx] = np.sum(err[ ground_truth_sort]) / len(tar_scores)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort]) / len(imp_scores)

    worst = -1.0
    worst_idx = 0
    for prior in P_Htar_thr:
        AC_thr = (C00 * (1 - fnr_thr) + C10 * fnr_thr) * prior + \
                 (C11 * (1 - fpr_thr) + C01 * fpr_thr) * (1 - prior)
        best = np.argmin(AC_thr)
        if AC_thr[best] > worst:
            worst = AC_thr[best]
            worst_idx = best
            P_Htar = prior

    thr = LLR[worst_idx]
    fnr = fnr_thr[worst_idx]
    fpr = fpr_thr[worst_idx]
    AC  = worst
    ###########################################################
    
    return thr, fnr, fpr, AC, P_Htar