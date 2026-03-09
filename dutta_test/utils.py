import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as stats
from scipy.stats import mannwhitneyu, wilcoxon,rankdata
def barplot_annotate_brackets(num1, num2, data, center, height, ax, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    if type(data) is str:
        text = data
    else:
        if data > 0.05:
          return
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)

   
def cliffs_delta(u_stat, n_A, n_B):
     # Compute_cliff_delta:
    U1 = u_stat
    U2 = n_A * n_B - U1

    # Cliff’s delta must use U1 (vals_A vs vals_B)
    cliffs_d = (U1 - U2) / (n_A * n_B)
    # Interpretation (Romano et al., 2006)
    abs_d = abs(cliffs_d)
    if abs_d < 0.147:
        effect = "Negligible"
    elif abs_d < 0.33:
        effect = "Small"
    elif abs_d < 0.474:
        effect = "Medium"
    else:
        effect = "Large"
        
    # print(f"Cliff's Delta: {cliffs_d:.3f} ({effect})")

    return (cliffs_d,effect)

def effect_size_r(u_stat, n_A, n_B):
    # A metric to use with Mann-Whitney U test is the effect size r:
    # Compute_effect_size_r:
    z=(u_stat-n_A*n_B/2)/np.sqrt(n_A * n_B * (n_A + n_B + 1) / 12)
    r=z/np.sqrt(n_A + n_B)

    # Interpretation (Cohen, 1988)
    if abs(r) < 0.1:
        effect = "Negligible"
    elif abs(r) < 0.3:
        effect = "Small"
    elif abs(r) < 0.5:
        effect = "Medium"
    else:
        effect = "Large"
    # print(f"Effect Size r: {r:.3f} ({effect})")

    return (r,effect)

def vargha_delaney(u_stat, n_A, n_B):
    # A non-parametric effect size measure for the Mann-Whitney U test is the Vargha-Delaney A measure:
    # Compute_Vargha_Delaney_A:
    A = u_stat / (n_A * n_B)

    # Interpretation (Vargha & Delaney, 2000)
    if 0.44 < A < 0.56:
        effect = "Negligible"
    elif 0.56 <= A < 0.64 or 0.36 < A <= 0.44:
        effect = "Small"
    elif 0.64 <= A < 0.71 or 0.29 < A <= 0.36:
        effect = "Medium"
    else:
        effect = "Large"
    # print(f"Vargha-Delaney A: {A:.3f} ({effect})")

    return (A,effect)

def effect_size_r_wilcoxon(x, y):
    
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    N = len(diff)

    stat, p = wilcoxon(x, y)

    mu = N * (N + 1) / 4
    sigma = np.sqrt(N * (N + 1) * (2*N + 1) / 24)

    z = (stat - mu) / sigma
    r = z / np.sqrt(N)

    if abs(r) < 0.1:
        effect = "Negligible"
    elif abs(r) < 0.3:
        effect = "Small"
    elif abs(r) < 0.5:
        effect = "Medium"
    else:
        effect = "Large"

    return r, effect

def probability_of_superiority_wilcoxon(x, y):
    x, y = np.array(x), np.array(y)
    diff = x - y
    # Remove ties as per Wilcoxon standard (Pratt method is an alternative)
    diff = diff[diff != 0]
    ties=diff[diff == 0]
    N = len(diff)
    
    if N == 0:
        return 0.5, "Negligible"

    # Get the sum of positive ranks specifically
    ranks = rankdata(np.abs(diff))
    w_plus = np.sum(ranks[diff > 0])
    
    # Total possible rank sum
    w_total = N * (N + 1) / 2
    
    # Probability of Superiority (A)
    # This represents P(X > Y)
    A = (w_plus+ 0.5 * len(ties)) / w_total
    
    # Classification based on Vargha and Delaney (2000)
    # We use the distance from 0.5 to determine magnitude
    val = abs(A - 0.5)
    
    if val < 0.06: # 0.44 to 0.56
        effect = "Negligible"
    elif val < 0.14: # 0.36 to 0.64
        effect = "Small"
    elif val < 0.21: # 0.29 to 0.71
        effect = "Medium"
    else:
        effect = "Large"

    return A, effect


def calculate_matched_pairs_effect_size_wilcoxon(x, y):
    x, y = np.array(x), np.array(y)
    diff = x - y
    diff = diff[diff != 0] # Remove ties
    N = len(diff)
    
    if N == 0:
        return 0, "Negligible"

    ranks = rankdata(np.abs(diff))
    R_plus = np.sum(ranks[diff > 0])
    R_minus = np.sum(ranks[diff < 0])
    total_rank_sum = R_plus + R_minus
    
    # 1. Matched-pairs rank biserial correlation (rc
    # If all values in x are greater than y, R_plus = total_rank_sum and R_minus = 0, giving rc = 1 (perfect positive association).
    rc = (R_plus - R_minus) / total_rank_sum
    

    
    # 2. Interpretation (using rc absolute thresholds from your text)
    abs_rc = abs(rc)
    if abs_rc < 0.11:
        effect = "Negligible"
    elif abs_rc < 0.28:
        effect = "Small"
    elif abs_rc < 0.43:
        effect = "Medium"
    else:
        effect = "Large"
        
    return rc, effect