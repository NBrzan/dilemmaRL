# import multiprocessing
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os
from utils import *
from joblib import Parallel, delayed
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')


USE_PARALLEL = False
GLOBAL_REP_REGISTRY = {}
RESET_REPUTATION_FOR_IPD_RUNS = True
ENABLE_REPUTATION = True

def run_ipd_sequential(ipd_scenario, alg_list, nMemory, prefix):
    results = []
    for algs in tqdm(alg_list):
        if ENABLE_REPUTATION:
            reputations = np.zeros(len(algs))
            rep_counts = np.zeros(len(algs), dtype=int)
            
            for i, alg in enumerate(algs):
                if alg in GLOBAL_REP_REGISTRY:
                    reputations[i] = GLOBAL_REP_REGISTRY[alg][0]
                    rep_counts[i] = GLOBAL_REP_REGISTRY[alg][1]
        else:
            reputations = None
            rep_counts = None
        
        res = run_ipd(ipd_scenario, algs, nMemory=nMemory, prefix=prefix, 
                      reputations=reputations, rep_counts=rep_counts)
        
        *core_res, updated_reps, updated_counts = res
        results.append(core_res)
        
        # Update registry
        if ENABLE_REPUTATION:
            for i, alg in enumerate(algs):
                GLOBAL_REP_REGISTRY[alg] = [updated_reps[i], updated_counts[i]]
            
    return results


def run_ipd_parallel(ipd_scenario, alg_list, nMemory, prefix):
    full_results = Parallel(n_jobs=-1)(delayed(run_ipd)(ipd_scenario, algs, nMemory=nMemory, prefix=prefix, reputations=None, rep_counts=None, enable_reputation=ENABLE_REPUTATION) for algs in tqdm(alg_list))
    return [res[:9] for res in full_results]



def save_results_to_csv(tab, alg_list, path):
    rows = []
    
    r_arr = tab['r']
    r_shape = r_arr.shape
    dims = len(r_shape) - 1
    T = r_shape[-1]
    
    
    for idx in np.ndindex(r_shape[:-1]):

        base_data = {
            'coop_ratio': tab['coop'][idx] if 'coop' in tab else 0,
            'conv_round': tab['conv'][idx] if 'conv' in tab else 0,
        }
        
        if dims == 2:
            if r_shape[0] == len(alg_list): # Tournament
                base_data['alg1'] = alg_list[idx[0]]
                base_data['alg2'] = alg_list[idx[1]]
            else:
                base_data['sample_id'] = idx[0] # BC
                base_data['alg'] = alg_list[idx[1]]
        elif dims == 3:
            base_data['alg1'] = alg_list[idx[0]] if idx[0] < len(alg_list) else f'Idx_{idx[0]}'# 3-agent
            base_data['alg2'] = alg_list[idx[1]] if idx[1] < len(alg_list) else f'Idx_{idx[1]}'
            base_data['alg3'] = alg_list[idx[2]] if idx[2] < len(alg_list) else f'Idx_{idx[2]}'
        
        if dims == 3 and base_data['coop_ratio'] == 0 and np.all(r_arr[idx] == 0):
            continue

        for t in range(T):
            row = base_data.copy()
            row['timestep'] = t
            full_idx = idx + (t,)
            if 'r' in tab: row['reward'] = r_arr[full_idx]
            if 'p' in tab: row['coop_pct'] = tab['p'][full_idx]
            rows.append(row)
            
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'Saved results to {path}')

def run_ipd_all(ipd_scenario, alg_list, nMemory, prefix):
    if RESET_REPUTATION_FOR_IPD_RUNS and ENABLE_REPUTATION:
        GLOBAL_REP_REGISTRY.clear()

    if USE_PARALLEL:
        return run_ipd_parallel(ipd_scenario, alg_list, nMemory, prefix)
    else:
        return run_ipd_sequential(ipd_scenario, alg_list, nMemory, prefix)


def run_bclone_single_alg(alg1, train_set, test_set, ipd_scenario, fd, nTrials, T, nMemory):
    from scipy.stats.stats import pearsonr
    algs = [alg1, 'Human']
    _, reward_from_A, reward_from_B, reward_from_C, reward_from_D = load_IPD(
        ipd_scenario, prefix=fd)
    reward_functions = (reward_from_A, reward_from_B,
                        reward_from_C, reward_from_D)
    ipd_case = IPD(algs, reward_functions, nTrials, T, nMemory=nMemory)
    train_data = None
    rep = None
    for j in np.arange(train_set.shape[0]):
        train_data = train_set[j, :, :]
        ipd_case.loadTraj(train_data, True)
        rep = ipd_case.run()
    ipd_case.pauseLearn()

    # Save the trained model results from the last training run
    path = './models/' + fd + '/trained_IPD_' + \
        str(ipd_scenario) + '_m_' + str(nMemory) + \
        '_p_' + '_'.join(algs) + '.csv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # with open(path, 'wb') as handle: (Replaced by CSV)
        # Results are now saved via save_results_to_csv later

    test_size = test_set.shape[0]
    alg_results = {
        'r': np.zeros((test_size, T)),
        'rstd': np.zeros((test_size, T)),
        'p': np.zeros((test_size, T)),
        'pstd': np.zeros((test_size, T)),
        's': np.zeros((test_size, T)),
        'sstd': np.zeros((test_size, T)),
        'd': np.zeros((test_size, T)),
        'dstd': np.zeros((test_size, T)),
        'pr': np.zeros(test_size),
        'coop_ratios': np.zeros(test_size), # cooperation ratio metric
        'convergence_rounds': np.zeros(test_size, dtype=int) # convergence round metric
    }

    # function that computes cooperation ratio and convergence round for each algorithm in the IPD experiment
    def _compute_cooperation_and_convergence(rep_local, agent_id=0, epsilon=0.01):
        percentage = np.mean(rep_local['percent' + str(agent_id)], axis=0) / 100.0
        coop = float(np.mean(percentage))
        if epsilon > 0:
            window = int(np.ceil(1.0 / epsilon))
        else:
            window = 1

        if window > len(percentage):
            window = len(percentage)

        conv = len(percentage)
        for t in range(0, max(1, len(percentage) - window + 1)):
            if np.all(np.abs(percentage[t:t + window] - coop) <= epsilon):
                conv = t + 1
                break

        return coop, int(conv)

    for k in np.arange(test_size):
        test_data = test_set[k, :, :]
        p_cor = np.array([np.sum(test_data[0, :t + 1] == 1)
                         * 100 / (t + 1) for t in range(T)])

        # Uses the last train_data from the training loop
        ipd_case.loadTraj(train_data, True)
        rep = ipd_case.run()
        rep['algs'] = algs
        rep['nTrials'] = nTrials
        rep['T'] = T
        rep['min_r'] = np.min([np.sum(reward_from_A(0)), np.sum(
            reward_from_B(0)), np.sum(reward_from_C(0)), np.sum(reward_from_D(0))])
        rep['max_r'] = np.max([np.sum(reward_from_A(0)), np.sum(
            reward_from_B(0)), np.sum(reward_from_C(0)), np.sum(reward_from_D(0))])

        fig_p = './figures/' + fd + '/test_' + \
            str(k) + '_p_IPD_' + str(ipd_scenario) + \
            '_m_' + str(nMemory) + '_' + '_'.join(algs)
        plot_p(rep, fig_p)

        rs, r, p, r_std, p_std = [], [], [], [], []
        for l in np.arange(len(algs)):
            one_rs = (rep['reward' + str(l)] - rep['min_r']) / \
                (rep['max_r'] - rep['min_r'])  # inline norm_r
            rs.append(one_rs)
            r.append(np.cumsum(np.mean(one_rs, 0)))
            p.append(np.mean(rep['percent' + str(l)], 0))
            r_std.append(np.std(one_rs, 0) / np.sqrt(nTrials))
            p_std.append(np.std(rep['percent' + str(l)], 0) / np.sqrt(nTrials))

        r, p, r_std, p_std, rs = np.array(r), np.array(
            p), np.array(r_std), np.array(p_std), np.array(rs)
        r_sum = np.sum(rs, 0)
        rs_sum = np.mean(r_sum, 0)
        rs_std = np.std(r_sum, 0) / np.sqrt(nTrials)
        p_dff = p - p_cor
        rs_dff = np.mean(p_dff, 1)
        rd_std = np.std(p_dff, 1) / np.sqrt(nTrials)

        coop_k, conv_k = _compute_cooperation_and_convergence(rep)
        alg_results['coop_ratios'][k] = coop_k
        alg_results['convergence_rounds'][k] = conv_k

        alg_results['r'][k, :] = r[0]
        alg_results['rstd'][k, :] = r_std[0]
        alg_results['p'][k, :] = p[0]
        alg_results['pstd'][k, :] = p_std[0]
        alg_results['s'][k, :] = rs_sum
        alg_results['sstd'][k, :] = rs_std
        alg_results['d'][k, :] = rs_dff[0]
        alg_results['dstd'][k, :] = rd_std[0]
        alg_results['pr'][k] = pearsonr(p[0], p_cor)[0] if np.isnan(
            pearsonr(p[0], p_cor)[0]) else 0

    return alg_results


SMALL_SIZE = 40
MEDIUM_SIZE = 50
BIGGER_SIZE = 60
IPD_SCENARIO = 1 # 1 = iterative 

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

MAB_algs = ['UCB', 'TS', 'eGreedy', 'EXP3', 'HBTS']
CB_algs = ['LinUCB', 'CTS', 'EXP4', 'SCTS']
RL_algs = ['QL', 'DQL', 'SARSA', 'SQL']
HC_algs = ['Coop', 'Dfct', 'Tit4Tat']
all_algs = ['UCB', 'TS', 'eGreedy', 'EXP3', 'HBTS', 'LinUCB', 'CTS',
            'EXP4', 'SCTS', 'QL', 'DQL', 'SARSA', 'SQL', 'Coop', 'Dfct', 'Tit4Tat']
agent_algs = ['UCB', 'TS', 'eGreedy', 'EXP3', 'HBTS', 'LinUCB',
              'CTS', 'EXP4', 'SCTS', 'QL', 'DQL', 'SARSA', 'SQL']

mMAB_algs = ['HBTS', 'bAD', 'bADD', 'bADHD', 'bbvFTD', 'bCP', 'bM', 'bPD']
mCB_algs = ['SCTS', 'cAD', 'cADD', 'cADHD', 'cbvFTD', 'cCP', 'cM', 'cPD']
mRL_algs = ['SQL', 'AD', 'ADD', 'ADHD', 'bvFTD', 'CP', 'M', 'PD']

# Case with 2 agents

fd = 'ipd1_m5'
nMemory = 5
print(f"\n Starting 2-agent tournament (nMemory={nMemory}, prefix={fd}) ")
T = 50
ALGS = all_algs
tab_r = np.zeros((len(ALGS), len(ALGS), T))
tab_r_std = np.zeros((len(ALGS), len(ALGS), T))
tab_p = np.zeros((len(ALGS), len(ALGS), T))
tab_p_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_sum = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_dff = np.zeros((len(ALGS), len(ALGS), T))
tab_rd_std = np.zeros((len(ALGS), len(ALGS), T))
tab_coop_ratio = np.zeros((len(ALGS), len(ALGS))) # cooperation ratio metric
tab_conv_round = np.zeros((len(ALGS), len(ALGS)), dtype=int) # convergence round metric

alg_pairs = [[alg1, alg2] for alg1 in ALGS for alg2 in ALGS]
results = run_ipd_all(IPD_SCENARIO, alg_pairs, nMemory, fd)

for idx, (alg1, alg2) in enumerate(alg_pairs):
    i = idx // len(ALGS)
    j = idx % len(ALGS)
    r, r_std, p, p_std, rs_sum, rs_std, rs_dff, rd_std, rep = results[idx]
    tab_r[i, j, :] = r[0]
    tab_r_std[i, j, :] = r_std[0]
    tab_p[i, j, :] = p[0]
    tab_p_std[i, j, :] = p_std[0]
    tab_r[j, i, :] = r[1]
    tab_r_std[j, i, :] = r_std[1]
    tab_p[j, i, :] = p[1]
    tab_p_std[j, i, :] = p_std[1]
    tab_rs_sum[i, j, :] = tab_rs_sum[j, i, :] = rs_sum
    tab_rs_std[i, j, :] = tab_rs_std[j, i, :] = rs_std
    tab_rs_dff[i, j, :] = rs_dff[0]
    tab_rd_std[i, j, :] = rd_std[0]
    tab_rs_dff[j, i, :] = rs_dff[1]
    tab_rd_std[j, i, :] = rd_std[1]
    if 'coop_ratios' in rep and 'convergence_rounds' in rep:
        tab_coop_ratio[i, j] = rep['coop_ratios'][0]
        tab_coop_ratio[j, i] = rep['coop_ratios'][1]
        tab_conv_round[i, j] = rep['convergence_rounds'][0]
        tab_conv_round[j, i] = rep['convergence_rounds'][1]

tab = {'r': tab_r, 'rstd': tab_r_std, 'p': tab_p, 'pstd': tab_p_std,
       's': tab_rs_sum, 'sstd': tab_rs_std, 'd': tab_rs_dff, 'dstd': tab_rd_std, 'coop': tab_coop_ratio, 'conv': tab_conv_round}
path = './models/ipd1_m5.csv'
os.makedirs(os.path.dirname(path), exist_ok=True)
save_results_to_csv(tab, ALGS, path)

# Case with 3 agents

fd = 'ipd1_m5_3ag'
nMemory = 5
print(f"\n Starting 3-agent tournament (nMemory={nMemory}, prefix={fd}) ")
T = 50
ALGS1 = MAB_algs
ALGS2 = CB_algs
ALGS3 = RL_algs
ALGSALL = agent_algs
MAB_range = np.arange(5)
CB_range = np.arange(5, 9)
RL_range = np.arange(9, 13)
tab_r = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_r_std = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_p = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_p_std = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_rs_sum = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_rs_std = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_rs_dff = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_rd_std = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_coop_ratio = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL))) # cooperation ratio metric
tab_conv_round = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL)), dtype=int) # convergence round metric

alg_triplets = [[alg1, alg2, alg3]
                for alg1 in ALGS1 for alg2 in ALGS2 for alg3 in ALGS3]
results = run_ipd_all(IPD_SCENARIO, alg_triplets, nMemory, fd)

for idx, (alg1, alg2, alg3) in enumerate(alg_triplets):
    it = idx // (len(ALGS2) * len(ALGS3))
    jt = (idx // len(ALGS3)) % len(ALGS2)
    kt = idx % len(ALGS3)
    i = MAB_range[it]
    j = CB_range[jt]
    k = RL_range[kt]
    r, r_std, p, p_std, rs_sum, rs_std, rs_dff, rd_std, rep = results[idx]
    tab_r[i, j, k, :] = r[0]
    tab_r_std[i, j, k, :] = r_std[0]
    tab_p[i, j, k, :] = p[0]
    tab_p_std[i, j, k, :] = p_std[0]
    tab_r[j, k, i, :] = r[1]
    tab_r_std[j, k, i, :] = r_std[1]
    tab_p[j, k, i, :] = p[1]
    tab_p_std[j, k, i, :] = p_std[1]
    tab_r[k, i, j, :] = r[2]
    tab_r_std[k, i, j, :] = r_std[2]
    tab_p[k, i, j, :] = p[2]
    tab_p_std[k, i, j, :] = p_std[2]
    tab_rs_sum[i, j, k, :] = tab_rs_sum[j, k,
                                        i, :] = tab_rs_sum[k, i, j, :] = rs_sum
    tab_rs_std[i, j, k, :] = tab_rs_std[j, k,
                                        i, :] = tab_rs_std[k, i, j, :] = rs_std
    tab_rs_dff[i, j, k, :] = rs_dff[0]
    tab_rd_std[i, j, k, :] = rd_std[0]
    tab_rs_dff[j, k, i, :] = rs_dff[1]
    tab_rd_std[j, k, i, :] = rd_std[1]
    tab_rs_dff[k, i, j, :] = rs_dff[2]
    tab_rd_std[k, i, j, :] = rd_std[2]

    if 'coop_ratios' in rep and 'convergence_rounds' in rep:
        tab_coop_ratio[i, j, k] = rep['coop_ratios'][0]
        tab_coop_ratio[j, k, i] = rep['coop_ratios'][1]
        tab_coop_ratio[k, i, j] = rep['coop_ratios'][2]
        tab_conv_round[i, j, k] = rep['convergence_rounds'][0]
        tab_conv_round[j, k, i] = rep['convergence_rounds'][1]
        tab_conv_round[k, i, j] = rep['convergence_rounds'][2]

tab = {'r': tab_r, 'rstd': tab_r_std, 'p': tab_p, 'pstd': tab_p_std,
       's': tab_rs_sum, 'sstd': tab_rs_std, 'd': tab_rs_dff, 'dstd': tab_rd_std, 'coop': tab_coop_ratio, 'conv': tab_conv_round}
path = './models/ipd1_m5_3ag.csv'
os.makedirs(os.path.dirname(path), exist_ok=True)
save_results_to_csv(tab, ALGSALL, path)

# Mental MAB agents

fd = 'ipd1_m5_mMAB'
print(f"\n Starting tournament for {fd} ")
nMemory = 5
T = 50
ALGS = mMAB_algs
tab_r = np.zeros((len(ALGS), len(ALGS), T))
tab_r_std = np.zeros((len(ALGS), len(ALGS), T))
tab_p = np.zeros((len(ALGS), len(ALGS), T))
tab_p_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_sum = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_dff = np.zeros((len(ALGS), len(ALGS), T))
tab_rd_std = np.zeros((len(ALGS), len(ALGS), T))
tab_coop_ratio = np.zeros((len(ALGS), len(ALGS))) # cooperation ratio metric
tab_conv_round = np.zeros((len(ALGS), len(ALGS)), dtype=int) # convergence round metric

alg_pairs = [[alg1, alg2] for alg1 in ALGS for alg2 in ALGS]
results = run_ipd_all(IPD_SCENARIO, alg_pairs, nMemory, fd)

for idx, (alg1, alg2) in enumerate(alg_pairs):
    i = idx // len(ALGS)
    j = idx % len(ALGS)
    r, r_std, p, p_std, rs_sum, rs_std, rs_dff, rd_std, rep = results[idx]
    tab_r[i, j, :] = r[0]
    tab_r_std[i, j, :] = r_std[0]
    tab_p[i, j, :] = p[0]
    tab_p_std[i, j, :] = p_std[0]
    tab_r[j, i, :] = r[1]
    tab_r_std[j, i, :] = r_std[1]
    tab_p[j, i, :] = p[1]
    tab_p_std[j, i, :] = p_std[1]
    tab_rs_sum[i, j, :] = tab_rs_sum[j, i, :] = rs_sum
    tab_rs_std[i, j, :] = tab_rs_std[j, i, :] = rs_std
    tab_rs_dff[i, j, :] = rs_dff[0]
    tab_rd_std[i, j, :] = rd_std[0]
    tab_rs_dff[j, i, :] = rs_dff[1]
    tab_rd_std[j, i, :] = rd_std[1]

    if 'coop_ratios' in rep and 'convergence_rounds' in rep:
        tab_coop_ratio[i, j] = rep['coop_ratios'][0]
        tab_coop_ratio[j, i] = rep['coop_ratios'][1]
        tab_conv_round[i, j] = rep['convergence_rounds'][0]
        tab_conv_round[j, i] = rep['convergence_rounds'][1]

tab = {'r': tab_r, 'rstd': tab_r_std, 'p': tab_p, 'pstd': tab_p_std,
       's': tab_rs_sum, 'sstd': tab_rs_std, 'd': tab_rs_dff, 'dstd': tab_rd_std, 'coop': tab_coop_ratio, 'conv': tab_conv_round}
path = './models/ipd1_m5_mMAB.csv'
os.makedirs(os.path.dirname(path), exist_ok=True)
save_results_to_csv(tab, ALGS, path)

# Mental CB agents

fd = 'ipd1_m5_mCB'
print(f"\n Starting tournament for {fd} ")
nMemory = 5
T = 50
ALGS = mCB_algs
tab_r = np.zeros((len(ALGS), len(ALGS), T))
tab_r_std = np.zeros((len(ALGS), len(ALGS), T))
tab_p = np.zeros((len(ALGS), len(ALGS), T))
tab_p_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_sum = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_dff = np.zeros((len(ALGS), len(ALGS), T))
tab_rd_std = np.zeros((len(ALGS), len(ALGS), T))
tab_coop_ratio = np.zeros((len(ALGS), len(ALGS))) # cooperation ratio metric
tab_conv_round = np.zeros((len(ALGS), len(ALGS)), dtype=int) # convergence round metric

alg_pairs = [[alg1, alg2] for alg1 in ALGS for alg2 in ALGS]
results = run_ipd_all(IPD_SCENARIO, alg_pairs, nMemory, fd)

for idx, (alg1, alg2) in enumerate(alg_pairs):
    i = idx // len(ALGS)
    j = idx % len(ALGS)
    r, r_std, p, p_std, rs_sum, rs_std, rs_dff, rd_std, rep = results[idx]
    tab_r[i, j, :] = r[0]
    tab_r_std[i, j, :] = r_std[0]
    tab_p[i, j, :] = p[0]
    tab_p_std[i, j, :] = p_std[0]
    tab_r[j, i, :] = r[1]
    tab_r_std[j, i, :] = r_std[1]
    tab_p[j, i, :] = p[1]
    tab_p_std[j, i, :] = p_std[1]
    tab_rs_sum[i, j, :] = tab_rs_sum[j, i, :] = rs_sum
    tab_rs_std[i, j, :] = tab_rs_std[j, i, :] = rs_std
    tab_rs_dff[i, j, :] = rs_dff[0]
    tab_rd_std[i, j, :] = rd_std[0]
    tab_rs_dff[j, i, :] = rs_dff[1]
    tab_rd_std[j, i, :] = rd_std[1]

    if 'coop_ratios' in rep and 'convergence_rounds' in rep:
        tab_coop_ratio[i, j] = rep['coop_ratios'][0]
        tab_coop_ratio[j, i] = rep['coop_ratios'][1]
        tab_conv_round[i, j] = rep['convergence_rounds'][0]
        tab_conv_round[j, i] = rep['convergence_rounds'][1]

tab = {'r': tab_r, 'rstd': tab_r_std, 'p': tab_p, 'pstd': tab_p_std,
       's': tab_rs_sum, 'sstd': tab_rs_std, 'd': tab_rs_dff, 'dstd': tab_rd_std, 'coop': tab_coop_ratio, 'conv': tab_conv_round}
path = './models/ipd1_m5_mCB.csv'
os.makedirs(os.path.dirname(path), exist_ok=True)
save_results_to_csv(tab, ALGS, path)

# Mental RL agents

fd = 'ipd1_m5_mRL'
print(f"\n Starting tournament for {fd} ")
nMemory = 5
T = 50
ALGS = mRL_algs
tab_r = np.zeros((len(ALGS), len(ALGS), T))
tab_r_std = np.zeros((len(ALGS), len(ALGS), T))
tab_p = np.zeros((len(ALGS), len(ALGS), T))
tab_p_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_sum = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_dff = np.zeros((len(ALGS), len(ALGS), T))
tab_rd_std = np.zeros((len(ALGS), len(ALGS), T))
tab_coop_ratio = np.zeros((len(ALGS), len(ALGS))) # cooperation ratio metric
tab_conv_round = np.zeros((len(ALGS), len(ALGS)), dtype=int) # convergence round metric

alg_pairs = [[alg1, alg2] for alg1 in ALGS for alg2 in ALGS]
results = run_ipd_all(IPD_SCENARIO, alg_pairs, nMemory, fd)

for idx, (alg1, alg2) in enumerate(alg_pairs):
    i = idx // len(ALGS)
    j = idx % len(ALGS)
    r, r_std, p, p_std, rs_sum, rs_std, rs_dff, rd_std, rep = results[idx]
    tab_r[i, j, :] = r[0]
    tab_r_std[i, j, :] = r_std[0]
    tab_p[i, j, :] = p[0]
    tab_p_std[i, j, :] = p_std[0]
    tab_r[j, i, :] = r[1]
    tab_r_std[j, i, :] = r_std[1]
    tab_p[j, i, :] = p[1]
    tab_p_std[j, i, :] = p_std[1]
    tab_rs_sum[i, j, :] = tab_rs_sum[j, i, :] = rs_sum
    tab_rs_std[i, j, :] = tab_rs_std[j, i, :] = rs_std
    tab_rs_dff[i, j, :] = rs_dff[0]
    tab_rd_std[i, j, :] = rd_std[0]
    tab_rs_dff[j, i, :] = rs_dff[1]
    tab_rd_std[j, i, :] = rd_std[1]

    if 'coop_ratios' in rep and 'convergence_rounds' in rep:
        tab_coop_ratio[i, j] = rep['coop_ratios'][0]
        tab_coop_ratio[j, i] = rep['coop_ratios'][1]
        tab_conv_round[i, j] = rep['convergence_rounds'][0]
        tab_conv_round[j, i] = rep['convergence_rounds'][1]

tab = {'r': tab_r, 'rstd': tab_r_std, 'p': tab_p, 'pstd': tab_p_std,
       's': tab_rs_sum, 'sstd': tab_rs_std, 'd': tab_rs_dff, 'dstd': tab_rd_std, 'coop': tab_coop_ratio, 'conv': tab_conv_round}
path = './models/ipd1_m5_mRL.csv'
os.makedirs(os.path.dirname(path), exist_ok=True)
save_results_to_csv(tab, ALGS, path)

# Behavioral Cloning
print(f"\n Starting Behavioral Cloning phase ({fd}) ")

data = pd.read_csv('./data/all_data.csv')
trajs = np.array(data[data['period'] == 10].iloc[:, 9:27])  # (8258, 18)
np.random.shuffle(trajs)
trajs = trajs.reshape((trajs.shape[0], 2, 9))  # (8258, 2, 9)
trajs[trajs == 0] = -1
split = 8000
train_set = trajs[:split, :, :]
test_set = trajs[split:, :, :]

# Save flattened train/test data to CSV
train_df = pd.DataFrame(train_set.reshape(train_set.shape[0], -1))
train_df['split'] = 'train'
test_df = pd.DataFrame(test_set.reshape(test_set.shape[0], -1))
test_df['split'] = 'test'
pd.concat([train_df, test_df]).to_csv('./data/processed_train_test.csv', index=False)
print('Saved processed data to ./data/processed_train_test.csv')


ALGS = agent_algs
fd = 'bclone_m5'
T = 9
nTrials = 10
nMemory = 5
test_size = test_set.shape[0]
tab_r = np.zeros((test_size, len(ALGS), T))
tab_r_std = np.zeros((test_size, len(ALGS), T))
tab_p = np.zeros((test_size, len(ALGS), T))
tab_p_std = np.zeros((test_size, len(ALGS), T))
tab_rs_sum = np.zeros((test_size, len(ALGS), T))
tab_rs_std = np.zeros((test_size, len(ALGS), T))
tab_rs_dff = np.zeros((test_size, len(ALGS), T))
tab_rd_std = np.zeros((test_size, len(ALGS), T))
tab_pr = np.zeros((test_size, len(ALGS)))
tab_coop_ratio = np.zeros((test_size, len(ALGS))) # cooperation ratio metric
tab_conv_round = np.zeros((test_size, len(ALGS)), dtype=int) # convergence

results_bclone = Parallel(n_jobs=-1)(delayed(run_bclone_single_alg)(alg, train_set,
                                                                    test_set, IPD_SCENARIO, fd, nTrials, T, nMemory) for alg in tqdm(ALGS))

for i, alg_results in enumerate(results_bclone):
    tab_r[:, i, :] = alg_results['r']
    tab_r_std[:, i, :] = alg_results['rstd']
    tab_p[:, i, :] = alg_results['p']
    tab_p_std[:, i, :] = alg_results['pstd']
    tab_rs_sum[:, i, :] = alg_results['s']
    tab_rs_std[:, i, :] = alg_results['sstd']
    tab_rs_dff[:, i, :] = alg_results['d']
    tab_rd_std[:, i, :] = alg_results['dstd']
    tab_pr[:, i] = alg_results['pr']
    tab_coop_ratio[:, i] = alg_results['coop_ratios']
    tab_conv_round[:, i] = alg_results['convergence_rounds']

tab = {'r': tab_r, 'rstd': tab_r_std, 'p': tab_p, 'pstd': tab_p_std, 's': tab_rs_sum,
       'sstd': tab_rs_std, 'd': tab_rs_dff, 'dstd': tab_rd_std, 'pr': tab_pr, 'coop': tab_coop_ratio, 'conv': tab_conv_round}
path = './models/bclone_m5.csv'
os.makedirs(os.path.dirname(path), exist_ok=True)
save_results_to_csv(tab, ALGS, path)


# Case with 2 agents

fd = 'ipd1_m1'
print(f"\n Starting 2-agent tournament (nMemory={nMemory}, prefix={fd}) ")
nMemory = 1
T = 50
ALGS = all_algs
tab_r = np.zeros((len(ALGS), len(ALGS), T))
tab_r_std = np.zeros((len(ALGS), len(ALGS), T))
tab_p = np.zeros((len(ALGS), len(ALGS), T))
tab_p_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_sum = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_dff = np.zeros((len(ALGS), len(ALGS), T))
tab_rd_std = np.zeros((len(ALGS), len(ALGS), T))
tab_coop_ratio = np.zeros((len(ALGS), len(ALGS))) # cooperation ratio metric
tab_conv_round = np.zeros((len(ALGS), len(ALGS)), dtype=int) # convergence round metric

alg_pairs = [[alg1, alg2] for alg1 in ALGS for alg2 in ALGS]
results = run_ipd_all(IPD_SCENARIO, alg_pairs, nMemory, fd)

for idx, (alg1, alg2) in enumerate(alg_pairs):
    i = idx // len(ALGS)
    j = idx % len(ALGS)
    r, r_std, p, p_std, rs_sum, rs_std, rs_dff, rd_std, rep = results[idx]
    tab_r[i, j, :] = r[0]
    tab_r_std[i, j, :] = r_std[0]
    tab_p[i, j, :] = p[0]
    tab_p_std[i, j, :] = p_std[0]
    tab_r[j, i, :] = r[1]
    tab_r_std[j, i, :] = r_std[1]
    tab_p[j, i, :] = p[1]
    tab_p_std[j, i, :] = p_std[1]
    tab_rs_sum[i, j, :] = tab_rs_sum[j, i, :] = rs_sum
    tab_rs_std[i, j, :] = tab_rs_std[j, i, :] = rs_std
    tab_rs_dff[i, j, :] = rs_dff[0]
    tab_rd_std[i, j, :] = rd_std[0]
    tab_rs_dff[j, i, :] = rs_dff[1]
    tab_rd_std[j, i, :] = rd_std[1]

    if 'coop_ratios' in rep and 'convergence_rounds' in rep:
        tab_coop_ratio[i, j] = rep['coop_ratios'][0]
        tab_coop_ratio[j, i] = rep['coop_ratios'][1]
        tab_conv_round[i, j] = rep['convergence_rounds'][0]
        tab_conv_round[j, i] = rep['convergence_rounds'][1]

tab = {'r': tab_r, 'rstd': tab_r_std, 'p': tab_p, 'pstd': tab_p_std,
       's': tab_rs_sum, 'sstd': tab_rs_std, 'd': tab_rs_dff, 'dstd': tab_rd_std, 'coop': tab_coop_ratio, 'conv': tab_conv_round}
path = './models/ipd1_m1.csv'
os.makedirs(os.path.dirname(path), exist_ok=True)
save_results_to_csv(tab, ALGS, path)

# Case with 3 agents

fd = 'ipd1_m1_3ag'
print(f"\n Starting 3-agent tournament (nMemory={nMemory}, prefix={fd}) ")
nMemory = 1
T = 50
ALGS1 = MAB_algs
ALGS2 = CB_algs
ALGS3 = RL_algs
ALGSALL = agent_algs
MAB_range = np.arange(5)
CB_range = np.arange(5, 9)
RL_range = np.arange(9, 13)
tab_r = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_r_std = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_p = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_p_std = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_rs_sum = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_rs_std = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_rs_dff = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_rd_std = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL), T))
tab_coop_ratio = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL))) # cooperation ratio metric
tab_conv_round = np.zeros((len(ALGSALL), len(ALGSALL), len(ALGSALL)), dtype=int) # convergence round metric

alg_triplets = [[alg1, alg2, alg3]
                for alg1 in ALGS1 for alg2 in ALGS2 for alg3 in ALGS3]
results = run_ipd_all(IPD_SCENARIO, alg_triplets, nMemory, fd)

for idx, (alg1, alg2, alg3) in enumerate(alg_triplets):
    it = idx // (len(ALGS2) * len(ALGS3))
    jt = (idx // len(ALGS3)) % len(ALGS2)
    kt = idx % len(ALGS3)
    i = MAB_range[it]
    j = CB_range[jt]
    k = RL_range[kt]
    r, r_std, p, p_std, rs_sum, rs_std, rs_dff, rd_std, rep = results[idx]
    tab_r[i, j, k, :] = r[0]
    tab_r_std[i, j, k, :] = r_std[0]
    tab_p[i, j, k, :] = p[0]
    tab_p_std[i, j, k, :] = p_std[0]
    tab_r[j, k, i, :] = r[1]
    tab_r_std[j, k, i, :] = r_std[1]
    tab_p[j, k, i, :] = p[1]
    tab_p_std[j, k, i, :] = p_std[1]
    tab_r[k, i, j, :] = r[2]
    tab_r_std[k, i, j, :] = r_std[2]
    tab_p[k, i, j, :] = p[2]
    tab_p_std[k, i, j, :] = p_std[2]
    tab_rs_sum[i, j, k, :] = tab_rs_sum[j, k,
                                        i, :] = tab_rs_sum[k, i, j, :] = rs_sum
    tab_rs_std[i, j, k, :] = tab_rs_std[j, k,
                                        i, :] = tab_rs_std[k, i, j, :] = rs_std
    tab_rs_dff[i, j, k, :] = rs_dff[0]
    tab_rd_std[i, j, k, :] = rd_std[0]
    tab_rs_dff[j, k, i, :] = rs_dff[1]
    tab_rd_std[j, k, i, :] = rd_std[1]
    tab_rs_dff[k, i, j, :] = rs_dff[2]
    tab_rd_std[k, i, j, :] = rd_std[2]

    if 'coop_ratios' in rep and 'convergence_rounds' in rep:
        tab_coop_ratio[i, j, k] = rep['coop_ratios'][0]
        tab_coop_ratio[j, k, i] = rep['coop_ratios'][1]
        tab_coop_ratio[k, i, j] = rep['coop_ratios'][2]
        tab_conv_round[i, j, k] = rep['convergence_rounds'][0]
        tab_conv_round[j, k, i] = rep['convergence_rounds'][1]
        tab_conv_round[k, i, j] = rep['convergence_rounds'][2]

tab = {'r': tab_r, 'rstd': tab_r_std, 'p': tab_p, 'pstd': tab_p_std,
       's': tab_rs_sum, 'sstd': tab_rs_std, 'd': tab_rs_dff, 'dstd': tab_rd_std, 'coop': tab_coop_ratio, 'conv': tab_conv_round}
path = './models/ipd1_m1_3ag.csv'
os.makedirs(os.path.dirname(path), exist_ok=True)
save_results_to_csv(tab, ALGSALL, path)

# Mental MAB agents

fd = 'ipd1_m1_mMAB'
print(f"\n Starting tournament for {fd} ")
nMemory = 1
T = 50
ALGS = mMAB_algs
tab_r = np.zeros((len(ALGS), len(ALGS), T))
tab_r_std = np.zeros((len(ALGS), len(ALGS), T))
tab_p = np.zeros((len(ALGS), len(ALGS), T))
tab_p_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_sum = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_dff = np.zeros((len(ALGS), len(ALGS), T))
tab_rd_std = np.zeros((len(ALGS), len(ALGS), T))
tab_coop_ratio = np.zeros((len(ALGS), len(ALGS))) # cooperation ratio metric
tab_conv_round = np.zeros((len(ALGS), len(ALGS)), dtype=int) # convergence round metric

alg_pairs = [[alg1, alg2] for alg1 in ALGS for alg2 in ALGS]
results = run_ipd_all(IPD_SCENARIO, alg_pairs, nMemory, fd)

for idx, (alg1, alg2) in enumerate(alg_pairs):
    i = idx // len(ALGS)
    j = idx % len(ALGS)
    r, r_std, p, p_std, rs_sum, rs_std, rs_dff, rd_std, rep = results[idx]
    tab_r[i, j, :] = r[0]
    tab_r_std[i, j, :] = r_std[0]
    tab_p[i, j, :] = p[0]
    tab_p_std[i, j, :] = p_std[0]
    tab_r[j, i, :] = r[1]
    tab_r_std[j, i, :] = r_std[1]
    tab_p[j, i, :] = p[1]
    tab_p_std[j, i, :] = p_std[1]
    tab_rs_sum[i, j, :] = tab_rs_sum[j, i, :] = rs_sum
    tab_rs_std[i, j, :] = tab_rs_std[j, i, :] = rs_std
    tab_rs_dff[i, j, :] = rs_dff[0]
    tab_rd_std[i, j, :] = rd_std[0]
    tab_rs_dff[j, i, :] = rs_dff[1]
    tab_rd_std[j, i, :] = rd_std[1]

    if 'coop_ratios' in rep and 'convergence_rounds' in rep:
        tab_coop_ratio[i, j] = rep['coop_ratios'][0]
        tab_coop_ratio[j, i] = rep['coop_ratios'][1]
        tab_conv_round[i, j] = rep['convergence_rounds'][0]
        tab_conv_round[j, i] = rep['convergence_rounds'][1]

tab = {'r': tab_r, 'rstd': tab_r_std, 'p': tab_p, 'pstd': tab_p_std,
       's': tab_rs_sum, 'sstd': tab_rs_std, 'd': tab_rs_dff, 'dstd': tab_rd_std, 'coop': tab_coop_ratio, 'conv': tab_conv_round}
path = './models/ipd1_m1_mMAB.csv'
os.makedirs(os.path.dirname(path), exist_ok=True)
save_results_to_csv(tab, ALGS, path)

# Mental CB agents

fd = 'ipd1_m1_mCB'
print(f"\n Starting tournament for {fd} ")
nMemory = 1
T = 50
ALGS = mCB_algs
tab_r = np.zeros((len(ALGS), len(ALGS), T))
tab_r_std = np.zeros((len(ALGS), len(ALGS), T))
tab_p = np.zeros((len(ALGS), len(ALGS), T))
tab_p_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_sum = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_dff = np.zeros((len(ALGS), len(ALGS), T))
tab_rd_std = np.zeros((len(ALGS), len(ALGS), T))
tab_coop_ratio = np.zeros((len(ALGS), len(ALGS))) # cooperation ratio metric
tab_conv_round = np.zeros((len(ALGS), len(ALGS)), dtype=int) # convergence round metric

alg_pairs = [[alg1, alg2] for alg1 in ALGS for alg2 in ALGS]
results = run_ipd_all(IPD_SCENARIO, alg_pairs, nMemory, fd)

for idx, (alg1, alg2) in enumerate(alg_pairs):
    i = idx // len(ALGS)
    j = idx % len(ALGS)
    r, r_std, p, p_std, rs_sum, rs_std, rs_dff, rd_std, rep = results[idx]
    tab_r[i, j, :] = r[0]
    tab_r_std[i, j, :] = r_std[0]
    tab_p[i, j, :] = p[0]
    tab_p_std[i, j, :] = p_std[0]
    tab_r[j, i, :] = r[1]
    tab_r_std[j, i, :] = r_std[1]
    tab_p[j, i, :] = p[1]
    tab_p_std[j, i, :] = p_std[1]
    tab_rs_sum[i, j, :] = tab_rs_sum[j, i, :] = rs_sum
    tab_rs_std[i, j, :] = tab_rs_std[j, i, :] = rs_std
    tab_rs_dff[i, j, :] = rs_dff[0]
    tab_rd_std[i, j, :] = rd_std[0]
    tab_rs_dff[j, i, :] = rs_dff[1]
    tab_rd_std[j, i, :] = rd_std[1]

    if 'coop_ratios' in rep and 'convergence_rounds' in rep:
        tab_coop_ratio[i, j] = rep['coop_ratios'][0]
        tab_coop_ratio[j, i] = rep['coop_ratios'][1]
        tab_conv_round[i, j] = rep['convergence_rounds'][0]
        tab_conv_round[j, i] = rep['convergence_rounds'][1]

tab = {'r': tab_r, 'rstd': tab_r_std, 'p': tab_p, 'pstd': tab_p_std,
       's': tab_rs_sum, 'sstd': tab_rs_std, 'd': tab_rs_dff, 'dstd': tab_rd_std, 'coop': tab_coop_ratio, 'conv': tab_conv_round}
path = './models/ipd1_m1_mCB.csv'
os.makedirs(os.path.dirname(path), exist_ok=True)
save_results_to_csv(tab, ALGS, path)

# Mental RL agents

fd = 'ipd1_m1_mRL'
print(f"\n Starting tournament for {fd} ")
nMemory = 1
T = 50
ALGS = mRL_algs
tab_r = np.zeros((len(ALGS), len(ALGS), T))
tab_r_std = np.zeros((len(ALGS), len(ALGS), T))
tab_p = np.zeros((len(ALGS), len(ALGS), T))
tab_p_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_sum = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_std = np.zeros((len(ALGS), len(ALGS), T))
tab_rs_dff = np.zeros((len(ALGS), len(ALGS), T))
tab_rd_std = np.zeros((len(ALGS), len(ALGS), T))
tab_coop_ratio = np.zeros((len(ALGS), len(ALGS))) # cooperation ratio metric
tab_conv_round = np.zeros((len(ALGS), len(ALGS)), dtype=int) # convergence round metric

alg_pairs = [[alg1, alg2] for alg1 in ALGS for alg2 in ALGS]
results = run_ipd_all(IPD_SCENARIO, alg_pairs, nMemory, fd)

for idx, (alg1, alg2) in enumerate(alg_pairs):
    i = idx // len(ALGS)
    j = idx % len(ALGS)
    r, r_std, p, p_std, rs_sum, rs_std, rs_dff, rd_std, rep = results[idx]
    tab_r[i, j, :] = r[0]
    tab_r_std[i, j, :] = r_std[0]
    tab_p[i, j, :] = p[0]
    tab_p_std[i, j, :] = p_std[0]
    tab_r[j, i, :] = r[1]
    tab_r_std[j, i, :] = r_std[1]
    tab_p[j, i, :] = p[1]
    tab_p_std[j, i, :] = p_std[1]
    tab_rs_sum[i, j, :] = tab_rs_sum[j, i, :] = rs_sum
    tab_rs_std[i, j, :] = tab_rs_std[j, i, :] = rs_std
    tab_rs_dff[i, j, :] = rs_dff[0]
    tab_rd_std[i, j, :] = rd_std[0]
    tab_rs_dff[j, i, :] = rs_dff[1]
    tab_rd_std[j, i, :] = rd_std[1]

    if 'coop_ratios' in rep and 'convergence_rounds' in rep:
        tab_coop_ratio[i, j] = rep['coop_ratios'][0]
        tab_coop_ratio[j, i] = rep['coop_ratios'][1]
        tab_conv_round[i, j] = rep['convergence_rounds'][0]
        tab_conv_round[j, i] = rep['convergence_rounds'][1]

tab = {'r': tab_r, 'rstd': tab_r_std, 'p': tab_p, 'pstd': tab_p_std,
       's': tab_rs_sum, 'sstd': tab_rs_std, 'd': tab_rs_dff, 'dstd': tab_rd_std, 'coop': tab_coop_ratio, 'conv': tab_conv_round}
path = './models/ipd1_m1_mRL.csv'
os.makedirs(os.path.dirname(path), exist_ok=True)
save_results_to_csv(tab, ALGS, path)

# Behavioral Cloning
print(f"\n Starting Behavioral Cloning phase ({fd}) ")

data = pd.read_csv('./data/all_data.csv')
trajs = np.array(data[data['period'] == 10].iloc[:, 9:27])  # (8258, 18)
np.random.shuffle(trajs)
trajs = trajs.reshape((trajs.shape[0], 2, 9))  # (8258, 2, 9)
trajs[trajs == 0] = -1
split = 8000
train_set = trajs[:split, :, :]
test_set = trajs[split:, :, :]

# Save flattened train/test data to CSV
train_df = pd.DataFrame(train_set.reshape(train_set.shape[0], -1))
train_df['split'] = 'train'
test_df = pd.DataFrame(test_set.reshape(test_set.shape[0], -1))
test_df['split'] = 'test'
pd.concat([train_df, test_df]).to_csv('./data/processed_train_test.csv', index=False)
print('Saved processed data to ./data/processed_train_test.csv')


ALGS = agent_algs
fd = 'bclone_m1'
T = 9
nTrials = 10
nMemory = 1
test_size = test_set.shape[0]
tab_r = np.zeros((test_size, len(ALGS), T))
tab_r_std = np.zeros((test_size, len(ALGS), T))
tab_p = np.zeros((test_size, len(ALGS), T))
tab_p_std = np.zeros((test_size, len(ALGS), T))
tab_rs_sum = np.zeros((test_size, len(ALGS), T))
tab_rs_std = np.zeros((test_size, len(ALGS), T))
tab_rs_dff = np.zeros((test_size, len(ALGS), T))
tab_rd_std = np.zeros((test_size, len(ALGS), T))
tab_pr = np.zeros((test_size, len(ALGS)))
tab_coop_ratio = np.zeros((test_size, len(ALGS))) # cooperation ratio metric
tab_conv_round = np.zeros((test_size, len(ALGS)), dtype=int) # convergence

results_bclone = Parallel(n_jobs=-1)(delayed(run_bclone_single_alg)(alg, train_set,
                                                                    test_set, IPD_SCENARIO, fd, nTrials, T, nMemory) for alg in tqdm(ALGS))

for i, alg_results in enumerate(results_bclone):
    tab_r[:, i, :] = alg_results['r']
    tab_r_std[:, i, :] = alg_results['rstd']
    tab_p[:, i, :] = alg_results['p']
    tab_p_std[:, i, :] = alg_results['pstd']
    tab_rs_sum[:, i, :] = alg_results['s']
    tab_rs_std[:, i, :] = alg_results['sstd']
    tab_rs_dff[:, i, :] = alg_results['d']
    tab_rd_std[:, i, :] = alg_results['dstd']
    tab_pr[:, i] = alg_results['pr']
    tab_coop_ratio[:, i] = alg_results['coop_ratios']
    tab_conv_round[:, i] = alg_results['convergence_rounds']

tab = {'r': tab_r, 'rstd': tab_r_std, 'p': tab_p, 'pstd': tab_p_std, 's': tab_rs_sum,
       'sstd': tab_rs_std, 'd': tab_rs_dff, 'dstd': tab_rd_std, 'pr': tab_pr, 'coop': tab_coop_ratio, 'conv': tab_conv_round}
path = './models/bclone_m1.csv'
os.makedirs(os.path.dirname(path), exist_ok=True)
save_results_to_csv(tab, ALGS, path)
