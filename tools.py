import numpy as np
import scipy
from scipy import stats


def astimate_acc1(a=None, b=None, s_a=None, s_b=None, r_a=None, r_b=None, L=None, N_list=None):
    M = a.shape[0]
    N_list = np.array(N_list)
    N = np.sum(N_list)

    A_M = 0
    y_list = []

    for k in range(L):
        y = 1

        for j in range(L):

            if j == k:
                continue

            son_m = 0
            for m in range(M):
                son_m += a[m] - b[m]

            mum_pq = 0

            for p in range(M):
                for q in range(M):
                    tmp_mum = s_a[p] * s_a[q] * r_a[p][q] + s_b[p] * s_b[q] * r_b[p][q]

                    mum_pq += tmp_mum

            mum_pq = np.sqrt(mum_pq)
            y *= scipy.stats.norm.cdf(son_m / mum_pq)

            y_list.append(scipy.stats.norm.cdf(son_m / mum_pq))

        N_k = N_list[k]
        A_M += N_k / N * y

    return A_M


def astimate_acc2(a=None, b=None, s=None, r_a=None, r_b=None, L=None, N_list=None):
    M = a.shape[0]
    N_list = np.array(N_list)
    N = np.sum(N_list)

    A_M = 0
    y_list = []

    for k in range(L):
        y = 1

        Delta_div_sigma = np.mean(a - b) / np.mean(s)

        for j in range(L):

            if j == k:
                continue

            son_m = M

            mum_pq = 0

            for p in range(M):
                for q in range(M):
                    tmp_mum = r_a[p][q] + r_b[p][q]

                    mum_pq += tmp_mum

            mum_pq = np.sqrt(mum_pq)
            V = son_m / mum_pq
            y *= scipy.stats.norm.cdf(Delta_div_sigma * V)

            y_list.append(scipy.stats.norm.cdf(son_m / mum_pq))

        N_k = N_list[k]
        A_M += N_k / N * y

    return A_M


