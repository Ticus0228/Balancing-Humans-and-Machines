import pandas as pd
import numpy as np
from tools import astimate_acc1, astimate_acc2

# select from human0~5
human_list = ['human0']

# select from alexnet, densenet161, googlenet, resnet152, vgg19
machine_list = ['alexnet']

# select from 80, 95, 110, 125
noise_level= 125

n_H = len(human_list)
n_M = len(machine_list)
M = n_H + n_M

df_machine_source = pd.read_csv('data/ImageNet_M.csv')
df_human_source = pd.read_csv('data/ImageNet_H.csv')

classAll = ['knife', 'keyboard', 'elephant', 'bicycle', 'airplane', 'clock', 'oven', 'chair', 'bear', 'boat', 'cat', 'bottle', 'truck', 'car', 'bird', 'dog']
L = len(classAll)
useful_col = ['image_name', 'noise_type', 'noise_level', 'category', 'model_name', 'model_pred', 'correct'] + classAll
df = pd.concat([df_machine_source[useful_col], df_human_source[useful_col]], ignore_index=True, sort=False)

comb = human_list + machine_list
df_comb_list = []
for model_i in comb:
    df_comb_list.append(df[(df['model_name'] == model_i) & (df['noise_level'] == noise_level)].reset_index(drop=True))

N_list = []
mu = np.ones(shape=[L, M, L]) * (-100)
s = np.ones(shape=[L, M, L]) * (-100)
r = np.ones(shape=[L, M, M, L, L]) * (-100)
df_model_1 = df_comb_list[0]

for k in range(L):
    class_k = classAll[k]
    N_k = len(df_model_1[df_model_1['category'] == class_k])
    N_list.append(N_k)

a_ignoreClass = np.ones(shape=[M]) * (-100)
b_ignoreClass = np.ones(shape=[M]) * (-100)
s_a_ignoreClass = np.ones(shape=[M]) * (-100)
s_b_ignoreClass = np.ones(shape=[M]) * (-100)
s_ignoreClassCorrect = np.ones(shape=[M]) * (-100)

r_a_ignoreClass = np.ones(shape=[M, M]) * (-100)
r_b_ignoreClass = np.ones(shape=[M, M]) * (-100)
r_ab_ignoreClass = np.ones(shape=[M, M]) * (-100)
r_ignoreClassCorrect = np.ones(shape=[M, M]) * (-100)

a_sample = []
b_sample = []

for m in range(M):
    df_model_m = df_comb_list[m]
    a_sample_m = []
    b_sample_m = []
    all_sample_m = []

    for k in range(L):
        class_k = classAll[k]

        for j in range(L):
            class_j = classAll[j]

            if j == k:
                a_tmp = df_model_m[df_model_m['category'] == class_k][class_j].values

                for i_flatten in a_tmp:
                    a_sample_m.append(i_flatten)
                    all_sample_m.append(i_flatten)

            elif j != k:
                b_tmp = df_model_m[df_model_m['category'] == class_k][class_j].values

                for i_flatten in b_tmp:
                    b_sample_m.append(i_flatten)
                    all_sample_m.append(i_flatten)

    a_sample.append(a_sample_m)
    b_sample.append(b_sample_m)

a_ignoreClass = np.mean(a_sample, axis=1)
b_ignoreClass = np.mean(b_sample, axis=1)

s_a_ignoreClass = np.std(a_sample, axis=1)
s_b_ignoreClass = np.std(b_sample, axis=1)
s_ignoreClassCorrect = (s_a_ignoreClass+s_b_ignoreClass)/2

for p in range(M):
    model_p_sample = a_sample[p] + b_sample[p]

    for q in range(M):
        r_a_ignoreClass[p][q] = np.corrcoef(a_sample[p], a_sample[q])[0][1]
        r_b_ignoreClass[p][q] = np.corrcoef(b_sample[p], b_sample[q])[0][1]
        model_q_sample = a_sample[q] + b_sample[q]
        r_ignoreClassCorrect[p][q] = np.corrcoef(model_p_sample, model_q_sample)[0][1]

ACC_1 = \
    astimate_acc1(a=a_ignoreClass, b=b_ignoreClass,
                  s_a=s_a_ignoreClass,
                  s_b=s_b_ignoreClass,
                  r_a=r_a_ignoreClass,
                  r_b=r_b_ignoreClass,
                  L=L, N_list=N_list)
ACC_2 = \
    astimate_acc2(a=a_ignoreClass, b=b_ignoreClass,
                  s=s_ignoreClassCorrect,
                  r_a=r_a_ignoreClass, r_b=r_b_ignoreClass,
                  L=L, N_list=N_list)

for m in range(M):
    if m == 0:
        df_comb = df_comb_list[m]
    else:
        df_comb[classAll] += df_comb_list[m][classAll]

for idx, row in df_comb.iterrows():
    maxCls = None
    maxClsValue = -1 * np.inf

    for class_j in classAll:

        if row[class_j] > maxClsValue:
            maxClsValue = row[class_j]
            maxCls = class_j

    row['model_pred'] = maxCls

    if maxCls == row['category']:
        row['correct'] = 1
    else:
        row['correct'] = 0

    df_comb.loc[idx, :] = row

ACC = df_comb['correct'].sum() / len(df_comb)

print('ACC=', ACC)
print('ACC_1=', ACC_1)
print('ACC_2=', ACC_2)
