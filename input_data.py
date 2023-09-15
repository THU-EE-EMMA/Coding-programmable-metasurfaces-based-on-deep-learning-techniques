import scipy.io as sio
import h5py

data = h5py.File('./data/data.mat')
Mask_U = data["Mask_U"]
Mask_L = data["Mask_L"]
Loc_dis = data["Loc_distri"]
PHI_re = data["Binary_label"]
N_test = data["N_test"]
batch=40

def input_data(test):
    if test:
        x = Mask_U[int(N_test[0][0]/ batch* 0.8)*batch:]
        x = x.reshape(-1,200,200,1)
        d = Mask_L[int(N_test[0][0]/batch * 0.8)*batch:]
        d = d.reshape(-1,200,200,1)
        k = Loc_dis[int(N_test[0][0]/batch * 0.8)*batch:]
        k = k.reshape(-1,200,200,1)
        y = PHI_re[int(N_test[0][0]/batch * 0.8)*batch:]
        y = y.reshape(-1,48,48,1)
    else:
        x = Mask_U[:int(N_test[0][0]/batch * 0.8)*batch]
        x = x.reshape(-1,200,200,1)
        d = Mask_L[:int(N_test[0][0]/batch * 0.8)*batch]
        d = d.reshape(-1,200,200,1)
        k = Loc_dis[:int(N_test[0][0]/batch * 0.8)*batch]
        k = k.reshape(-1,200,200,1)
        y = PHI_re[:int(N_test[0][0]/batch * 0.8)*batch]
        y = y.reshape(-1,48,48,1)
    return x,d,k,y
