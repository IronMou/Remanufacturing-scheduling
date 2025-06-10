import numpy as np
from torch.utils.data import Dataset
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import DataParallel
def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high,seed=None):
    if seed != None:
        np.random.seed(seed)

    time0 = np.random.randint(low=low, high=high, size=(n_j, n_m,n_m-1))
    time1=np.random.randint(low=1, high=high, size=(n_j, n_m,1))
    times=np.concatenate((time0,time1),-1)
    for i in range(n_j):
        times[i]= permute_rows(times[i])
    return times

#-99~99 randint -- 5
#-99~99 0~99 uniform -- 0
#-99~99 1~99 uniform -- 0


class FJSPDataset(Dataset):

    def __init__(self,n_j, n_m, low, high,num_samples=1000000,seed=None,  offset=0, distribution=None):
        super(FJSPDataset, self).__init__()

        self.data_set = []
        self.job_num = n_j
        self.mach_num = n_m
        if seed != None:
            np.random.seed(seed)
        time0 = np.random.uniform(low=low, high=high, size=(num_samples, n_j, n_m, n_m - 1))
        time1 = np.random.uniform(low=0, high=high, size=(num_samples, n_j, n_m, 1))
        times = np.concatenate((time0, time1), -1)
        for j in range(num_samples):
            for i in range(n_j):
                times[j][i] = permute_rows(times[j][i])
            # Sample points randomly in [0, 1] square
        self.data = np.array(times)
        self.size = len(self.data)
        if os.environ.get("FJSP_VERBOSE", "0") == "1":
            self.save()

    def getdata(self):
        return self.data

    def save(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(current_dir, 'data')
        os.makedirs(dataset_dir, exist_ok=True)
        for i, data in enumerate(self.data):
            file_name = f'sample{i}_j{self.job_num}_m{self.mach_num}.csv'
            file_path = os.path.join(dataset_dir, file_name)
            if os.path.exists(file_path):
                # 删除文件
                os.remove(file_path)
            for j, time_matrix in enumerate(data):
                # 追加模式
                with open(file_path, 'a') as f:
                    np.savetxt(f, time_matrix, delimiter=',', header=f'job{j}')


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

'''a = TSPDataset(3,3,-1,1,200,2000)
a = a.getdata()
print(a)
for t in range(2000):
    for i in range(3):
        for j in range(3):
            durmch = a[t][i][j][torch.where(a[t][i][j] > 0)]
            a[t][i][j] = torch.tensor([durmch.mean() if i < 0 else i for i in a[t][i][j]])

print(a)'''
def override(fn):
    """
    override decorator
    """
    return fn
