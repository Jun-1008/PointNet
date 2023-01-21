import os
import csv
import torch
import numpy as np

# list_t = []
i = 0

csv_path = ".\\leap_data\\"
# 絶対パスでなくても良さそう
csvs = os.listdir(csv_path)

def csv_data_sampler():
    list_t = []
    for csv_file in csvs:
        csv_data = os.path.join(csv_path, csv_file)
        with open(csv_data, 'r') as f:
            csvreader = csv.reader(f)
            content = [row for row in csvreader]
            # for row in csvreader:
            #     content = row
            content_np = np.array(content, dtype=float)
            content_t = torch.tensor(content_np, dtype=torch.float)
            list_t.append(content_t)

    sample_t = torch.cat(list_t)
    sample_t_v = sample_t.view(10, 93, 3)
    # print(sample_t.size())
    # print(sample_t)

    x = torch.tensor([[0.], [0.], [0.], [0.], [0.], [1.], [1.], [1.], [1.], [1.]])
    # print(x)

    data_shuffle = torch.randperm(10)
    return sample_t_v[data_shuffle].view(-1, 3), x[data_shuffle].view(-1, 1)
    # return sample_t_v[data_shuffle]
    # return sample_t_v[data_shuffle].view(-1, 3)
    # return x[data_shuffle]

# test =csv_data_sampler() 
# print(test.size())
# print(test)

# data = csv_data_sampler()

# print(data.size())
# print(data)
# print(data.x.size())

