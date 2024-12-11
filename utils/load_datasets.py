import numpy as np
import torch
from sklearn.datasets import fetch_openml
from torchvision import datasets, transforms
from utils.funcs import *
from collections import defaultdict

dataset2id = {
    "iris": 61,
    "wind": 847,
}


def load_dataset(dataset, *, n_valued=1, n_perf=1, n_test=1, dataset_seed=2024, path="dataset"):
    if dataset in ["MNIST", "FMNIST"]:
        return load_MNIST(dataset, n_valued, n_perf, n_test, dataset_seed, path)
    else:
        return load_OpenML(dataset, n_valued, n_perf, dataset_seed, path)


def load_OpenML(dataset, n_valued, n_perf, dataset_seed, path, percent_train=0.8):
    global dataset2id
    data, target = fetch_openml(data_id=dataset2id[dataset], data_home=path, return_X_y=True, as_frame=False)
    target_unique = np.unique(target)
    num_class = len(target_unique)
    dict_transform = dict(zip(target_unique, range(num_class)))
    target = np.array([dict_transform[key] for key in target])

    num_total = len(target)
    num_train = int(np.round(num_total * percent_train))
    with set_numpy_seed(dataset_seed):
        pi = np.random.permutation(num_total)
    data_train, label_train = data[pi[:num_train]], target[pi[:num_train]]
    data_mean = np.mean(data_train, axis=0, keepdims=True)
    data_std = np.std(data_train, axis=0, keepdims=True)
    data_train = np.divide(data_train - data_mean,  data_std)

    label2pos = defaultdict()
    num_cut = np.inf
    for i in range(num_class):
        pos = np.where(label_train == i)[0]
        num = len(pos)
        label2pos[i] = pos
        if num < num_cut:
            num_cut = num
    assert n_valued + n_perf <= num_cut * num_class

    pos_all = np.zeros(num_class, dtype=np.int32)
    pos_valued = np.empty(n_valued, dtype=np.int32)
    pos_perf = np.empty(n_perf, dtype=np.int32)
    for pos_cur in range(n_valued + n_perf):
        pos_class = pos_cur % num_class
        if pos_cur < n_valued:
            pos_valued[pos_cur] = label2pos[pos_class][pos_all[pos_class]]
        else:
            pos_perf[pos_cur - n_valued] = label2pos[pos_class][pos_all[pos_class]]
        pos_all[pos_class] += 1
    with set_numpy_seed(dataset_seed):
        np.random.shuffle(pos_valued)
        np.random.shuffle(pos_perf)
    X_valued, y_valued = data_train[pos_valued], label_train[pos_valued]
    X_perf, y_perf = data_train[pos_perf], label_train[pos_perf]

    return (torch.tensor(X_valued, dtype=torch.float64), torch.tensor(y_valued, dtype=torch.int64)), \
           (torch.tensor(X_perf, dtype=torch.float64), torch.tensor(y_perf, dtype=torch.int64)), None, num_class


def load_MNIST(dataset, n_valued, n_perf, n_test, dataset_seed, path):
    assert n_valued + n_perf <= 60000
    assert n_test <= 10000
    if dataset == "FMNIST":
        download_func = datasets.FashionMNIST
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(0.2860, 0.3530)])
    elif dataset == "MNIST":
        download_func = datasets.MNIST
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(0.1307, 0.3081)])
    else:
        raise NotImplementedError(f"Check {dataset}")

    trainset = download_func(path, download=True, train=True, transform=transform)
    testset = download_func(path, download=True, train=False, transform=transform)
    with set_torch_seed(dataset_seed):
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=n_valued + n_perf, shuffle=True)
        for X, y in trainloader:
            break
        testloader = torch.utils.data.DataLoader(testset, batch_size=n_test, shuffle=True)
        for X_test, y_test in testloader:
            break
    X_valued, X_perf = X[:n_valued][:, None, :, :, :], X[n_valued:n_valued + n_perf]
    y_valued, y_perf = y[:n_valued][:, None], y[n_valued:n_valued + n_perf]
    X_test, y_test = X_test[:, None, :, :, :], y_test[:, None]

    X_valued, X_perf, X_test = X_valued.type(torch.float64), X_perf.type(torch.float64), X_test.type(torch.float64)
    return (X_valued, y_valued), (X_perf, y_perf), (X_test, y_test), 10