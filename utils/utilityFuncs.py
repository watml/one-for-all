import os

import numpy as np
import torch
from utils.load_datasets import *
from utils import models
from scipy import special

class gameTraining:
    def __init__(self, *, X_valued, y_valued, X_perf, y_perf, num_class, metric, arch, lr, game_seed=2024):
        self.X_train, self.y_train = X_valued, y_valued
        self.X_perf, self.y_perf = X_perf, y_perf
        self.arch = arch
        self.metric = metric
        self.game_seed = game_seed
        self.num_class = num_class
        self.num_player = len(y_valued)
        self.half_num_class = num_class // 2

        # load model and optimizer
        if arch == "logistic":
            self.model = models.LogisticRegression(self.X_perf.shape[1], num_class)
        elif arch == "LeNet":
            self.model = models.LeNet()
        else:
            raise NotImplementedError(f"Check {arch}")
        self.model.double() # float64 is used for more consistent reproducibility across platforms

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def evaluate(self, subset):
        assert isinstance(subset[0], np.bool_)
        subset = subset[:self.num_player] # there may be a null player at the end.
        self.train_model(self.X_train[subset], self.y_train[subset])
        return self.output_score()

    def dist_evaluate(self, subset, z_extra):
        assert isinstance(subset[0], np.bool_)
        X_extra, y_extra = z_extra
        X = torch.cat((self.X_train[subset], X_extra))
        y = torch.cat((self.y_train[subset], y_extra))
        self.train_model(X, y)
        return self.output_score()

    def train_model(self, X, y):
        # to avoid that for some fixed cardinality the data point (self.X_train[-1], self.y_train[-1]) is always the
        # last one fed into the model in one-epoch-one-mini-batch learning.
        with set_numpy_seed((X > 0).sum().item() + (y < self.half_num_class).sum().item() + self.game_seed):
            pi = np.random.permutation(len(y))
        X, y = X[pi], y[pi]

        with set_torch_seed(self.game_seed):
            for layer in self.model.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        if len(y):
            for datum, label in zip(X, y):
                logit = self.model(datum)
                loss = self.criterion(logit, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def output_score(self):
        self.model.eval()
        with torch.no_grad():
            logit = self.model(self.X_perf)
        assert ~torch.isnan(logit.sum())
        self.model.train()
        if self.metric == "accuracy":
            predict = np.argmax(logit.numpy(), 1)
            label = self.y_perf.numpy()
            score = np.sum(predict == label) / len(label)
        elif self.metric == "cross_entropy":
            score = -self.criterion(logit, self.y_perf).numpy()
        else:
            raise NotImplementedError(f"Check {self.metric}")
        return score


class gameKNN:
    def __init__(self, *, X_valued, y_valued, X_perf, y_perf, K=20):
        self.X_valued, self.y_valued = X_valued, y_valued
        self.num_player = len(self.y_valued)
        self.X_perf, self.y_perf = X_perf, y_perf
        self.num_perf = len(self.y_perf)
        self.K = K
        self._alpha = None

    @property
    def alpha(self):
        if self._alpha is None:
            # dist = np.linalg.norm(self.X_perf[:, None, :] - self.X_valued[None, :, :], axis=2)
            # self._alpha = dist.argsort(axis=1).argsort(axis=1)
            self._alpha = np.empty((self.num_perf, self.num_player), dtype=np.int64)
            for i, xp in enumerate(self.X_perf):
                dist = np.linalg.norm(self.X_valued - xp[None, :], axis=1)
                self._alpha[i] = dist.argsort().argsort()
        return self._alpha

    def evaluate(self, subset):
        assert isinstance(subset[0], np.bool_)
        if subset.sum():
            y_sub = self.y_valued[subset]
            alpha = self.alpha[:, subset]
            acc = 0.
            for i in range(self.num_perf):
                alpha_sub = alpha[i].argsort()
                yt = self.y_perf[i]
                acc += (y_sub[alpha_sub][:self.K] == yt).sum() / self.K
            return acc / self.num_perf
        else:
            return 0

    def get_Shapley(self):
        tmp = np.arange(1, self.num_player + 1)
        coeff = np.divide(np.minimum(tmp, self.K), tmp)
        value_exact = np.zeros(self.num_player, dtype=np.float64)
        value_cur = np.zeros(self.num_player, dtype=np.float64)
        for i in range(self.num_perf):
            yt = self.y_perf[i]
            alpha = self.alpha[i]
            y_cur = self.y_valued[alpha.argsort()]
            value_cur[-1] = (y_cur[-1] == yt) / self.K * coeff[-1]
            for j in range(self.num_player - 2, -1, -1):
                value_cur[j] = value_cur[j + 1] + (int(y_cur[j] == yt) - int(y_cur[j + 1] == yt)) / self.K * coeff[j]
            value_exact += value_cur[alpha]
        return value_exact / self.num_perf


class gameSOU:
    def __init__(self, *, num_player, num_unanimity, path="exp/SOUGame", seed=2024):
        os.makedirs(path, exist_ok=True)
        data_saved = os.path.join(path, f"n_player={num_player};n_unanimity={num_unanimity};seed={seed}.npz")
        self.num_player = num_player
        if os.path.exists(data_saved):
            data = np.load(data_saved)
            self.coalitions = data["coalitions"]
            self.weights = data["weights"]
        else:
            s_range = np.arange(1, num_player)
            pos_range = np.arange(num_player)
            self.coalitions = np.zeros((num_unanimity, num_player), dtype=bool)
            for i in range(num_unanimity):
                s = np.random.choice(s_range)
                pos = np.random.choice(pos_range, size=s, replace=False)
                self.coalitions[i, pos] = True
            self.weights = np.random.uniform(low=-num_player, high=num_player, size=num_unanimity)
            np.savez_compressed(data_saved, coalitions=self.coalitions, weights=self.weights)
        self.sizes = self.coalitions.sum(axis=1)

    def evaluate(self, subset):
        assert isinstance(subset[0], np.bool_)
        subset = subset[:self.num_player]
        tmp = np.logical_and(self.coalitions, subset[None, :]).sum(axis=1) == self.sizes
        return (tmp * self.weights).sum()
        # index = subset.sum() >= self.sizes
        # tmp = np.logical_and(self.coalitions[index], subset[None, :]).sum(axis=1) == self.sizes[index]
        # return (tmp * self.weights[index]).sum()

    def get_semivalue(self, *, semivalue, semivalue_param=None):
        if semivalue == "shapley":
            tmp = 1 / self.sizes
        elif semivalue == "weighted_banzhaf":
            tmp = np.power(semivalue_param, self.sizes - 1, dtype=np.float128)
        elif semivalue == "beta_shapley":
            tmp = special.beta(semivalue_param[1] + self.sizes - 1, semivalue_param[0]) \
                  / special.beta(semivalue_param[1], semivalue_param[0])
        return np.dot(self.weights * tmp, self.coalitions)














