import numpy as np
import itertools
from scipy import special
import sys
import multiprocessing as mp
from tqdm import tqdm
from utils.funcs import vd_tqdm
from scipy.linalg import lstsq

class runEstimator:
    def __init__(self, *, estimator, n_process, semivalue, semivalue_param, game_func, game_args, num_player, nue_avg,
                 nue_per_proc, nue_track_avg, estimator_seed=2024, file_prog=None, **kwargs_estimator):
        self.estimator = estimator
        self.n_process = n_process - 1 # one process is used for aggregating results
        self.file_prog = file_prog
        self.semivalue = semivalue
        self.semivalue_param = semivalue_param
        self.game_func = game_func
        self.game_args = game_args
        self.estimator_seed = estimator_seed
        self.num_player = num_player

        # the number of utility evaluations used to do estimation on average (divided by the number of players)
        self.nue_avg = nue_avg

        # the number of utility evaluations each process will run in one batch.
        self.nue_per_proc = nue_per_proc

        # record the estimates of all players after using nue_track_avg, 2*nue_track_avg, ..., utility evaluations on average.
        self.nue_track_avg = nue_track_avg

        self.kwargs_estimator = kwargs_estimator

    def run(self):
        estimator_args = dict(
            semivalue=self.semivalue,
            semivalue_param=self.semivalue_param,
            game_func=self.game_func,
            game_args=self.game_args,
            num_player=self.num_player,
            nue_avg=self.nue_avg,
            nue_per_proc=self.nue_per_proc,
            nue_track_avg=self.nue_track_avg,
            estimator_seed=self.estimator_seed
        )
        estimator = getattr(sys.modules[__name__], self.estimator)(**estimator_args, **self.kwargs_estimator)
        print(f"The number of utility evalutions each process runs in one batch is {estimator.nue_per_proc_run}")
        if self.n_process > 1:
            with mp.Pool(self.n_process) as pool:
                process = pool.imap(estimator.run, estimator.sampling())
                for chunk in vd_tqdm(process, total=-(-estimator.num_sample//estimator.batch_size),
                                  miniters=self.n_process, maxinterval=float('inf'), file_prog=self.file_prog):
                    estimator.aggregate(chunk)
        else:
            for samples in tqdm(estimator.sampling(), total=-(-estimator.num_sample//estimator.batch_size)):
                estimator.aggregate(estimator.run(samples))
        return estimator.finalize()


class estimatorTemplate:
    def __init__(self, *, semivalue, semivalue_param, game_func, game_args, num_player, nue_avg, nue_per_proc, nue_track_avg,
                 estimator_seed, **kwargs_estimator):
        self.kwargs_estimator = kwargs_estimator
        self.semivalue = semivalue
        self.semivalue_param = semivalue_param
        self.game_func = game_func
        self.game_args = game_args
        self.num_player = num_player
        self.nue_avg = nue_avg
        self.nue_per_proc = nue_per_proc
        self.nue_track_avg = nue_track_avg
        self.estimator_seed = estimator_seed

        num_traj = self.nue_avg // self.nue_track_avg
        self.values_traj = np.empty((num_traj, self.num_player), dtype=np.float64)
        self.pos_traj = 0
        self.buff = self.interval_track = self.batch_size = None
        self.pos_buffer = 0
        self.samples = None

        self.lock_switch = True
        self.switch_state = False

    @property
    def switch(self):
        return self.switch_state

    @switch.setter
    def switch(self, state):
        if not self.lock_switch:
            self.switch_state = state

    @property
    def buffer_size(self):
        return self.interval_track + self.batch_size - 1

    def run(self):
        pass

    def _init_indiv(self):
        pass

    def sampling(self):
        self._init_indiv()
        np.random.seed(self.estimator_seed)

        count = 0
        for _ in range(self.num_sample):
            if not self.switch:
                self.samples[count] = self._generator()
                self.switch = True
            else:
                self.samples[count] = 1 - self.samples[count - 1]
                self.switch = False
            count += 1
            if count == self.batch_size:
                yield self.samples.copy()
                count = 0
        if count:
            yield self.samples[:count]

    def _generator(self):
        pass

    def aggregate(self, results_collect):
        self.buffer[self.pos_buffer:self.pos_buffer + len(results_collect)] = results_collect
        self.pos_buffer += len(results_collect)
        num_collect = self.pos_buffer // self.interval_track
        if num_collect:
            for i in range(num_collect):
                self._process(self.buffer[i*self.interval_track:(i+1)*self.interval_track])
                self.values_traj[self.pos_traj] = self._estimate()
                self.pos_traj += 1
            num_left = self.pos_buffer - (i + 1) * self.interval_track
            self.buffer[:num_left] = self.buffer[(i + 1) * self.interval_track:self.pos_buffer]
            self.pos_buffer = num_left

    def finalize(self):
        if self.pos_buffer:
            self._process(self.buffer[:self.pos_buffer])
            values_final = self._estimate()
        else:
            values_final = self.values_traj[-1]
        return values_final, self.values_traj

    def _process(self, inputs):
        pass

    def _estimate(self):
        pass

    def distribution_cardinality(self):
        if self.semivalue == "shapley":
            weights = np.full(self.num_player, 1. / self.num_player, dtype=np.float64)
        elif self.semivalue == "weighted_banzhaf":
            weights = np.ones(self.num_player, dtype=np.float64)
            for k in range(self.num_player):
                for i in range(k):
                    weights[k] *= (self.num_player - 1 - i) / (i + 1) * self.semivalue_param * (1 - self.semivalue_param)
                weights[k] *= (1 - self.semivalue_param) ** (self.num_player - 1 - 2 * k)
        elif self.semivalue == "beta_shapley":
            alpha, beta = self.semivalue_param
            weights = np.ones(self.num_player, dtype=np.float64)
            tmp_range = np.arange(1, self.num_player, dtype=np.float64)
            weights *= np.divide(tmp_range, tmp_range + (alpha + beta - 1)).prod()
            for s in range(self.num_player):
                r_cur = weights[s]
                tmp_range = np.arange(1, s + 1, dtype=np.float64)
                r_cur *= np.divide(tmp_range + (beta - 1), tmp_range).prod()
                tmp_range = np.arange(1, self.num_player - s, dtype=np.float64)
                r_cur *= np.divide((alpha - 1) + tmp_range, tmp_range).prod()
                weights[s] = r_cur
        else:
            raise NotImplementedError(f"Check {self.semivalue}")
        return weights


class exact_value(estimatorTemplate):
    def __init__(self, **kwargs):
        super(exact_value, self).__init__(**kwargs)
        self.values = np.zeros(self.num_player, dtype=np.float64)
        self.num_sample = 2 ** (self.num_player - 1)
        self.batch_size = -(-self.nue_per_proc // (2 * self.num_player))
        self.nue_per_proc_run = self.batch_size * 2 * self.num_player

    def sampling(self):
        count = 0
        samples = np.empty((self.batch_size, self.num_player-1), dtype=bool)
        for subset in itertools.product([True, False], repeat=self.num_player-1):
            samples[count] = subset
            count += 1
            if count == self.batch_size:
                yield samples.copy()
                count = 0
        if count:
            yield samples[:count]

    def run(self, samples):
        weights = np.empty(self.num_player, dtype=np.float64)
        for i in range(self.num_player):
            if self.semivalue == "shapley":
                weights[i] = special.beta(self.num_player - i, i + 1)
            elif self.semivalue == "weighted_banzhaf":
                weights[i] = (self.semivalue_param ** i) * ((1 - self.semivalue_param) ** (self.num_player - 1 - i))
            elif self.semivalue == "beta_shapley":
                weights[i] = 1
                alpha, beta = self.semivalue_param
                for k in range(1, i+1):
                    weights[i] *= (beta+k-1) / (alpha+beta+k-1)
                for k in range(i+1, self.num_player):
                    weights[i] *= (alpha+k-i-1) / (alpha+beta+k-1)
            else:
                raise NotImplementedError(f"Check {self.semivalue}")

        game = self.game_func(**self.game_args)
        fragment = np.zeros(self.num_player)
        right_index = np.zeros(self.num_player, dtype=bool)
        left_index = np.ones_like(right_index)
        for sample in samples:
            weight = weights[sample.sum()]
            right_index[:self.num_player - 1] = sample
            left_index[:self.num_player - 1] = sample
            fragment[-1] += weight * (game.evaluate(left_index) - game.evaluate(right_index))
            for player in range(self.num_player - 1):
                right_index[-1], right_index[player] = right_index[player], right_index[-1]
                left_index[-1], left_index[player] = left_index[player], left_index[-1]
                fragment[player] += weight * (game.evaluate(left_index) - game.evaluate(right_index))
                right_index[-1], right_index[player] = right_index[player], right_index[-1]
                left_index[-1], left_index[player] = left_index[player], left_index[-1]
        return fragment

    def aggregate(self, fragment):
        self.values += fragment

    def finalize(self):
        return self.values, self.values[None, :]


class sampling_lift(estimatorTemplate):
    def __init__(self, **kwargs):
        super(sampling_lift, self).__init__(**kwargs)
        self.interval_track = self.nue_track_avg // 2
        self.num_sample = self.nue_avg // 2
        self.batch_size = -(-self.nue_per_proc // (2 * self.num_player))
        self.nue_per_proc_run = self.batch_size * 2 * self.num_player

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player - 1), dtype=bool)

    def _init_indiv(self):
        assert self.nue_track_avg % 2 == 0
        assert self.nue_avg % 2 == 0

    def _generator(self):
        if self.semivalue == "weighted_banzhaf":
            t = self.semivalue_param
        elif self.semivalue == "shapley":
            t = np.random.rand()
        elif self.semivalue == "beta_shapley":
            t = np.random.beta(self.semivalue_param[1], self.semivalue_param[0])
        else:
            raise NotImplementedError
        return np.random.binomial(1, t, size=self.num_player - 1).astype(bool)

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        subset = np.zeros(self.num_player, dtype=bool)
        for i, sample in enumerate(samples):
            results = results_collect[i]
            subset[:self.num_player-1] = sample
            results[-1] -= game.evaluate(subset)
            subset[-1] = 1
            results[-1] += game.evaluate(subset)
            for player in range(self.num_player - 1):
                subset[-1], subset[player] = subset[player], subset[-1]
                results[player] += game.evaluate(subset)
                subset[player] = 0
                results[player] -= game.evaluate(subset)
                subset[player] = 1
                subset[-1], subset[player] = subset[player], subset[-1]
            subset[-1] = 0
        return results_collect

    def _process(self, inputs):
        num_pre = self.results_aggregate["count"]
        num_cur = len(inputs) + num_pre
        self.results_aggregate["estimates"] *= num_pre / num_cur
        self.results_aggregate["estimates"] += inputs.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        return self.results_aggregate["estimates"]


class sampling_lift_paired(sampling_lift):
    def __init__(self, **kwargs):
        super(sampling_lift_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        assert self.nue_track_avg % 2 == 0
        assert self.nue_avg % 2 == 0
        if self.semivalue == "weighted_banzhaf":
            assert self.semivalue_param == 0.5
        if self.semivalue == "beta_shapley":
            assert self.semivalue_param[0] == self.semivalue_param[1]


class WSL(sampling_lift):
    def __init__(self, **kwargs):
        super(WSL, self).__init__(**kwargs)
        self.weights = self.distribution_cardinality() * self.num_player

    def _init_indiv(self):
        assert self.nue_track_avg % 2 == 0
        assert self.nue_avg % 2 == 0
        assert self.semivalue != "shapley"  # for the Shapley, sampling_lift = WSL

    def _generator(self):
        t = np.random.rand()
        return np.random.binomial(1, t, size=self.num_player - 1).astype(bool)

    def run(self, samples):
        results_collect = super(WSL, self).run(samples)
        scalars = self.weights[samples.sum(axis=1)]
        return scalars[:, None] * results_collect


class WSL_paired(WSL):
    def __init__(self, **kwargs):
        super(WSL_paired, self).__init__(**kwargs)
        self.lock_switch = False


class WSL_banzhaf(WSL):
    def __init__(self, **kwargs):
        super(WSL, self).__init__(**kwargs)
        tmp = 2 ** (self.num_player - 1)
        vs = self.distribution_cardinality()
        self.weights = np.array([tmp / special.binom(self.num_player - 1, s) * vs[s] for s in range(self.num_player)])

    def _init_indiv(self):
        assert self.nue_track_avg % 2 == 0
        assert self.nue_avg % 2 == 0
        assert not (self.semivalue == "weighted_banzhaf" and self.semivalue_param == 0.5)

    def _generator(self):
        return np.random.binomial(1, 0.5, size=self.num_player - 1).astype(bool)


class WSL_banzhaf_paired(WSL_banzhaf):
    def __init__(self, **kwargs):
        super(WSL_banzhaf_paired, self).__init__(**kwargs)
        self.lock_switch = False


class permutation(sampling_lift):
    # the evaluation of U(0) is not counted for the total budget of utility evaluations.
    def __init__(self, **kwargs):
        super(sampling_lift, self).__init__(**kwargs)
        self.num_sample = self.nue_avg
        self.interval_track = self.nue_track_avg
        self.batch_size = -(-self.nue_per_proc // self.num_player)
        self.nue_per_proc_run = self.batch_size * self.num_player

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=np.int64)

    def _init_indiv(self):
        assert self.semivalue == "shapley"

    def _generator(self):
        return np.random.permutation(self.num_player)

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        subset = np.zeros(self.num_player, dtype=bool)
        empty_value = game.evaluate(subset)
        for i, sample in enumerate(samples):
            results = results_collect[i]
            pre_value = empty_value
            for j in range(self.num_player):
                player = sample[j]
                results[player] -= pre_value
                subset[player] = True
                cur_value = game.evaluate(subset)
                results[player] += cur_value
                pre_value = cur_value
            subset.fill(False)
        return results_collect


class permutation_paired(permutation):
    def __init__(self, **kwargs):
        super(permutation_paired, self).__init__(**kwargs)
        self.takeInverse = False
        self.pi_pre = None

    def _generator(self):
        if self.takeInverse:
            self.takeInverse = False
            return np.argsort(self.pi_pre)
        else:
            self.takeInverse = True
            self.pi_pre = np.random.permutation(self.num_player)
            return self.pi_pre


class weighted_permutation(permutation):
    def __init__(self, **kwargs):
        super(weighted_permutation, self).__init__(**kwargs)
        self.weights = self.distribution_cardinality() * self.num_player

    def _init_indiv(self):
        assert self.semivalue != "shapley"

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        subset = np.zeros(self.num_player, dtype=bool)
        empty_value = game.evaluate(subset)
        for i, sample in enumerate(samples):
            results = results_collect[i]
            pre_value = empty_value
            for j in range(self.num_player):
                player = sample[j]
                results[player] -= pre_value
                subset[player] = True
                cur_value = game.evaluate(subset)
                results[player] += cur_value
                results[player] *= self.weights[j]
                pre_value = cur_value
            subset.fill(False)
        return results_collect


class weighted_permutation_paired(weighted_permutation, permutation_paired):
    def __init__(self, **kwargs):
        super(weighted_permutation_paired, self).__init__(**kwargs)
        self.takeInverse = False
        self.pi_pre = None

    def _generator(self):
        return permutation_paired._generator(self)


class MSR(estimatorTemplate):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = np.zeros((4, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

    def _init_indiv(self):
        assert self.semivalue == "weighted_banzhaf"
        assert 0 < self.semivalue_param and self.semivalue_param < 1

    def _generator(self):
        return np.random.binomial(1, self.semivalue_param, size=self.num_player).astype(bool)

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.empty((len(samples), self.num_player + 1), dtype=np.float64)
        results_collect[:, :self.num_player] = samples
        for i, sample in enumerate(samples):
            results_collect[i, -1] = game.evaluate(sample)
        return results_collect

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        self.results_aggregate[0] += (ues * subsets).sum(axis=0)
        self.results_aggregate[1] += subsets.sum(axis=0)
        subsets = 1 - subsets
        self.results_aggregate[2] += (ues * subsets).sum(axis=0)
        self.results_aggregate[3] += subsets.sum(axis=0)

    def _estimate(self):
        counts = self.results_aggregate[1].copy()
        counts[counts == 0] = -1
        left = np.divide(self.results_aggregate[0], counts)
        counts = self.results_aggregate[3].copy()
        counts[counts == 0] = -1
        right = np.divide(self.results_aggregate[2], counts)
        return left - right


class MSR_paired(MSR):
    def __init__(self, **kwargs):
        super(MSR_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        assert self.semivalue == "weighted_banzhaf"
        assert self.semivalue_param == 0.5



class improved_AME(estimatorTemplate):
    def __init__(self, **kwargs):
        super(improved_AME, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.samples = np.empty((self.batch_size, self.num_player + 1), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 2), dtype=np.float64)
        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)


    def _generator(self):
        sample = np.empty(self.num_player + 1, dtype=np.float64)
        if self.semivalue == "weighted_banzhaf":
            t = self.semivalue_param
        elif self.semivalue == "shapley":
            t = np.random.rand()
        elif self.semivalue == "beta_shapley":
            t = np.random.beta(self.semivalue_param[1], self.semivalue_param[0])
        else:
            raise NotImplementedError
        sample[-1] = t
        sample[:-1] = np.random.binomial(1, t, size=self.num_player)
        return sample

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.empty((len(samples), self.num_player + 2), dtype=np.float64)
        results_collect[:, :-1] = samples
        for i, sample in enumerate(samples):
            results_collect[i, -1] = game.evaluate(sample[:-1].astype(bool))
        return results_collect

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        weights = inputs[:, [-2]]

        num_pre = self.results_aggregate["count"]
        num_cur = len(inputs) + num_pre
        self.results_aggregate["estimates"] *= num_pre / num_cur

        self.results_aggregate["estimates"] += (ues * np.reciprocal(weights) * subsets).sum(axis=0) / num_cur
        subsets = 1 - subsets
        weights = 1 - weights
        self.results_aggregate["estimates"] -= (ues * np.reciprocal(weights) * subsets).sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur


    def _estimate(self):
        return self.results_aggregate["estimates"]




class weighted_MSR(MSR):
    def __init__(self, **kwargs):
        super(weighted_MSR, self).__init__(**kwargs)
        self.weights = self.distribution_cardinality()
        self.scalar = 2**(self.num_player - 1)

    def _init_indiv(self):
        assert not (self.semivalue == "weighted_banzhaf" and self.semivalue_param == 0.5)

    def _generator(self):
        return np.random.binomial(1, 0.5, size=self.num_player).astype(bool)

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        sizes = subsets.sum(axis=1).astype(np.int64)
        weights = np.array([self.scalar / special.binom(self.num_player - 1, s - 1) * self.weights[s - 1] if s > 0 else 1 for s in sizes])
        self.results_aggregate[0] += (ues * subsets * weights[:, None]).sum(axis=0)
        self.results_aggregate[1] += subsets.sum(axis=0)
        subsets = 1 - subsets
        weights = np.array([self.scalar / special.binom(self.num_player - 1, s) * self.weights[s] if s < self.num_player else 1 for s in sizes])
        self.results_aggregate[2] += (ues * subsets * weights[:, None]).sum(axis=0)
        self.results_aggregate[3] += subsets.sum(axis=0)


class weighted_MSR_paired(weighted_MSR):
    def __init__(self, **kwargs):
        super(weighted_MSR_paired, self).__init__(**kwargs)
        self.lock_switch = False


class kernelSHAP(MSR):
    @staticmethod
    def calculate_constants(game_func, game_args, num_player):
        game = game_func(**game_args)
        subset = np.zeros(num_player, dtype=bool)
        v_empty = game.evaluate(subset)
        subset.fill(True)
        v_full = game.evaluate(subset)
        return v_empty, v_full

    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(mat_A=np.zeros((self.num_player, self.num_player), dtype=np.float64),
                                      vec_b=np.zeros(self.num_player, dtype=np.float64),
                                      count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))

    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = np.arange(1, self.num_player, dtype=np.float64)
        weights = 1 / np.multiply(tmp, tmp[::-1])
        self.weights = weights / weights.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _generator(self):
        s = np.random.choice(self.s_range, p=self.weights)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset = np.zeros(self.num_player, dtype=bool)
        subset[pos] = True
        return subset

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        A_tmp = subsets.T @ subsets
        b_tmp = subsets * (ues - self.constants[0])

        num_pre = self.results_aggregate["count"]
        num_cur = len(b_tmp) + num_pre
        self.results_aggregate["mat_A"] *= num_pre / num_cur
        self.results_aggregate["mat_A"] += A_tmp / num_cur
        self.results_aggregate["vec_b"] *= num_pre / num_cur
        self.results_aggregate["vec_b"] += b_tmp.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        A_inv = np.linalg.pinv(self.results_aggregate["mat_A"])
        vec_b = self.results_aggregate["vec_b"]
        vec_1 = np.ones(len(vec_b))
        v_empty, v_full = self.constants
        tmp = vec_b - (np.dot(vec_1, np.dot(A_inv, vec_b)) - v_full + v_empty) / np.dot(vec_1, np.dot(A_inv, vec_1))
        return np.dot(A_inv, tmp)


class kernelSHAP_paired(kernelSHAP):
    def __init__(self, **kwargs):
        super(kernelSHAP_paired, self).__init__(**kwargs)
        self.lock_switch = False


class leverage(kernelSHAP):
    def __init__(self, **kwargs):
        super(leverage, self).__init__(**kwargs)
        self.results_aggregate = dict(mat_A=np.zeros((self.num_player, self.num_player), dtype=np.float64),
                                      vec_b=np.zeros(self.num_player, dtype=np.float64),
                                      count=0)


    def _init_indiv(self):
        super(leverage, self)._init_indiv()
        self.weights = np.sqrt(self.weights * self.num_player)


    def _generator(self):
        s = np.random.choice(self.s_range)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset = np.zeros(self.num_player, dtype=bool)
        subset[pos] = True
        return subset

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, -1] - self.constants[0]

        sizes = subsets.sum(axis=1)
        vec_tmp = sizes / self.num_player
        subsets -= vec_tmp[:, None]
        weights_tmp = self.weights[sizes.astype(np.int64) - 1]
        subsets = weights_tmp[:, None] * subsets
        A_tmp = subsets.T @ subsets
        b_tmp = ues - ((self.constants[1] - self.constants[0]) / self.num_player) * sizes
        b_tmp = np.multiply(weights_tmp, b_tmp)
        b_tmp = np.dot(b_tmp, subsets)

        num_pre = self.results_aggregate["count"]
        num_cur = len(inputs) + num_pre
        self.results_aggregate["mat_A"] *= num_pre / num_cur
        self.results_aggregate["mat_A"] += A_tmp / num_cur
        self.results_aggregate["vec_b"] *= num_pre / num_cur
        self.results_aggregate["vec_b"] += b_tmp / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        # A_inv = np.linalg.pinv(self.results_aggregate["mat_A"])
        tmp, _, _, _ = lstsq(self.results_aggregate["mat_A"], self.results_aggregate["vec_b"], lapack_driver='gelsy',
                    check_finite=False)
        return tmp + (self.constants[1] - self.constants[0]) / self.num_player

class leverage_paired(leverage):
    def __init__(self, **kwargs):
        super(leverage_paired, self).__init__(**kwargs)
        self.lock_switch = False


class modified_leverage(kernelSHAP):
    def __init__(self, **kwargs):
        super(modified_leverage, self).__init__(**kwargs)
        self.results_aggregate = dict(vec_b=np.zeros(self.num_player, dtype=np.float64),
                                      count=0)


    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = np.arange(1, self.num_player, dtype=np.float64)
        self.weights = (self.num_player - 1) / np.multiply(tmp, tmp[::-1])
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)


    def _generator(self):
        s = np.random.choice(self.s_range)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset = np.zeros(self.num_player, dtype=bool)
        subset[pos] = True
        return subset

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, -1] - self.constants[0]

        sizes = subsets.sum(axis=1)
        vec_tmp = sizes / self.num_player
        subsets -= vec_tmp[:, None]
        b_tmp = ues - ((self.constants[1] - self.constants[0]) / self.num_player) * sizes
        b_tmp = np.multiply(self.weights[sizes.astype(np.int64) - 1], b_tmp)
        b_tmp = np.dot(b_tmp, subsets)

        num_pre = self.results_aggregate["count"]
        num_cur = len(inputs) + num_pre
        self.results_aggregate["vec_b"] *= num_pre / num_cur
        self.results_aggregate["vec_b"] += b_tmp / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        vec = self.results_aggregate["vec_b"]
        tmp = self.num_player * vec - vec.sum()
        return tmp + (self.constants[1] - self.constants[0]) / self.num_player


class modified_leverage_paired(modified_leverage):
    def __init__(self, **kwargs):
        super(modified_leverage_paired, self).__init__(**kwargs)
        self.lock_switch = False


class test_leverage(kernelSHAP):
    def __init__(self, **kwargs):
        super(test_leverage, self).__init__(**kwargs)
        self.results_aggregate = dict(vec_b=np.zeros(self.num_player, dtype=np.float64),
                                      count=0)


    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = np.arange(1, self.num_player, dtype=np.float64)
        tmp = 1 / np.multiply(tmp, tmp[::-1])
        tmp_sum = tmp.sum()
        self.weights = tmp_sum * np.ones(self.num_player - 1, dtype=np.float64)
        self.p = tmp / tmp_sum
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)


    def _generator(self):
        s = np.random.choice(self.s_range, p=self.p)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset = np.zeros(self.num_player, dtype=bool)
        subset[pos] = True
        return subset

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, -1] - self.constants[0]

        sizes = subsets.sum(axis=1)
        vec_tmp = sizes / self.num_player
        subsets -= vec_tmp[:, None]
        b_tmp = ues - ((self.constants[1] - self.constants[0]) / self.num_player) * sizes
        b_tmp = np.multiply(self.weights[sizes.astype(np.int64) - 1], b_tmp)
        b_tmp = np.dot(b_tmp, subsets)

        num_pre = self.results_aggregate["count"]
        num_cur = len(inputs) + num_pre
        self.results_aggregate["vec_b"] *= num_pre / num_cur
        self.results_aggregate["vec_b"] += b_tmp / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        vec = self.results_aggregate["vec_b"]
        tmp = self.num_player * vec - vec.sum()
        return tmp + (self.constants[1] - self.constants[0]) / self.num_player

class test_leverage_paired(test_leverage):
    def __init__(self, **kwargs):
        super(test_leverage_paired, self).__init__(**kwargs)
        self.lock_switch = False


class unbiased_kernelSHAP(kernelSHAP):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))
        self.scalar = 2 * np.reciprocal(np.arange(1, self.num_player, dtype=np.float64)).sum()

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        tmp = (subsets - subsets.sum(axis=1, keepdims=True) / self.num_player) * (ues - self.constants[0])

        num_pre = self.results_aggregate["count"]
        num_cur = len(tmp) + num_pre
        self.results_aggregate["estimates"] *= num_pre / num_cur
        self.results_aggregate["estimates"] += tmp.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        return (self.constants[1] - self.constants[0]) / self.num_player + \
               self.results_aggregate["estimates"] * self.scalar


class unbiased_kernelSHAP_paired(unbiased_kernelSHAP, kernelSHAP_paired):
    def __init__(self, **kwargs):
        super(unbiased_kernelSHAP_paired, self).__init__(**kwargs)
        self.lock_switch = False


class ARM(MSR):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = (self.nue_avg * self.num_player) // 2
        self.interval_track = (self.nue_track_avg * self.num_player) // 2
        self.batch_size = -(-self.nue_per_proc // 2)
        self.nue_per_proc_run = self.batch_size * 2

        self.results_aggregate = np.zeros((4, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, 2, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, 2, self.num_player), dtype=bool)

    def _init_indiv(self):
        assert (self.nue_avg * self.num_player) % 2 == 0
        assert (self.nue_track_avg * self.num_player) % 2 == 0

        weight = self.distribution_cardinality()
        weight_left = np.divide(weight, np.arange(1, self.num_player + 1))
        self.weight_left = weight_left / weight_left.sum()
        weight_right = np.divide(weight, np.arange(self.num_player, 0, -1))
        self.weight_right = weight_right / weight_right.sum()

        self.s_range_left = np.arange(1, self.num_player + 1)
        self.s_range_right = np.arange(self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _generator(self):
        subset = np.zeros((2, self.num_player), dtype=bool)
        s = np.random.choice(self.s_range_left, p=self.weight_left)
        pos_left = np.random.choice(self.pos_range, size=s, replace=False)
        s = np.random.choice(self.s_range_right, p=self.weight_right)
        pos_right = np.random.choice(self.pos_range, size=s, replace=False)
        subset[0, pos_left] = True
        subset[1, pos_right] = True
        return subset

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.empty((len(samples), 2, self.num_player + 1), dtype=np.float64)
        results_collect[:, :, :self.num_player] = samples
        for i, sample in enumerate(samples):
            results_collect[i, 0, -1] = game.evaluate(sample[0])
            results_collect[i, 1, -1] = game.evaluate(sample[1])
        return results_collect

    def _process(self, inputs):
        subsets = inputs[:, 0, :self.num_player]
        ues = inputs[:, 0, [-1]]
        self.results_aggregate[0] += (ues * subsets).sum(axis=0)
        self.results_aggregate[1] += subsets.sum(axis=0)
        subsets = 1 - inputs[:, 1, :self.num_player]
        ues = inputs[:, 1, [-1]]
        self.results_aggregate[2] += (ues * subsets).sum(axis=0)
        self.results_aggregate[3] += subsets.sum(axis=0)


class ARM_shapley(ARM):
    def __init__(self, **kwargs):
        super(ARM_shapley, self).__init__(**kwargs)

    def _init_indiv(self):
        assert self.semivalue != "shapley"
        assert (self.nue_avg * self.num_player) % 2 == 0
        assert (self.nue_track_avg * self.num_player) % 2 == 0

        weight_left = self.num_player / np.arange(1, self.num_player + 1)
        self.weight_left = weight_left / weight_left.sum()
        self.weight_right = self.weight_left[::-1]

        self.s_range_left = np.arange(1, self.num_player + 1)
        self.s_range_right = np.arange(self.num_player)
        self.pos_range = np.arange(self.num_player)

        self.weights = self.distribution_cardinality() * self.num_player

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.empty((len(samples), 2, self.num_player + 1), dtype=np.float64)
        results_collect[:, :, :self.num_player] = samples
        for i, sample in enumerate(samples):
            subset = sample[0]
            results_collect[i, 0, -1] = game.evaluate(subset) * self.weights[subset.sum() - 1]
            subset = sample[1]
            results_collect[i, 1, -1] = game.evaluate(subset) * self.weights[subset.sum()]
        return results_collect


class ARM_banzhaf(ARM_shapley):
    def __init__(self, **kwargs):
        super(ARM_shapley, self).__init__(**kwargs)
        tmp = 2 ** (self.num_player - 1)
        vs = self.distribution_cardinality()
        self.weights = np.array([tmp / special.binom(self.num_player - 1, s) * vs[s] for s in range(self.num_player)])

    def _init_indiv(self):
        # assert not (self.semivalue == "weighted_banzhaf" and self.semivalue_param == 0.5)
        assert (self.nue_avg * self.num_player) % 2 == 0
        assert (self.nue_track_avg * self.num_player) % 2 == 0

        weights = np.ones(self.num_player, dtype=np.float64)
        for k in range(self.num_player):
            for i in range(k):
                weights[k] *= (self.num_player - 1 - i) / (i + 1) * 0.5**2
            weights[k] *= 0.5 ** (self.num_player - 1 - 2 * k)

        weight_left = np.divide(weights, np.arange(1, self.num_player + 1))
        self.weight_left = weight_left / weight_left.sum()
        weight_right = np.divide(weights, np.arange(self.num_player, 0, -1))
        self.weight_right = weight_right / weight_right.sum()

        self.s_range_left = np.arange(1, self.num_player + 1)
        self.s_range_right = np.arange(self.num_player)
        self.pos_range = np.arange(self.num_player)

    def run(self, samples):
        return super(ARM_banzhaf, self).run(samples)


class complement(estimatorTemplate):
    def __init__(self, **kwargs):
        super(complement, self).__init__(**kwargs)
        self.num_sample = (self.nue_avg * self.num_player) // 2
        self.interval_track = (self.nue_track_avg * self.num_player) // 2
        self.batch_size = -(-self.nue_per_proc // 2)
        self.nue_per_proc_run = self.batch_size * 2

        self.results_aggregate = np.zeros((2, self.num_player, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

    def _init_indiv(self):
        assert self.semivalue == "shapley"
        assert (self.nue_avg * self.num_player) % 2 == 0
        assert (self.nue_track_avg * self.num_player) % 2 == 0

        self.s_range = np.arange(1, self.num_player + 1)

    def _generator(self):
        subset = np.zeros(self.num_player, dtype=bool)
        s = np.random.choice(self.s_range)
        pi = np.random.permutation(self.num_player)
        subset[pi[:s]] = True
        # Note what in the above is equal to
        # pos = np.random.choice(np.arange(self.num_player), size=s, replace=False)
        # subset[pos] = True
        # But we stay loyal to the original paper
        return subset

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player + 1), dtype=np.float64)
        results_collect[:, :self.num_player] = samples
        for i, sample in enumerate(samples):
            results_collect[i, -1] += game.evaluate(sample)
            results_collect[i, -1] -= game.evaluate(~sample)
        return results_collect

    def _process(self, inputs):
        for take in inputs:
            subset = take[:self.num_player].astype(bool)
            subset_c = ~subset
            v = take[-1]
            subset_size = subset.sum()
            self.results_aggregate[0, subset, subset_size - 1] += v
            self.results_aggregate[0, subset_c, self.num_player - subset_size - 1] -= v
            self.results_aggregate[1, subset, subset_size - 1] += 1
            self.results_aggregate[1, subset_c, self.num_player - subset_size - 1] += 1

    def _estimate(self):
        # what in the below seems to fail occasionally, it returns nan for some entry while it should be a real number.
        # tmp = np.divide(self.results_aggregate[0], self.results_aggregate[1], where=self.results_aggregate[1] != 0)
        # return tmp.mean(axis=1)
        counts = self.results_aggregate[1].copy()
        counts[counts == 0] = -1
        return np.mean(np.divide(self.results_aggregate[0], counts), axis=1)


class AME(estimatorTemplate):
    def __init__(self, **kwargs):
        super(AME, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(mat_A=np.zeros((self.num_player, self.num_player), dtype=np.float64),
                                      vec_b=np.zeros(self.num_player, dtype=np.float64))
        self.buffer = np.empty((self.buffer_size, self.num_player + 2), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player + 1), dtype=np.float64)

    def _init_indiv(self):
        if self.semivalue == "weighted_banzhaf":
            assert 0 < self.semivalue_param and self.semivalue_param < 1
            self.variance = 1 / self.semivalue_param / (1 - self.semivalue_param)
        elif self.semivalue == "beta_shapley":
            assert 1 < self.semivalue_param[0] and 1 < self.semivalue_param[1]
            alpha, beta = self.semivalue_param
            ab = alpha + beta
            self.variance = (ab - 1) * (ab - 2) / (alpha - 1) / (beta - 1)
        else:
            raise NotImplementedError

    def _generator(self):
        sample = np.empty(self.num_player + 1, dtype=np.float64)
        if self.semivalue == "weighted_banzhaf":
            prob = self.semivalue_param
        elif self.semivalue == "beta_shapley":
            prob = np.random.beta(self.semivalue_param[1], self.semivalue_param[0])
        else:
            raise NotImplementedError
        sample[:-1] = np.random.binomial(1, prob, size=self.num_player)
        sample[-1] = prob
        return sample

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player + 2), dtype=np.float64)
        results_collect[:, :-1] = samples
        for i, sample in enumerate(samples):
            subset = sample[:self.num_player].astype(bool)
            results_collect[i, -1] = game.evaluate(subset)
        return results_collect

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ps = inputs[:, [-2]]
        ues = inputs[:, [-1]]
        tmp = subsets * (1 / ps) - (1 - subsets) * (1 / (1 - ps))
        self.results_aggregate["mat_A"] += tmp.T @ tmp
        self.results_aggregate["vec_b"] += (ues * tmp).sum(axis=0)

    def _estimate(self):
        return self.variance * (np.linalg.pinv(self.results_aggregate["mat_A"]) @ self.results_aggregate["vec_b"])


class AME_paired(AME):
    def __init__(self, **kwargs):
        super(AME_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        super(AME_paired, self)._init_indiv()
        if self.semivalue == "weighted_banzhaf":
            assert self.semivalue_param == 0.5
        if self.semivalue == "beta_shapley":
            assert self.semivalue_param[0] == self.semivalue_param[1]


class group_testing(sampling_lift):
    def __init__(self, **kwargs):
        super(sampling_lift, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player + 1), dtype=bool)

    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = 1 / np.arange(1, self.num_player + 1, dtype=np.float64)
        weights = tmp + tmp[::-1]
        self.const = weights.sum()
        self.weights = weights / self.const
        self.s_range = np.arange(1, self.num_player + 1)
        self.pos_range = np.arange(self.num_player + 1)

    def _generator(self):
        subset = np.zeros(self.num_player + 1, dtype=bool)
        s = np.random.choice(self.s_range, p=self.weights)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset[pos] = True
        return subset

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        for i, sample in enumerate(samples):
            tmp = sample * game.evaluate(sample)
            results_collect[i] = tmp[:self.num_player] - tmp[-1]
        return results_collect * self.const


class group_testing_paired(group_testing):
    def __init__(self, **kwargs):
        super(group_testing_paired, self).__init__(**kwargs)
        self.lock_switch = False


class GELS_ranking(kernelSHAP):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = np.zeros((2, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

    def _init_indiv(self):
        self.num_player -= 1
        weights = self.distribution_cardinality()
        self.num_player += 1
        tmp = np.arange(1, self.num_player, dtype=np.float64)
        tmp = np.multiply(tmp / self.num_player, (self.num_player - tmp) / (self.num_player - 1))
        tmp = np.reciprocal(tmp)
        weights = np.multiply(weights, tmp)
        self.weights = weights / weights.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _generator(self):
        return super(GELS_ranking, self)._generator()

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        self.results_aggregate[0] += (ues * subsets).sum(axis=0)
        self.results_aggregate[1] += subsets.sum(axis=0)

    def _estimate(self):
        counts = self.results_aggregate[1].copy()
        counts[counts == 0] = -1
        return np.divide(self.results_aggregate[0], counts)


class GELS_ranking_paired(GELS_ranking):
    def __init__(self, **kwargs):
        super(GELS_ranking_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        super(GELS_ranking_paired, self)._init_indiv()
        if self.semivalue == "weighted_banzhaf":
            assert self.semivalue_param == 0.5
        if self.semivalue == "beta_shapley":
            assert self.semivalue_param[0] == self.semivalue_param[1]


class GELS(GELS_ranking):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        weights = self.distribution_cardinality()
        self.scalar = (np.divide(weights, np.arange(self.num_player, 0, -1)) * self.num_player).sum()
        self.num_player += 1

        self.results_aggregate = np.zeros((2, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.zeros((self.batch_size, self.num_player), dtype=bool)

    def _estimate(self):
        estimates = super(GELS, self)._estimate() * self.scalar
        return estimates[:-1] - estimates[-1]


class GELS_paired(GELS, GELS_ranking_paired):
    # For the Shapley value, this estimator is equal to group_testing_paired
    def __init__(self, **kwargs):
        super(GELS_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        GELS_ranking_paired._init_indiv(self)


class WGELS_shapley(GELS):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.scalar = (1 / np.arange(self.num_player, 0, -1)).sum()
        self.reweights = self.distribution_cardinality() * self.num_player
        self.num_player += 1

        self.results_aggregate = np.zeros((2, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.zeros((self.batch_size, self.num_player), dtype=bool)


    def _init_indiv(self):
        assert self.semivalue != "shapley"
        tmp = np.arange(1, self.num_player, dtype=np.float64)
        tmp = np.multiply(tmp / self.num_player, (self.num_player - tmp) / (self.num_player - 1))
        tmp = np.reciprocal(tmp)
        weights = tmp / (self.num_player - 1)
        self.weights = weights / weights.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        last_player = subsets[:, -1].astype(bool)
        sizes = subsets[:, :-1].sum(axis=1).astype(np.int64)
        weights = np.array([self.reweights[s] if pre else self.reweights[s-1] for (pre, s) in zip(last_player, sizes)])
        self.results_aggregate[0] += (ues * subsets * weights[:, None]).sum(axis=0)
        self.results_aggregate[1] += subsets.sum(axis=0)


class WGELS_shapley_paired(WGELS_shapley):
    def __init__(self, **kwargs):
        super(WGELS_shapley_paired, self).__init__(**kwargs)
        self.lock_switch = False


class WGELS_banzhaf(WGELS_shapley):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.banzhaf_weights = np.ones(self.num_player, dtype=np.float64)
        for k in range(self.num_player):
            for i in range(k):
                self.banzhaf_weights[k] *= (self.num_player - 1 - i) / (i + 1) * 0.5 ** 2
            self.banzhaf_weights[k] *= 0.5 ** (self.num_player - 1 - 2 * k)
        self.scalar = (np.divide(self.banzhaf_weights, np.arange(self.num_player, 0, -1)) * self.num_player).sum()

        weights = self.distribution_cardinality()
        tmp = 2**(self.num_player - 1)
        self.reweights = np.array([tmp / special.binom(self.num_player - 1, s) * weights[s] for s in range(self.num_player)])
        self.num_player += 1
        self.results_aggregate = np.zeros((2, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.zeros((self.batch_size, self.num_player), dtype=bool)

    def _init_indiv(self):
        assert not (self.semivalue == "weighted_banzhaf" and self.semivalue_param == 0.5)
        tmp = np.arange(1, self.num_player, dtype=np.float64)
        tmp = np.multiply(tmp / self.num_player, (self.num_player - tmp) / (self.num_player - 1))
        tmp = np.reciprocal(tmp)
        weights = np.multiply(self.banzhaf_weights, tmp)
        self.weights = weights / weights.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)


class WGELS_banzhaf_paired(WGELS_banzhaf):
    def __init__(self, **kwargs):
        super(WGELS_banzhaf_paired, self).__init__(**kwargs)
        self.lock_switch = False


class GELS_shapley(GELS_ranking):
    def __init__(self, **kwargs):
        super(GELS_shapley, self).__init__(**kwargs)
        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))

    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = 1 / np.arange(1, self.num_player, dtype=np.float64)
        weights = np.multiply(tmp, tmp[::-1])
        self.weights = weights / weights.sum()
        self.scalar = tmp.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _estimate(self):
        estimates = super(GELS_shapley, self)._estimate() * self.scalar
        offset = (self.constants[1] - self.constants[0] - estimates.sum()) / self.num_player
        return estimates + offset


class GELS_shapley_paired(GELS_shapley):
    # This estimator is equal to unbiased_kernelSHAP_paired
    def __init__(self, **kwargs):
        super(GELS_shapley_paired, self).__init__(**kwargs)
        self.lock_switch = False


class simSHAP(kernelSHAP):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))

    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = np.arange(1, self.num_player, dtype=np.float64)
        weights = 1 / np.multiply(tmp, tmp[::-1])
        self.gamma = weights.sum()
        self.weights = weights / self.gamma
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        sizes = subsets.sum(axis=1, keepdims=True)

        tmp = ((self.num_player - sizes) * subsets - sizes * (1 - subsets)) * ues
        num_pre = self.results_aggregate["count"]
        num_cur = num_pre + ues.shape[0]
        self.results_aggregate["estimates"] *= num_pre / num_cur
        self.results_aggregate["estimates"] += tmp.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        return self.results_aggregate["estimates"] * self.gamma \
               + (self.constants[1] - self.constants[0]) / self.num_player


class simSHAP_paired(simSHAP):
    def __init__(self, **kwargs):
        super(simSHAP_paired, self).__init__(**kwargs)
        self.lock_switch = False


class OFA(MSR):
    @staticmethod
    def calculate_constants(game_func, game_args, num_player):
        game = game_func(**game_args)
        subset = np.zeros(num_player, dtype=bool)
        v_empty = game.evaluate(subset)
        v_singleton = np.empty(num_player, dtype=np.float64)
        for i in range(num_player):
            subset[i] = True
            v_singleton[i] = game.evaluate(subset)
            subset[i] = False

        subset.fill(True)
        v_full = game.evaluate(subset)
        v_remove = np.empty(num_player, dtype=np.float64)
        for i in range(num_player):
            subset[i] = False
            v_remove[i] = game.evaluate(subset)
            subset[i] = True

        return v_empty, v_full, v_singleton, v_remove

    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(estimates=np.zeros((self.num_player, self.num_player - 3, 2), dtype=np.float64),
                                      counts=np.zeros((self.num_player, self.num_player - 3, 2), dtype=np.int64))
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))
        self.weights = self.distribution_cardinality()

        self.p_sampling = None

    def _init_indiv(self):
        self.s_range = np.arange(2, self.num_player - 1)
        self.pos_range = np.arange(self.num_player)


    def _generator(self):
        subset = np.zeros(self.num_player, dtype=bool)
        if self.p_sampling is None:
            s = np.random.choice(self.s_range)
        else:
            s = np.random.choice(self.s_range, p=self.p_sampling)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset[pos] = True
        return subset

    def _process(self, inputs):
        for take in inputs:
            subset = take[:self.num_player].astype(bool)
            subset_c = ~subset
            v = take[-1]
            idx = subset.sum() - 2
            counts_pre = self.results_aggregate["counts"][subset, idx, 0]
            counts_cur = counts_pre + 1
            self.results_aggregate["estimates"][subset, idx, 0] *= counts_pre / counts_cur
            self.results_aggregate["estimates"][subset, idx, 0] += v / counts_cur
            self.results_aggregate["counts"][subset, idx, 0] += 1

            counts_pre = self.results_aggregate["counts"][subset_c, idx, 1]
            counts_cur = counts_pre + 1
            self.results_aggregate["estimates"][subset_c, idx, 1] *= counts_pre / counts_cur
            self.results_aggregate["estimates"][subset_c, idx, 1] += v / counts_cur
            self.results_aggregate["counts"][subset_c, idx, 1] += 1

    def _estimate(self):
        tmp = (self.results_aggregate["estimates"][:, :, 0] * self.weights[None, 1:self.num_player - 2]).sum(axis=1)
        tmp += self.constants[1] * self.weights[-1]
        tmp += self.constants[2] * self.weights[0]
        tmp += (self.constants[3].sum() - self.constants[3]) * self.weights[-2] / (self.num_player - 1)

        tmp -= (self.results_aggregate["estimates"][:, :, 1] * self.weights[None, 2:self.num_player - 1]).sum(axis=1)
        tmp -= self.constants[0] * self.weights[0]
        tmp -= self.constants[3] * self.weights[-1]
        tmp -= (self.constants[2].sum() - self.constants[2]) * self.weights[1] / (self.num_player - 1)
        return tmp


class OFA_optimal(OFA):
    def __init__(self, **kwargs):
        super(OFA_optimal, self).__init__(**kwargs)
        assert self.semivalue != "shapley"
        tmp = self.num_player / np.arange(2, self.num_player - 1)
        tmp = np.sqrt(tmp + tmp[::-1])
        self.p_sampling = tmp / tmp.sum()


class OFA_optimal_paired(OFA_optimal):
    def __init__(self, **kwargs):
        super(OFA_optimal_paired, self).__init__(**kwargs)
        self.lock_switch = False


class OFA_fixed(OFA):
    def __init__(self, **kwargs):
        super(OFA_fixed, self).__init__(**kwargs)
        weights = self.distribution_cardinality()
        tmp = weights[1:self.num_player - 2]**2 / np.arange(2, self.num_player - 1)
        tmp += weights[2:self.num_player - 1]**2 / np.arange(self.num_player - 2, 1, -1)
        tmp = tmp**0.5
        self.p_sampling = tmp / tmp.sum()


class OFA_fixed_paired(OFA_fixed):
    def __init__(self, **kwargs):
        super(OFA_fixed_paired, self).__init__(**kwargs)
        self.lock_switch = False

        if self.semivalue == "weighted_banzhaf":
            assert self.semivalue_param == 0.5
        if self.semivalue == "beta_shapley":
            assert self.semivalue_param[0] == self.semivalue_param[1]


class OFA_baseline(estimatorTemplate):
    def __init__(self, **kwargs):
        super(OFA_baseline, self).__init__(**kwargs)
        assert self.nue_avg % 2 == 0
        self.num_sample = self.nue_avg // 2
        assert self.nue_track_avg % 2 == 0
        self.interval_track = self.nue_track_avg // 2
        self.batch_size = -(-self.nue_per_proc // (self.num_player * 2))
        self.nue_per_proc_run = self.batch_size * self.num_player * 2

        self.results_aggregate = np.zeros((self.num_player, self.num_player), dtype=np.float64)
        self.count = np.zeros(self.num_player, dtype=np.int64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=np.int64)

        self.weights = self.distribution_cardinality()

    def _init_indiv(self):
        self.current_player = 0
        self.index = np.ones(self.num_player, dtype=bool)
        self.players = np.arange(self.num_player)


    def _generator(self):
        subset = np.empty(self.num_player, dtype=np.int64)
        subset[0] = self.current_player
        self.index[self.current_player] = False
        pi = np.random.permutation(self.num_player - 1)
        subset[1:] = self.players[self.index][pi]
        self.index[self.current_player] = True
        self.current_player = (self.current_player + 1) % self.num_player
        return subset


    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player + 1), dtype=np.float64)
        results_collect[:, 0] = samples[:, 0]
        subset = np.zeros(self.num_player, dtype=bool)
        for i, sample in enumerate(samples):
            current_player = sample[0]
            perm = sample[1:]
            results_collect[i, 1] -= game.evaluate(subset)
            subset[current_player] = True
            results_collect[i, 1] += game.evaluate(subset)
            for j, player in enumerate(perm):
                subset[player] = True
                results_collect[i, j + 2] += game.evaluate(subset)
                subset[current_player] = False
                results_collect[i, j + 2] -= game.evaluate(subset)
                subset[current_player] = True
            subset.fill(False)
        return results_collect


    def _process(self, inputs):
        for take in inputs:
            current_player = int(take[0])
            count_cur = self.count[current_player]
            self.results_aggregate[current_player] *= count_cur / (count_cur + 1)
            self.results_aggregate[current_player] += take[1:] / (count_cur + 1)
            self.count[current_player] += 1


    def _estimate(self):
        return (self.results_aggregate * self.weights[None, :]).sum(axis=1)


class OFA_baseline_paired(OFA_baseline):
    def __init__(self, **kwargs):
        super(OFA_baseline_paired, self).__init__(**kwargs)
        self.pi_pre = None
        self.take_inverse = False

    def _generator(self):
        subset = np.empty(self.num_player, dtype=np.int64)
        if self.take_inverse:
            subset[0] = self.current_player
            subset[1:] = self.players[self.index][np.argsort(self.pi_pre)]
            self.index[self.current_player] = True
            self.current_player = (self.current_player + 1) % self.num_player
            self.take_inverse = False
        else:
            subset[0] = self.current_player
            self.index[self.current_player] = False
            pi = np.random.permutation(self.num_player - 1)
            subset[1:] = self.players[self.index][pi]
            self.pi_pre = pi
            self.take_inverse = True
        return subset


class SHAP_IQ(kernelSHAP):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

        with mp.Pool(1) as pool:
            constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))
        self.scalar = 2 * np.reciprocal(np.arange(1, self.num_player, dtype=np.float64)).sum()

        weights = self.distribution_cardinality()
        self.constant = (constants[1] - constants[0]) * weights[-1]
        self.empty = constants[0]
        tmp = np.arange(self.num_player - 1, -1, -1)
        self.weights_p = tmp * weights
        self.weights_n = tmp[::-1] * weights

    def _init_indiv(self):
        tmp = np.arange(1, self.num_player, dtype=np.float64)
        weights = 1 / np.multiply(tmp, tmp[::-1])
        self.weights = weights / weights.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player].astype(bool)
        ues = inputs[:, [-1]]
        sizes = subsets.sum(axis=1)
        tmp = subsets * self.weights_p[sizes - 1][:, None]
        subsets = ~subsets
        tmp -= subsets * self.weights_n[sizes][:, None]
        tmp = tmp * (ues - self.empty)

        num_pre = self.results_aggregate["count"]
        num_cur = len(tmp) + num_pre
        self.results_aggregate["estimates"] *= num_pre / num_cur
        self.results_aggregate["estimates"] += tmp.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        return self.constant + \
               self.results_aggregate["estimates"] * self.scalar


class SHAP_IQ_paired(SHAP_IQ):
    def __init__(self, **kwargs):
        super(SHAP_IQ_paired, self).__init__(**kwargs)
        self.lock_switch = False

