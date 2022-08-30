import numpy as np
import pickle
from typing import Tuple, List


# Hawkes process
def exp_kernel(t_current: float, t_history: np.ndarray) -> np.ndarray:
    """
    calculate kappa(t-t') = exp(-(t-t'))
    :param t_current: current timestamp
    :param t_history: a vector of timestamps with size (m,)
    :return:
        return
        kappa: a matrix with size (m,)
    """
    dt = t_current - t_history
    kappa = np.exp(-dt)
    kappa[kappa >= 1] = 0
    return kappa


def exp_kernel_int(dt: np.ndarray) -> np.ndarray:
    """
    calculate the integration of exponential impact kernel
    :param dt: a nonnegative ndarray with any size
    :return:
        kappa_int: 1 - exp(-dt)
    """
    dt[dt < 0] = 0
    return 1 - np.exp(-dt)


def hawkes_intensity(t_current: float, time_sequence: np.ndarray, type_sequence: np.ndarray,
                     mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Compute the intensity function of a multi-variate Hawkes process with exponential impact kernel at time t_current
    :param t_current: a float number representing current timestamp
    :param time_sequence: a float array with size (m,) representing history event timestamps
    :param type_sequence: a long array with size (m,) representing history event types
    :param mu: a vector with size (d,), representing the base intensity, each dimension corresponds to an event-type
    :param alpha: a matrix with size (d,d), representing the infectivity matrix.
    :return:
        lambda_t: a d-dimensional array representing the intensity function at time t_current
    """
    dim = mu.shape[0]
    lambda_t = np.zeros((dim,))
    # Implement the intensity function of Hawkes process with exponential kernel
    kernel = exp_kernel(t_current, time_sequence)
    if time_sequence.shape[0] > 0:
        lambda_t = mu + np.sum(alpha[:, type_sequence] @ kernel.reshape(kernel.shape[0], 1), axis=1)
    else:
        lambda_t = mu
    return lambda_t


def hawkes_simulator(num_seqs: int, time_interval: float, mu: np.ndarray, alpha: np.ndarray) -> List[List[np.ndarray]]:
    """
    Simulate event sequences based on a multi-variate Hawkes process with exponential impact kernel
    :param num_seqs: the number of event sequences
    :param time_interval: the length of time interval per sequence
    :param mu: a vector with size (d,), representing the base intensity, each dimension corresponds to an event-type
    :param alpha: a matrix with size (d,d), representing the infectivity matrix.
    :return:
        sequences: a list of the lists in the format [array_time, array_type].
        For the i-th list, the first array with size (M_i, ) represents M_i timestamps,
        the second array with size (M_i, ) represents M_i event types
    """
    sequences = []
    for i in range(num_seqs):
        current_t = 0.0
        num_events = 0
        time_sequence = []
        type_sequence = []
        # implement the Ogata's thinning algorithm
        lambda_t = mu
        while current_t < time_interval:
            dt = np.random.exponential(1 / np.sum(lambda_t))
            u = np.random.rand()
            lambda_tdt = hawkes_intensity(t_current=current_t+dt,
                                          time_sequence=np.array(time_sequence),
                                          type_sequence=np.array(type_sequence),
                                          mu=mu,
                                          alpha=alpha)
            if current_t + dt < time_interval and u <= np.sum(lambda_t) / np.sum(lambda_tdt):
                time_sequence.append(current_t + dt)
                d = np.random.multinomial(n=1, pvals=(lambda_tdt / np.sum(lambda_tdt)).tolist()).argmax()
                type_sequence.append(d)
            current_t = current_t + dt
            lambda_t = hawkes_intensity(t_current=current_t,
                                        time_sequence=np.array(time_sequence),
                                        type_sequence=np.array(type_sequence),
                                        mu=mu,
                                        alpha=alpha)
        print('Seq {}, length={}'.format(i + 1, len(time_sequence)))
        sequences.append([np.array(time_sequence), np.array(type_sequence)])
    return sequences


def hawkes_mle_learner(sequences: List[List[np.ndarray]], dim: int, num_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Learning a Hawkes process with exponential impact kernel by maximum likelihood estimation
    :param sequences: a list of the lists in the format [array_time, array_type].
        For the i-th list, the first array with size (M_i, ) represents M_i timestamps,
        the second array with size (M_i, ) represents M_i event types
    :param dim: the dimension of the Hawkes process, which is equal to the number of event types appearing in data
    :param num_iter: the number of EM iterations
    :return:
        mu: a vector withs size (dim,) representing the base intensity
        alpha: a matrix with size (dim, dim) representing infectivity matrix
    """
    mu = np.random.RandomState(seed=1).rand(dim)
    alpha = np.random.RandomState(seed=1).rand(dim, dim)
    # implement the MLE of Hawkes process
    N = len(sequences)

    # \sum_{i=1}^{N} T_i
    T = [seq[0][-1] for seq in sequences]

    # \sum_{i=1}^{N} \sum_{d_{m,i}=d'} K(T_i-t_{m,i})
    K = np.zeros((dim,))
    for i in range(N):
        time_sequence = sequences[i][0]
        type_sequence = sequences[i][1]
        for idx, type in enumerate(type_sequence):
            K[type] += exp_kernel_int(np.array([T[i]-time_sequence[idx]]))

    for k in range(num_iter):
        new_mu = np.zeros(mu.shape)
        for i in range(N):
            time_sequence = sequences[i][0]
            type_sequence = sequences[i][1]
            for idx, type in enumerate(type_sequence):
                lambda_t = hawkes_intensity(t_current=time_sequence[idx],
                                            time_sequence=np.array(time_sequence[:idx]),
                                            type_sequence=np.array(type_sequence[:idx]),
                                            mu=mu,
                                            alpha=alpha)
                new_mu[type] += mu[type] / lambda_t[type]

        new_alpha = np.zeros(alpha.shape)
        for i in range(N):
            time_sequence = sequences[i][0]
            type_sequence = sequences[i][1]
            for idx1, type1 in enumerate(type_sequence):
                for idx2, type2 in enumerate(type_sequence[:idx1]):
                    lambda_t = hawkes_intensity(t_current=time_sequence[idx1],
                                                time_sequence=np.array(time_sequence[:idx1]),
                                                type_sequence=np.array(type_sequence[:idx1]),
                                                mu=mu,
                                                alpha=alpha)
                    new_alpha[type1, type2] += alpha[type1, type2] * np.exp(-(time_sequence[idx1]-time_sequence[idx2])) / lambda_t[type1]

        mu = new_mu / np.sum(T)
        alpha = new_alpha / K
    return mu, alpha


def hawkes_rmle_learner(sequences: List[List[np.ndarray]], dim: int,
                        num_iter: int, beta: float, reg_type: str = 'Fro') -> Tuple[np.ndarray, np.ndarray]:
    """
    Learning a Hawkes process with exponential impact kernel by maximum likelihood estimation with a regularizer

    min_{mu > 0, A > 0} -log L(mu, A; S) + beta * ||A||_F^2 (or + beta * ||A||_1)

    :param sequences: a list of the lists in the format [array_time, array_type].
        For the i-th list, the first array with size (M_i, ) represents M_i timestamps,
        the second array with size (M_i, ) represents M_i event types
    :param dim: the dimension of the Hawkes process, which is equal to the number of event types appearing in data
    :param num_iter: the number of EM iterations
    :param beta: the weight of regularizer
    :param reg_type: 'Fro' or 'L1'
    :return:
        mu: a vector withs size (dim,) representing the base intensity
        alpha: a matrix with size (dim, dim) representing infectivity matrix
    """
    mu = np.random.RandomState(seed=1).rand(dim)
    alpha = np.random.RandomState(seed=1).rand(dim, dim)
    # implement regularized MLE of Hawkes process
    N = len(sequences)

    T = [seq[0][-1] for seq in sequences]
    K = np.zeros((dim,))
    for i in range(N):
        time_sequence = sequences[i][0]
        type_sequence = sequences[i][1]
        for idx, type in enumerate(type_sequence):
            K[type] += exp_kernel_int(np.array([T[i]-time_sequence[idx]]))

    for k in range(num_iter):
        new_mu = np.zeros(mu.shape)
        for i in range(N):
            time_sequence = sequences[i][0]
            type_sequence = sequences[i][1]
            for idx, type in enumerate(type_sequence):
                lambda_t = hawkes_intensity(t_current=time_sequence[idx],
                                            time_sequence=np.array(time_sequence[:idx]),
                                            type_sequence=np.array(type_sequence[:idx]),
                                            mu=mu,
                                            alpha=alpha)
                new_mu[type] += mu[type] / lambda_t[type]

        p = np.zeros(alpha.shape)
        for i in range(N):
            time_sequence = sequences[i][0]
            type_sequence = sequences[i][1]
            for idx1, type1 in enumerate(type_sequence):
                for idx2, type2 in enumerate(type_sequence[:idx1]):
                    lambda_t = hawkes_intensity(t_current=time_sequence[idx1],
                                                time_sequence=np.array(time_sequence[:idx1]),
                                                type_sequence=np.array(type_sequence[:idx1]),
                                                mu=mu,
                                                alpha=alpha)
                    p[type1, type2] += alpha[type1, type2] * np.exp(-(time_sequence[idx1]-time_sequence[idx2])) / lambda_t[type1]

        mu = new_mu / np.sum(T)
        if reg_type == 'Fro':
            alpha = (-K+np.sqrt(K*K+8*beta*p))/(4*beta)
            # alpha = p / (K + 2*beta)
        elif reg_type == 'L1':
            alpha = p / (K + beta)
    return mu, alpha


def rmse(real: np.ndarray, est: np.ndarray) -> np.ndarray:
    """
    Relative mean square error
    ||real - est||^2 / ||real||^2
    :param real: an arbitrary size array
    :param est: an array with the same size
    :return: the rmse
    """
    # return np.sum(np.abs(real - est) ** 2) / np.sum(np.abs(real) ** 2)
    # return np.sum(||est - real||_2) /sum(||real||_2)
    return np.linalg.norm((est - real), ord = 2)/np.linalg.norm(real, ord = 2)
np.random.seed(1)
# predefined Hawkes process model
time_length = 20.0
num_sequences = 50
num_em_iter = 10
# reg = 3
mu_real = np.array([0.4, 0.3, 0.2, 0.15, 0.1])
d = mu_real.shape[0]
alpha_real = np.array([[0.1, 0.1, 0.0, 0.0, 0.0],
                       [0.1, 0.2, 0.0, 0.1, 0.1],
                       [0.1, 0.0, 0.2, 0.2, 0.0],
                       [0.0, 0.1, 0.1, 0.1, 0.1],
                       [0.2, 0.0, 0.3, 0.1, 0.3]])

"""Implement intensity function + Simulate a set of event sequences"""
simulated_sequences = hawkes_simulator(num_seqs=num_sequences, time_interval=time_length, mu=mu_real, alpha=alpha_real)
with open('./tpp-data/data_retweet/train.pkl', 'wb') as f:
    pickle.dump(simulated_sequences, f)

"""Learning Hawkes process by MLE"""
with open('./tpp-data/data_retweet/train.pkl', 'rb') as f:
    training_sequences = pickle.load(f, encoding='latin-1')
# learning by given sequences
mu_est1, alpha_est1 = hawkes_mle_learner(sequences=training_sequences, dim=d, num_iter=num_em_iter)
#
# learning by your simulated sequences
mu_est2, alpha_est2 = hawkes_mle_learner(sequences=simulated_sequences, dim=d, num_iter=num_em_iter)
