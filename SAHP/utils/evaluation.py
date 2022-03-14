import numpy as np
import tqdm
from models.base import SeqGenerator


def generate_multiple_sequences(generator: SeqGenerator, tmax: float, n_gen_seq: int = 100):
    """

    Args:
        generator:
        tmax: end time for the simulations
        n_gen_seq: number of samples to take
    """
    print("tmax:", tmax)
    # Build a statistic for the no. of events
    gen_seq_lengths = []
    gen_seq_types_lengths = []
    for i in range(n_gen_seq):
        print('Generating the {} sequence'.format(i))
        generator.generate_sequence(tmax, record_intensity=False)
        gen_seq_times = generator.event_times
        gen_seq_types = np.array(generator.event_types)
        gen_seq_lengths.append(len(gen_seq_times))
        gen_seq_types_lengths.append([
            (gen_seq_types == i).sum() for i in range(generator.model.input_size)
        ])
    gen_seq_lengths = np.array(gen_seq_lengths)
    gen_seq_types_lengths = np.array(gen_seq_types_lengths)

    print("Mean generated sequence length: {}".format(gen_seq_lengths.mean()))
    print("Generated sequence length std. dev: {}".format(gen_seq_lengths.std()))
    return gen_seq_lengths, gen_seq_types_lengths


def predict_test(model, seq_times, seq_types, seq_lengths, pad, device='cpu',
                 hmax: float = 40., use_jupyter: bool = False, rnn: bool = True):
    """Run predictions on testing dataset

    Args:
        seq_lengths:
        seq_types:
        seq_times:
        model:
        hmax:
        use_jupyter:

    Returns:

    """
    incr_estimates = []
    incr_real = []
    incr_errors = []
    types_real = []
    types_estimates = []
    test_size = seq_times.shape[0]
    if use_jupyter:
        index_range_ = tqdm.tnrange(test_size)
    else:
        index_range_ = tqdm.trange(test_size)
    for index_ in index_range_:
        _seq_data = (seq_times[index_],
                     seq_types[index_],
                     seq_lengths[index_])
        if rnn:
            est, real_dt, err, real_type, est_type = model.read_predict(*_seq_data, hmax)
        else:
            est, real_dt, err, real_type, est_type = model.read_predict(*_seq_data, pad, device, hmax)

        if err != err: # is nan
            continue
        incr_estimates.append(est)
        incr_real.append(real_dt)
        incr_errors.append(err)
        types_real.append(real_type)
        types_estimates.append(est_type)

    incr_estimates = np.asarray(incr_estimates)
    incr_errors = np.asarray(incr_errors)
    types_real = np.asarray(types_real)
    types_estimates = np.asarray(types_estimates)
    return incr_estimates, incr_errors, types_real, types_estimates

