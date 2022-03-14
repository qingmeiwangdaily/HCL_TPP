import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor


class SeqGenerator:
    def __init__(self, model: nn.Module, record_intensity: bool = False):
        self.model = model
        self.process_dim = model.input_size - 1  # process dimension
        print("Process models dim:\t{}\tHidden units:\t{}".format(self.process_dim, model.hidden_size))
        self.event_times = []
        self.event_types = []
        self.decay_hist = []
        self.hidden_hist = []
        self.intens_hist = []
        self._plot_times = []
        self.event_intens = []
        self.record_intensity: bool = record_intensity

    def _restart_sequence(self):
        self.event_times = []
        self.event_types = []
        self.event_intens = []
        self.decay_hist = []
        self.hidden_hist = []
        self.intens_hist = []
        self._plot_times = []


    def generate_sequence(self, tmax: float, record_intensity: bool):
        raise NotImplementedError


    def plot_events_and_intensity(self, model_name: str = None, debug=False):
        gen_seq_times = self.event_times
        gen_seq_types = self.event_types
        sequence_length = len(gen_seq_times)
        print("no. of events: {}".format(sequence_length))
        evt_times = np.array(gen_seq_times)
        evt_types = np.array(gen_seq_types)
        fig, ax = plt.subplots(1, 1, sharex='all', dpi=100,
                               figsize=(9, 4.5))
        ax: plt.Axes
        inpt_size = self.process_dim
        ax.set_xlabel('Time $t$ (s)')
        intens_hist = np.stack(self.intens_hist)[:, 0]
        labels = ["type {}".format(i) for i in range(self.process_dim)]
        for y, lab in zip(intens_hist.T, labels):
            # plot intensity curve
            ax.plot(self._plot_times, y, linewidth=.7, label=lab)
            # pass
        ax.set_ylabel(r"Intensities $\lambda^i_t$")
        title = "Event arrival times and intensities for generated sequence"
        if model_name is None:
            model_name = self.model.__class__.__name__
        title += " ({})".format(model_name)
        ax.set_title(title)
        ylims = ax.get_ylim()
        ts_y = np.stack(self.event_intens)[:, 0]
        for k in range(inpt_size):
            mask = evt_types == k
            print(k, end=': ')
            if k == self.process_dim:
                print("starter type")
                # label = "start event".format(k)
                y = self.intens_hist[0].sum(axis=1)
            else:
                print("type {}".format(k))
                y = ts_y[mask, k]
                # label = "type {} event".format(k)
            # plot event point
            # ax.scatter(evt_times[mask], y, s=9, zorder=5, alpha=0.8)
            # plot time column
            ax.vlines(evt_times[mask], ylims[0], ylims[1], linewidth=0.2, linestyles='-', alpha=0.5)

        # Useful for debugging the sampling for the intensity curve.
        if debug:
            for s in self._plot_times:
                ax.vlines(s, ylims[0], ylims[1], linewidth=0.3, linestyles='--', alpha=0.6, colors='red')

        ax.set_ylim(*ylims)
        ax.legend()
        fig.tight_layout()
        return fig


def predict_from_hidden(model, h_t_vals, dt_vals, next_dt, next_type, plot, hmax: float = 40.,
                        n_samples=1000, print_info: bool = False):
    model.eval()
    timestep = hmax / n_samples

    intens_t_vals: Tensor = model.intensity_layer(h_t_vals)
    intens_t_vals_sum = intens_t_vals.sum(dim=1)
    integral_ = torch.cumsum(timestep * intens_t_vals_sum, dim=0)
    # density for the time-until-next-event law
    density = intens_t_vals_sum * torch.exp(-integral_)
    # Check density
    if print_info:
        print("sum of density:", (timestep * density).sum())
    t_pit = dt_vals * density  # integrand for the time estimator
    ratio = intens_t_vals / intens_t_vals_sum[:, None]
    prob_type = ratio * density[:, None]  # integrand for the types
    # trapeze method
    estimate_dt = (timestep * 0.5 * (t_pit[1:] + t_pit[:-1])).sum()
    estimate_type_prob = (timestep * 0.5 * (prob_type[1:] + prob_type[:-1])).sum(dim=0)
    if print_info:
        print("type probabilities:", estimate_type_prob)
    estimate_type = torch.argmax(estimate_type_prob)
    next_dt += 1e-5
    error_dt = ((estimate_dt - next_dt)/next_dt)** 2#, normalization, np.abs,
    if plot:
        process_dim = model.process_dim
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        ax0.plot(dt_vals.numpy(), density.numpy(),
                 linestyle='-', linewidth=.8)
        ax0.set_title("Probability density $p_i(u)$\nof the next increment")
        ax0.set_xlabel("Time $u$")
        ax0.set_ylabel('density $p_i(u)$')
        ylims = ax0.get_ylim()
        ax0.vlines(estimate_dt.item(), *ylims,
                   linestyle='--', linewidth=.7, color='red',
                   label=r'estimate $\hat{t}_i - t_{i-1}$')
        ax0.vlines(next_dt.item(), *ylims,
                   linestyle='--', linewidth=.7, color='green',
                   label=r'true $t_i - t_{i-1}$')
        ax0.set_ylim(ylims)
        ax0.legend()
        ax1.plot(dt_vals.numpy(), intens_t_vals_sum.numpy(),
                 linestyle='-', linewidth=.7, label=r'total intensity $\bar\lambda$')
        for k in range(process_dim):
            ax1.plot(dt_vals.numpy(), intens_t_vals[:, k].numpy(),
                     label='type {}'.format(k),
                     linestyle='--', linewidth=.7)
        ax1.set_title("Intensities")
        ax1.set_xlabel("Time $t$")
        ax1.legend()
        # definite integral of the density
        return (estimate_dt, next_dt, error_dt, next_type, estimate_type), fig
    return estimate_dt, next_dt, error_dt, next_type, estimate_type
