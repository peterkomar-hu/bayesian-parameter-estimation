import matplotlib.pyplot as plt
import numpy as np
import pickle
import pystan


# def get_model(model_code_file, model_pickle_file):
#     try:
#         f_model_code = open(model_code_file, 'r')
#         model_code = f_model_code.read()
#         f_model_code.close()
#     except:
#         raise 'Failed to read model_code_file {}.'.format(model_code_file)

#     needs_rebuild = False
#     try:
#         f_model_pickle = open(model_pickle_file, 'rb')
#         f_model_pickle.close()
#     except:
#         print 'Failed to read model_pickle_file {}'.format(model_pickle_file)
#         needs_rebuild = True

#     if not needs_rebuild:
#         with open(model_pickle_file, 'rb') as f_model_pickle:
#             try:
#                 model = pickle.load(f_model_pickle)
#                 assert model.model_code.strip() == model_code.strip()
#                 return model
#             except:
#                 print 'Model in model_pickle_file {} is different '.format(model_pickle_file) + \
#                       'from model in model_code_file {}.'.format(model_code_file)
#                 needs_rebuild = True

#     if needs_rebuild:
#         model = pystan.StanModel(model_code=model_code)
#         with open(model_pickle_file, 'wb') as f_model_pickle:
#             pickle.dump(model, f_model_pickle)
#         return model


def plot_traces(fit, chains=None):
    labels = fit.flatnames + ['log-prob']
    samples = fit.extract(permuted=False, inc_warmup=False)
    if chains is not None:
        samples = samples[:, chains, :]
    chains_to_show = samples.shape[1]
    t = range(len(samples))

    fig, axes_rows = plt.subplots(len(labels), chains_to_show, figsize=(3*chains_to_show, len(labels)*1), sharey='row')
    for row_idx, row in enumerate(axes_rows):
        row[0].set_ylabel(labels[row_idx], rotation=0)
        for col_idx, ax in enumerate(row):
            ax.plot(t, samples[:, col_idx, row_idx])

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_posteriors(fit, chains=None):
    labels = fit.flatnames + ['log-prob']
    samples = fit.extract(permuted=False, inc_warmup=False)
    if chains is not None:
        samples = samples[:, chains, :]
    fig, axes = plt.subplots(len(labels), 1, figsize=(5, len(labels)*2))
    for idx, par in enumerate(labels):
        ax = axes[idx]
        ax.hist(samples[:, :, idx].flatten(), bins=50, normed=True)
        ax.set_xlabel(par)
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def posterior_stats(fit, chains=None):
    samples = fit.extract(permuted=False, inc_warmup=False)
    if chains is not None:
        samples = samples[:, chains, :]
    labels = fit.flatnames + ['log-prob']
    par_chains = []
    for idx, par in enumerate(labels):
        par_chains.append(samples[:, :, idx].flatten())
    par_chains = np.array(par_chains)

    mean = np.mean(par_chains, axis=1)
    cov = np.cov(par_chains)
    std = np.sqrt(np.diag(cov))
    corr = np.corrcoef(par_chains)

    stats = {
        'par': labels,
        'mean': mean,
        'std': std,
        'corr': corr
    }
    return stats


def subsample_fits(fit, draws=10, chains=None):
    labels = fit.flatnames
    samples = fit.extract(permuted=False, inc_warmup=False)
    if chains is not None:
        samples = samples[:, chains, :]

    chain_idxs = np.random.choice(samples.shape[1], size=draws)
    sample_idxs = np.random.choice(samples.shape[0], size=draws, replace=False)

    mcmc_fits = []
    for idx in range(draws):
        mcmc_fits.append(dict(zip(labels, samples[sample_idxs[idx], chain_idxs[idx], :-1])))

    return mcmc_fits
