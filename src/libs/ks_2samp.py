from libs.kolmogn import kolmogn
import numpy as np

def ks_2samp(data1, data2, alternative='two-sided'):

    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
        alternative.lower()[0], alternative)
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError(f'Invalid value for alternative: {alternative}')
    if np.ma.is_masked(data1):
        data1 = data1.compressed()
    if np.ma.is_masked(data2):
        data2 = data2.compressed()
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    if min(n1, n2) == 0:
        raise ValueError('Data passed to ks_2samp must not be empty')

    data_all = np.concatenate([data1, data2])
    # using searchsorted solves equal data problem
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    cddiffs = cdf1 - cdf2

    # Identify the location of the statistic
    argminS = np.argmin(cddiffs)
    argmaxS = np.argmax(cddiffs)

    # Ensure sign of minS is not negative.
    minS = np.clip(-cddiffs[argminS], 0, 1)
    maxS = cddiffs[argmaxS]

    if alternative == 'less' or (alternative == 'two-sided' and minS > maxS):
        d = minS
    else:
        d = maxS

    prob = -np.inf

    m, n = sorted([float(n1), float(n2)], reverse=True)
    en = m * n / (m + n)
    if alternative == 'two-sided':
        prob = kolmogn(x=d, n=int(np.round(en)),cdf=False)
    else:
        z = np.sqrt(en) * d
        expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
        prob = np.exp(expt)

    prob = np.clip(prob, 0, 1)
    return d, prob