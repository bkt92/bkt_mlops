import numpy as np
from scipy.special import smirnov
cimport cython

_E128 = 128
_EP128 = np.ldexp(np.longdouble(1), _E128)
_EM128 = np.ldexp(np.longdouble(1), -_E128)
_SQRT2PI = np.sqrt(2 * np.pi)
_LOG_2PI = np.log(2 * np.pi)
_PI_SQUARED = np.pi ** 2
_PI_FOUR = np.pi ** 4
_PI_SIX = np.pi ** 6
_SQRT3 = np.sqrt(3)
_MIN_LOG = -708
_STIRLING_COEFFS = [-2.955065359477124183e-2, 6.4102564102564102564e-3,
                    -1.9175269175269175269e-3, 8.4175084175084175084e-4,
                    -5.952380952380952381e-4, 7.9365079365079365079e-4,
                    -2.7777777777777777778e-3, 8.3333333333333333333e-2]

cdef float clip(float a,float min_value, float max_value):
    return min(max(a, min_value), max_value)
    
cdef float _clip_prob(float p):
    """clips a probability to range 0<=p<=1."""
    return clip(p, 0.0, 1.0)

cdef float _select_and_clip_prob(float cdfprob, float sfprob, cdf=True):
    """Selects either the CDF or SF, and then clips to range 0<=p<=1."""
    p = cdfprob if cdf else sfprob
    return _clip_prob(p)

def _log_nfactorial_div_n_pow_n(int n):
    # Computes n! / n**n
    #    = (n-1)! / n**(n-1)
    # Uses Stirling's approximation, but removes n*log(n) up-front to
    # avoid subtractive cancellation.
    #    = log(n)/2 - n + log(sqrt(2pi)) + sum B_{2j}/(2j)/(2j-1)/n**(2j-1)
    rn = 1.0/n
    return np.log(n)/2 - n + _LOG_2PI/2 + rn * np.polyval(_STIRLING_COEFFS, rn/n)

def _kolmogn_DMTW(int n, float d, cdf=True):
    r"""Computes the Kolmogorov CDF:  Pr(D_n <= d) using the MTW approach to
    the Durbin matrix algorithm.

    Durbin (1968); Marsaglia, Tsang, Wang (2003). [1], [3].
    """
    # Write d = (k-h)/n, where k is positive integer and 0 <= h < 1
    # Generate initial matrix H of size m*m where m=(2k-1)
    # Compute k-th row of (n!/n^n) * H^n, scaling intermediate results.
    # Requires memory O(m^2) and computation O(m^2 log(n)).
    # Most suitable for small m.

    if d >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf)
    nd = n * d
    if nd <= 0.5:
        return _select_and_clip_prob(0.0, 1.0, cdf)
    k = int(np.ceil(nd))
    h = k - nd
    m = 2 * k - 1

    H = np.zeros([m, m])

    # Initialize: v is first column (and last row) of H
    #  v[j] = (1-h^(j+1)/(j+1)!  (except for v[-1])
    #  w[j] = 1/(j)!
    # q = k-th row of H (actually i!/n^i*H^i)
    intm = np.arange(1, m + 1)
    v = 1.0 - h ** intm
    w = np.empty(m)
    fac = 1.0
    for j in intm:
        w[j - 1] = fac
        fac /= j  # This might underflow.  Isn't a problem.
        v[j - 1] *= fac
    tt = max(2 * h - 1.0, 0)**m - 2*h**m
    v[-1] = (1.0 + tt) * fac

    for i in range(1, m):
        H[i - 1:, i] = w[:m - i + 1]
    H[:, 0] = v
    H[-1, :] = np.flip(v, axis=0)

    Hpwr = np.eye(np.shape(H)[0])  # Holds intermediate powers of H
    nn = n
    expnt = 0  # Scaling of Hpwr
    Hexpnt = 0  # Scaling of H
    while nn > 0:
        if nn % 2:
            Hpwr = np.matmul(Hpwr, H)
            expnt += Hexpnt
        H = np.matmul(H, H)
        Hexpnt *= 2
        # Scale as needed.
        if np.abs(H[k - 1, k - 1]) > _EP128:
            H /= _EP128
            Hexpnt += _E128
        nn = nn // 2

    p = Hpwr[k - 1, k - 1]

    # Multiply by n!/n^n
    for i in range(1, n + 1):
        p = i * p / n
        if np.abs(p) < _EM128:
            p *= _EP128
            expnt -= _E128

    # unscale
    if expnt != 0:
        p = np.ldexp(p, expnt)

    return _select_and_clip_prob(p, 1.0-p, cdf)

def _pomeranz_compute_j1j2(i, n, ll, ceilf, roundf):
    """Compute the endpoints of the interval for row i."""
    if i == 0:
        j1, j2 = -ll - ceilf - 1, ll + ceilf - 1
    else:
        # i + 1 = 2*ip1div2 + ip1mod2
        ip1div2, ip1mod2 = divmod(i + 1, 2)
        if ip1mod2 == 0:  # i is odd
            if ip1div2 == n + 1:
                j1, j2 = n - ll - ceilf - 1, n + ll + ceilf - 1
            else:
                j1, j2 = ip1div2 - 1 - ll - roundf - 1, ip1div2 + ll - 1 + ceilf - 1
        else:
            j1, j2 = ip1div2 - 1 - ll - 1, ip1div2 + ll + roundf - 1

    return max(j1 + 2, 0), min(j2, n)

def _kolmogn_Pomeranz(n, x, cdf=True):
    r"""Computes Pr(D_n <= d) using the Pomeranz recursion algorithm.

    Pomeranz (1974) [2]
    """

    # V is n*(2n+2) matrix.
    # Each row is convolution of the previous row and probabilities from a
    #  Poisson distribution.
    # Desired CDF probability is n! V[n-1, 2n+1]  (final entry in final row).
    # Only two rows are needed at any given stage:
    #  - Call them V0 and V1.
    #  - Swap each iteration
    # Only a few (contiguous) entries in each row can be non-zero.
    #  - Keep track of start and end (j1 and j2 below)
    #  - V0s and V1s track the start in the two rows
    # Scale intermediate results as needed.
    # Only a few different Poisson distributions can occur
    t = n * x
    ll = int(np.floor(t))
    f = 1.0 * (t - ll)  # fractional part of t
    g = min(f, 1.0 - f)
    ceilf = (1 if f > 0 else 0)
    roundf = (1 if f > 0.5 else 0)
    npwrs = 2 * (ll + 1)    # Maximum number of powers needed in convolutions
    gpower = np.empty(npwrs)  # gpower = (g/n)^m/m!
    twogpower = np.empty(npwrs)  # twogpower = (2g/n)^m/m!
    onem2gpower = np.empty(npwrs)  # onem2gpower = ((1-2g)/n)^m/m!
    # gpower etc are *almost* Poisson probs, just missing normalizing factor.

    gpower[0] = 1.0
    twogpower[0] = 1.0
    onem2gpower[0] = 1.0
    expnt = 0
    g_over_n, two_g_over_n, one_minus_two_g_over_n = g/n, 2*g/n, (1 - 2*g)/n
    for m in range(1, npwrs):
        gpower[m] = gpower[m - 1] * g_over_n / m
        twogpower[m] = twogpower[m - 1] * two_g_over_n / m
        onem2gpower[m] = onem2gpower[m - 1] * one_minus_two_g_over_n / m

    V0 = np.zeros([npwrs])
    V1 = np.zeros([npwrs])
    V1[0] = 1  # first row
    V0s, V1s = 0, 0  # start indices of the two rows

    j1, j2 = _pomeranz_compute_j1j2(0, n, ll, ceilf, roundf)
    for i in range(1, 2 * n + 2):
        # Preserve j1, V1, V1s, V0s from last iteration
        k1 = j1
        V0, V1 = V1, V0
        V0s, V1s = V1s, V0s
        V1.fill(0.0)
        j1, j2 = _pomeranz_compute_j1j2(i, n, ll, ceilf, roundf)
        if i == 1 or i == 2 * n + 1:
            pwrs = gpower
        else:
            pwrs = (twogpower if i % 2 else onem2gpower)
        ln2 = j2 - k1 + 1
        if ln2 > 0:
            conv = np.convolve(V0[k1 - V0s:k1 - V0s + ln2], pwrs[:ln2])
            conv_start = j1 - k1  # First index to use from conv
            conv_len = j2 - j1 + 1  # Number of entries to use from conv
            V1[:conv_len] = conv[conv_start:conv_start + conv_len]
            # Scale to avoid underflow.
            if 0 < np.max(V1) < _EM128:
                V1 *= _EP128
                expnt -= _E128
            V1s = V0s + j1 - k1

    # multiply by n!
    ans = V1[n - V1s]
    for m in range(1, n + 1):
        if np.abs(ans) > _EP128:
            ans *= _EM128
            expnt += _E128
        ans *= m

    # Undo any intermediate scaling
    if expnt != 0:
        ans = np.ldexp(ans, expnt)
    ans = _select_and_clip_prob(ans, 1.0 - ans, cdf)
    return ans

def _kolmogn_PelzGood(n, x, cdf=True):
    """Computes the Pelz-Good approximation to Prob(Dn <= x) with 0<=x<=1.

    Start with Li-Chien, Korolyuk approximation:
        Prob(Dn <= x) ~ K0(z) + K1(z)/sqrt(n) + K2(z)/n + K3(z)/n**1.5
    where z = x*sqrt(n).
    Transform each K_(z) using Jacobi theta functions into a form suitable
    for small z.
    Pelz-Good (1976). [6]
    """
    if x <= 0.0:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
    if x >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf=cdf)

    z = np.sqrt(n) * x
    zsquared, zthree, zfour, zsix = z**2, z**3, z**4, z**6

    qlog = -_PI_SQUARED / 8 / zsquared
    if qlog < _MIN_LOG:  # z ~ 0.041743441416853426
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)

    q = np.exp(qlog)

    # Coefficients of terms in the sums for K1, K2 and K3
    k1a = -zsquared
    k1b = _PI_SQUARED / 4

    k2a = 6 * zsix + 2 * zfour
    k2b = (2 * zfour - 5 * zsquared) * _PI_SQUARED / 4
    k2c = _PI_FOUR * (1 - 2 * zsquared) / 16

    k3d = _PI_SIX * (5 - 30 * zsquared) / 64
    k3c = _PI_FOUR * (-60 * zsquared + 212 * zfour) / 16
    k3b = _PI_SQUARED * (135 * zfour - 96 * zsix) / 4
    k3a = -30 * zsix - 90 * z**8

    K0to3 = np.zeros(4)
    # Use a Horner scheme to evaluate sum c_i q^(i^2)
    # Reduces to a sum over odd integers.
    maxk = int(np.ceil(16 * z / np.pi))
    for k in range(maxk, 0, -1):
        m = 2 * k - 1
        msquared, mfour, msix = m**2, m**4, m**6
        qpower = np.power(q, 8 * k)
        coeffs = np.array([1.0,
                           k1a + k1b*msquared,
                           k2a + k2b*msquared + k2c*mfour,
                           k3a + k3b*msquared + k3c*mfour + k3d*msix])
        K0to3 *= qpower
        K0to3 += coeffs
    K0to3 *= q
    K0to3 *= _SQRT2PI
    # z**10 > 0 as z > 0.04
    K0to3 /= np.array([z, 6 * zfour, 72 * z**7, 6480 * z**10])

    # Now do the other sum over the other terms, all integers k
    # K_2:  (pi^2 k^2) q^(k^2),
    # K_3:  (3pi^2 k^2 z^2 - pi^4 k^4)*q^(k^2)
    # Don't expect much subtractive cancellation so use direct calculation
    q = np.exp(-_PI_SQUARED / 2 / zsquared)
    ks = np.arange(maxk, 0, -1)
    ksquared = ks ** 2
    sqrt3z = _SQRT3 * z
    kspi = np.pi * ks
    qpwers = q ** ksquared
    k2extra = np.sum(ksquared * qpwers)
    k2extra *= _PI_SQUARED * _SQRT2PI/(-36 * zthree)
    K0to3[2] += k2extra
    k3extra = np.sum((sqrt3z + kspi) * (sqrt3z - kspi) * ksquared * qpwers)
    k3extra *= _PI_SQUARED * _SQRT2PI/(216 * zsix)
    K0to3[3] += k3extra
    powers_of_n = np.power(n * 1.0, np.arange(len(K0to3)) / 2.0)
    K0to3 /= powers_of_n

    if not cdf:
        K0to3 *= -1
        K0to3[0] += 1

    Ksum = sum(K0to3)
    return Ksum

def kolmogn(n, x, cdf=True):
    """Computes the CDF(or SF) for the two-sided Kolmogorov-Smirnov statistic.

    x must be of type float, n of type integer.

    Simard & L'Ecuyer (2011) [7].
    """
    if np.isnan(n):
        return n  # Keep the same type of nan
    if int(n) != n or n <= 0:
        return np.nan
    if x >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf=cdf)
    if x <= 0.0:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
    t = n * x
    if t <= 1.0:  # Ruben-Gambino: 1/2n <= x <= 1/n
        if t <= 0.5:
            return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
        if n <= 140:
            prob = np.prod(np.arange(1, n+1) * (1.0/n) * (2*t - 1))
        else:
            prob = np.exp(_log_nfactorial_div_n_pow_n(n) + n * np.log(2*t-1))
        return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
    if t >= n - 1:  # Ruben-Gambino
        prob = 2 * (1.0 - x)**n
        return _select_and_clip_prob(1 - prob, prob, cdf=cdf)
    if x >= 0.5:  # Exact: 2 * smirnov
        prob = 2 * smirnov(n, x)
        return _select_and_clip_prob(1.0 - prob, prob, cdf=cdf)

    nxsquared = t * x
    if n <= 140:
        if nxsquared <= 0.754693:
            prob = _kolmogn_DMTW(n, x, cdf=True)
            return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
        if nxsquared <= 4:
            prob = _kolmogn_Pomeranz(n, x, cdf=True)
            return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
        # Now use Miller approximation of 2*smirnov
        prob = 2 * smirnov(n, x)
        return _select_and_clip_prob(1.0 - prob, prob, cdf=cdf)

    # Split CDF and SF as they have different cutoffs on nxsquared.
    if not cdf:
        if nxsquared >= 370.0:
            return 0.0
        if nxsquared >= 2.2:
            prob = 2 * smirnov(n, x)
            return _clip_prob(prob)
        # Fall through and compute the SF as 1.0-CDF
    if nxsquared >= 18.0:
        cdfprob = 1.0
    elif n <= 100000 and n * x**1.5 <= 1.4:
        cdfprob = _kolmogn_DMTW(n, x, cdf=True)
    else:
        cdfprob = _kolmogn_PelzGood(n, x, cdf=True)
    return _select_and_clip_prob(cdfprob, 1.0 - cdfprob, cdf=cdf)

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
    minS = clip(-cddiffs[argminS], 0, 1)
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

    prob = clip(prob, 0, 1)
    return d, prob

def ks_drift(x_ref: np.ndarray, x: np.ndarray, float p_val = 0.05):
    x = x.reshape(x.shape[0], -1)
    x_ref = x_ref.reshape(x_ref.shape[0], -1)
    n_features = x_ref.reshape(x_ref.shape[0], -1).shape[-1]
    p_vals = np.zeros(n_features, dtype=float)
    dist = np.zeros_like(p_vals)
    for f in range(n_features):
        dist[f], p_vals[f] = ks_2samp(x_ref[:, f], x[:, f], alternative= 'two-sided')
    threshold = p_val / n_features
    drift_pred = int((p_vals < threshold).any()) 
    return drift_pred, p_vals