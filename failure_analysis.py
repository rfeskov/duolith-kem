"""
Exact decapsulation-failure analysis for DUOLITH-KEM.

Two complementary methods:

1.  FFT convolution of the exact per-term PMFs.  Exact in the bulk of the
    distribution (validated against Monte Carlo); numerically valid down to
    probabilities of roughly 1e-13 (float64 noise floor), which is enough to
    validate the model on scaled-down parameter sets where failure rates are
    ~1e-2, and to cross-check component distributions.

2.  Rigorous Chernoff upper bound over the exact moment generating function
    (MGF) of the total decryption error, evaluated in log domain.  The true
    failure event (a decoding error) is a SUBSET of {|err| >= threshold}, so

        P[failure per symbol] <= 2 * inf_{lambda>0} exp(-lambda*thr) * M(lambda)

    is a provable, conservative upper bound, computable to arbitrary depth
    (e.g. 2^-600) without any floating-point underflow problems.

Monte Carlo on scaled parameter sets validates the whole pipeline
(components, independence assumptions, decoder rule) end to end.
"""

import os
import sys

import numpy as np

import duolith_kem as dk

# ---------------------------------------------------------------------------
# PMF toolkit (bulk / validation)
# ---------------------------------------------------------------------------

class PMF:
    __slots__ = ("pmf", "lo")

    def __init__(self, pmf, lo):
        self.pmf = np.asarray(pmf, dtype=np.float64)
        self.lo = int(lo)

    @property
    def hi(self):
        return self.lo + len(self.pmf) - 1


def next_pow2(x):
    return 1 << int(x - 1).bit_length()


def conv_fft(a: PMF, b: PMF) -> PMF:
    n = len(a.pmf) + len(b.pmf) - 1
    size = next_pow2(n)
    out = np.fft.irfft(np.fft.rfft(a.pmf, size) * np.fft.rfft(b.pmf, size), size)[:n]
    return PMF(np.clip(out, 0.0, None), a.lo + b.lo)


def conv_pow(a: PMF, k: int) -> PMF:
    n = (len(a.pmf) - 1) * k + 1
    size = next_pow2(n)
    out = np.fft.irfft(np.fft.rfft(a.pmf, size) ** k, size)[:n]
    return PMF(np.clip(out, 0.0, None), a.lo * k)


def conv_all(pmf_list) -> PMF:
    n = sum(len(p.pmf) - 1 for p in pmf_list) + 1
    size = next_pow2(n)
    acc = np.ones(size // 2 + 1, dtype=np.complex128)
    lo = 0
    for p in pmf_list:
        acc *= np.fft.rfft(p.pmf, size)
        lo += p.lo
    out = np.fft.irfft(acc, size)[:n]
    return PMF(np.clip(out, 0.0, None), lo)


def cbd_pmf(eta) -> PMF:
    base = np.array([0.25, 0.5, 0.25])
    pmf = np.array([1.0])
    for _ in range(eta):
        pmf = np.convolve(pmf, base)
    return PMF(pmf, -eta)


def product_pmf(a: PMF, b: PMF) -> PMF:
    corners = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi)
    lo, hi = min(corners), max(corners)
    out = np.zeros(hi - lo + 1, dtype=np.float64)
    for i, pa in enumerate(a.pmf):
        if pa == 0.0:
            continue
        xa = a.lo + i
        idx = xa * (b.lo + np.arange(len(b.pmf))) - lo
        out += pa * np.bincount(idx, weights=b.pmf, minlength=len(out))
    return PMF(out, lo)


def compression_err_pmf(q, d) -> PMF:
    xs = np.arange(q, dtype=np.int64)
    errs = (dk.decompress(dk.compress(xs, q, d), q, d) - xs) % q
    errs = np.where(errs > q // 2, errs - q, errs)
    half = q >> d
    lo, hi = -half - 1, half + 1
    hist = np.bincount(errs - lo, minlength=hi - lo + 1).astype(np.float64)
    return PMF(hist / q, lo)


def mgf_log(p: PMF, lam):
    """log E[exp(lam * X)] for PMF p (numpy-vectorised, log domain)."""
    x = np.arange(p.lo, p.lo + len(p.pmf))
    m = np.max(p.pmf * x * lam) if False else 0.0
    lg = np.log(np.maximum(p.pmf, 1e-300)) + lam * x
    m = lg.max()
    return m + np.log(np.exp(lg - m).sum())


def chernoff_two_sided(mgf_terms, thr):
    """Rigorous bound: P[|err| >= thr] <= 2 * inf_lam exp(-lam*thr)*M(lam).

    mgf_terms: list of (log-MGF function, multiplicity) pairs; the total MGF
    is the product.
    """
    def total_log_mgf(lam):
        return sum(k * f(lam) for f, k in mgf_terms)

    def one_sided(sign):
        def obj(lam):
            return -lam * thr + total_log_mgf(sign * lam)
        # golden-section search over lam in (1e-9, 30)
        a, b = 1e-9, 30.0
        gr = (np.sqrt(5) - 1) / 2
        c, d = b - gr * (b - a), a + gr * (b - a)
        for _ in range(200):
            if obj(c) < obj(d):
                b = d
            else:
                a = c
            c, d = b - gr * (b - a), a + gr * (b - a)
        lam = 0.5 * (a + b)
        return obj(lam), lam

    lp, lam_p = one_sided(+1)
    ln, lam_n = one_sided(-1)
    # P[|err|>=thr] = P[err>=thr] + P[err<=-thr]; each bounded by its Chernoff bound
    log_p = max(lp, ln) + np.log(np.exp(min(lp - max(lp, ln), 0))
                                 + np.exp(min(ln - max(lp, ln), 0)))
    return np.exp(log_p) / np.log(2) * -1.0, lam_p, lam_n   # returns log2 magnitude


def chernoff_log2(mgf_terms, thr):
    """Return log2 of the rigorous two-sided Chernoff bound.

    mgf_terms: list of (callable lam -> log M(lam), multiplicity).
    """
    def total_log_mgf(lam):
        return sum(k * f(lam) for f, k in mgf_terms)

    def one_sided(sign):
        def obj(lam):
            return -lam * thr + total_log_mgf(sign * lam)
        a, b = 1e-9, 40.0
        gr = (np.sqrt(5) - 1) / 2
        c, d = b - gr * (b - a), a + gr * (b - a)
        for _ in range(200):
            if obj(c) < obj(d):
                b = d
            else:
                a = c
            c, d = b - gr * (b - a), a + gr * (b - a)
        return obj(0.5 * (a + b))

    lp = one_sided(+1)
    ln = one_sided(-1)
    m = max(lp, ln)
    return (m + np.log(np.exp(lp - m) + np.exp(ln - m))) / np.log(2)


def chernoff_log2_robust(prod_mgf, sdu_mgf, dpks_mgf, n_mult, thr, w_u, w_pk, w_v, cbd_mgf):
    """Assumption-free variant: compression errors are replaced by their
    worst-case bound |err| <= w/2 (M(worst) = exp(lam*w/2)), valid for ANY
    distribution supported on that interval.  product mgfs remain exact."""
    def total_log_mgf(lam):
        return (n_mult[0] * prod_mgf(lam) + n_mult[1] * sdu_mgf(lam)
                + n_mult[2] * dpks_mgf(lam) + cbd_mgf(lam)
                + lam * (w_u + w_v + w_pk) / 2)

    def one_sided(sign):
        def obj(lam):
            return -lam * thr + total_log_mgf(sign * lam)
        a, b = 1e-9, 40.0
        gr = (np.sqrt(5) - 1) / 2
        c, d = b - gr * (b - a), a + gr * (b - a)
        for _ in range(200):
            if obj(c) < obj(d):
                b = d
            else:
                a = c
            c, d = b - gr * (b - a), a + gr * (b - a)
        return obj(0.5 * (a + b))

    lp = one_sided(+1)
    ln = one_sided(-1)
    m = max(lp, ln)
    return (m + np.log(np.exp(lp - m) + np.exp(ln - m))) / np.log(2)


# ---------------------------------------------------------------------------
# Wall analyses
# ---------------------------------------------------------------------------

def wall_s_terms(p):
    """MGF terms and bulk PMF terms of the Wall-S per-coefficient error."""
    n, k, q, eta = p["nS"], p["kS"], p["qS"], p["etaS"]
    cbd = cbd_pmf(eta)
    prod = product_pmf(cbd, cbd)
    du = compression_err_pmf(q, p["duS"])
    dv = compression_err_pmf(q, p["dvS"])
    sdu = product_pmf(cbd, du)
    mgf = [(lambda pp: (lambda lam: mgf_log(pp, lam)))(prod, ), ]  # placeholder
    return dict(cbd=cbd, prod=prod, du=du, dv=dv, sdu=sdu, n=n, k=k, q=q)


def wall_s_chernoff(p):
    t = wall_s_terms(p)
    n, k = t["n"], t["k"]
    q = t["q"]
    h = (q + 1) // 2
    thr = (h + 1) // 2
    prod_f = lambda lam: mgf_log(t["prod"], lam)
    sdu_f = lambda lam: mgf_log(t["sdu"], lam)
    dv_f = lambda lam: mgf_log(t["dv"], lam)
    cbd_f = lambda lam: mgf_log(t["cbd"], lam)
    terms = [
        (prod_f, k * n),   # e^T r
        (prod_f, k * n),   # s^T e1
        (sdu_f, k * n),    # s^T du
        (dv_f, 1),         # dv
        (cbd_f, 1),        # e2
    ]
    log2_pc = chernoff_log2(terms, thr)
    # assumption-free variant: compression errors worst-case bounded
    w_u = q / float(1 << p["duS"]) / 2 + 1
    w_v = q / float(1 << p["dvS"]) / 2 + 1
    log2_pc_rob = chernoff_log2_robust(prod_f, sdu_f, sdu_f, (k * n, k * n, 0),
                                       thr, w_u, 0.0, w_v, cbd_f)
    return dict(log2_per_symbol=log2_pc, log2_total=log2_pc + np.log2(n),
                log2_per_symbol_rob=log2_pc_rob,
                log2_total_rob=log2_pc_rob + np.log2(n), thr=thr)


def wall_u_terms(p):
    n, t, q, eta = p["nU"], p["tU"], p["qU"], p["etaU"]
    cbd = cbd_pmf(eta)
    prod = product_pmf(cbd, cbd)
    du = compression_err_pmf(q, p["duU"])
    dv = compression_err_pmf(q, p["dvU"])
    dpk = compression_err_pmf(q, p["dtU"])
    sdu = product_pmf(cbd, du)
    dpks = product_pmf(dpk, cbd)
    return dict(cbd=cbd, prod=prod, du=du, dv=dv, dpk=dpk, sdu=sdu, dpks=dpks,
                n=n, t=t, q=q)


def wall_u_chernoff(p):
    t = wall_u_terms(p)
    n = t["n"]
    q = t["q"]
    thr = q >> (p["bU"] + 1)
    prod_f = lambda lam: mgf_log(t["prod"], lam)
    sdu_f = lambda lam: mgf_log(t["sdu"], lam)
    dpks_f = lambda lam: mgf_log(t["dpks"], lam)
    cbd_f = lambda lam: mgf_log(t["cbd"], lam)
    dv_f = lambda lam: mgf_log(t["dv"], lam)
    terms = [
        (prod_f, 2 * n),   # E^T S' + S^T E'
        (sdu_f, n),        # S^T du
        (dpks_f, n),       # dpk^T S'
        (cbd_f, 1),        # E''
        (dv_f, 1),         # dv
    ]
    log2_pe = chernoff_log2(terms, thr)
    w_u = q / float(1 << p["duU"]) / 2 + 1
    w_v = q / float(1 << p["dvU"]) / 2 + 1
    w_pk = q / float(1 << p["dtU"]) / 2 + 1
    log2_pe_rob = chernoff_log2_robust(prod_f, sdu_f, dpks_f, (2 * n, n, n),
                                       thr, w_u, w_pk, w_v, cbd_f)
    return dict(log2_per_symbol=log2_pe,
                log2_total=log2_pe + np.log2(t["t"] * t["t"]), thr=thr,
                log2_per_symbol_rob=log2_pe_rob,
                log2_total_rob=log2_pe_rob + np.log2(t["t"] * t["t"]))


def wall_s_bulk(p):
    """FFT bulk distribution (for validation plots / scaled sets)."""
    t = wall_s_terms(p)
    n, k, q = t["n"], t["k"], t["q"]
    total = conv_all([conv_pow(t["prod"], k * n), conv_pow(t["prod"], k * n),
                      conv_pow(t["sdu"], k * n), t["dv"], t["cbd"]])
    dist = np.bincount(np.arange(total.lo, total.lo + len(total.pmf)) % q,
                       weights=total.pmf, minlength=q)
    h = (q + 1) // 2
    thr = (h + 1) // 2
    x = np.arange(q)
    xc = np.where(x <= q // 2, x, x - q)
    dec1 = np.abs(xc) >= thr
    p0 = dist[dec1].sum()
    p1 = np.roll(dist, h)[~dec1].sum()
    return 0.5 * (p0 + p1)


def wall_u_bulk(p):
    t = wall_u_terms(p)
    n, tU, q = t["n"], t["t"], t["q"]
    total = conv_all([conv_pow(t["prod"], n), conv_pow(t["prod"], n),
                      conv_pow(t["sdu"], n), conv_pow(t["dpks"], n),
                      t["cbd"], t["dv"]])
    dist = np.bincount(np.arange(total.lo, total.lo + len(total.pmf)) % q,
                       weights=total.pmf, minlength=q)
    step = q >> p["bU"]
    x = np.arange(q)
    chunk = ((x * (1 << p["bU"]) + q // 2) // q) % (1 << p["bU"])
    pw = [np.roll(dist, c * step)[chunk != c].sum() for c in range(1 << p["bU"])]
    return float(np.mean(pw))


# ---------------------------------------------------------------------------
# Monte Carlo validation on scaled parameter sets
# ---------------------------------------------------------------------------

def mc_wall_s(name, trials):
    p = dk._P(name)
    mlen = p["nS"] // 8
    fails = 0
    obs = 0
    for _ in range(trials):
        pkS, s = dk._wall_s_keypair_pke(os.urandom(32), os.urandom(32), p)
        m = os.urandom(mlen)
        ctS = dk._wall_s_encrypt_pke(pkS, m, os.urandom(32), p)
        m2 = dk._wall_s_decrypt_pke(pkS, s, ctS, p)
        a = np.unpackbits(np.frombuffer(m, np.uint8), bitorder="little")
        b = np.unpackbits(np.frombuffer(m2, np.uint8), bitorder="little")
        fails += int((a.astype(int) ^ b.astype(int)).sum())
        obs += 8 * mlen
    return fails / obs, fails, obs


def mc_wall_u(name, trials):
    p = dk._P(name)
    mlen = p["tU"] * p["tU"] * p["bU"] // 8
    fails = 0
    obs = 0
    for _ in range(trials):
        pkU, S = dk._wall_u_keypair_pke(os.urandom(32), os.urandom(32), p)
        m = os.urandom(mlen)
        ctU = dk._wall_u_encrypt_pke(pkU, m, os.urandom(32), p)
        m2 = dk._wall_u_decrypt_pke(S, ctU, p)
        a = np.unpackbits(np.frombuffer(m, np.uint8), bitorder="little").reshape(-1, p["bU"])
        b = np.unpackbits(np.frombuffer(m2, np.uint8), bitorder="little").reshape(-1, p["bU"])
        fails += int((a != b).any(axis=1).sum())
        obs += a.shape[0]
    return fails / obs, fails, obs


# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("DUOLITH-KEM decapsulation-failure analysis")
    print("=" * 78)
    for name in ("DUOLITH-128", "DUOLITH-192"):
        p = dk._P(name)
        ws = wall_s_chernoff(p)
        wu = wall_u_chernoff(p)
        print("\n[%s]" % name)
        print("  Wall-S (k=%d, q=%d):  Chernoff bound per coefficient <= 2^%.1f"
              % (p["kS"], p["qS"], ws["log2_per_symbol"]))
        print("                          union bound over %d coeff  <= 2^%.1f"
              % (p["nS"], ws["log2_total"]))
        print("        assumption-free worst-case-compression bound <= 2^%.1f (union)"
              % ws["log2_total_rob"])
        print("  Wall-U (n=%d, q=2^%.0f): Chernoff bound per entry      <= 2^%.1f"
              % (p["nU"], np.log2(p["qU"]), wu["log2_per_symbol"]))
        print("                          union bound over %d entries  <= 2^%.1f"
              % (p["tU"] ** 2, wu["log2_total"]))
        print("        assumption-free worst-case-compression bound <= 2^%.1f (union)"
              % wu["log2_total_rob"])
        worst = max(ws["log2_total"], wu["log2_total"])
        worst_rob = max(ws["log2_total_rob"], wu["log2_total_rob"])
        print("  combined decapsulation failure <= 2^%.1f   (robust: <= 2^%.1f)"
              % (worst, worst_rob))

    print("\nScaled-set validation (FFT bulk vs Monte Carlo):")
    for name, wall, trials in (("SCALED-S", "S", 4000), ("SCALED-U", "U", 3000)):
        p = dk._P(name)
        if wall == "S":
            fft_pred = wall_s_bulk(p)
            meas, f, obs = mc_wall_s(name, trials)
            unit = "coefficient"
        else:
            fft_pred = wall_u_bulk(p)
            meas, f, obs = mc_wall_u(name, trials)
            unit = "entry"
        se = (meas * (1 - meas) / obs) ** 0.5
        z = abs(meas - fft_pred) / se if se > 0 else 0
        print("  %-9s FFT %.4e | MC %.4e +/- %.2e (%d/%d)  z=%.1f  %s"
              % (name, fft_pred, meas, se, f, obs, z,
                 "MATCH" if z < 4 else "MISMATCH"))

    print("\nScaled-set Chernoff bounds (conservative; should sit above MC rates):")
    for name in ("SCALED-S", "SCALED-U"):
        p = dk._P(name)
        ws = wall_s_chernoff(p)
        wu = wall_u_chernoff(p)
        print("  %-9s Wall-S bound 2^%.2f per coeff, Wall-U bound 2^%.2f per entry"
              % (name, ws["log2_per_symbol"], wu["log2_per_symbol"]))


if __name__ == "__main__":
    main()
