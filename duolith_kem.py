"""
DUOLITH-KEM v1.0 -- reference implementation
=============================================

DUOLITH-KEM is a dual-layer ("two-wall") post-quantum key encapsulation
mechanism (KEM).  Its two independent computational walls are:

  Wall-S (structured):   Module-LWE over R = Z_q[X]/(X^256 + 1),
                         q = 7681 (prime, 512 | q-1, NTT-friendly),
                         module rank k in {2, 3}, centered binomial noise CBD(3).
                         This is the same ring family and the same modulus
                         that Kyber used in round 1 of the NIST process.

  Wall-U (unstructured): plain (unstructured) Learning With Errors over
                         Z_{2^16} in FrodoKEM-style amortized matrix form,
                         secret dimension n in {640, 768}, CBD(8) noise
                         (sigma = 2), with the matrix A expanded from a seed.

  Encapsulation picks two independent uniform secrets m_S (256 bits) and
  m_U (256 bits), encrypts m_S under Wall-S and m_U under Wall-U, and derives

      K = SHA3-256("DUOLITH/KDF/v1" || m_S || m_U || H(pk) || H(ct)).

  Because K is the hash of a concatenation, K remains indistinguishable from
  random as long as AT LEAST ONE of m_S, m_U stays hidden, i.e. as long as at
  least one wall holds (a security OR-combiner; the two walls are algebraically
  heterogeneous on purpose: a hypothetical structural break of cyclotomic
  Module-LWE does not transfer to unstructured LWE, and vice versa).

  IND-CCA2 security is obtained with the implicit-rejection Fujisaki-Okamoto
  transform of Hofheinz-Hoevelmanns-Kiltz (eprint 2017/604): decapsulation
  re-encrypts the recovered messages with randomness derived from the
  recovered message and H(ct); on any mismatch a pseudorandom rejection key
  K = SHA3-256("DUOLITH/REJECT/v1" || z || H(ct)) is returned.

SERIALIZATION (all integers little-endian, bit-packed LSB-first):
  pk = pkS || pkU
       pkS = seedS(32) || t_raw   (k*256 coefficients, 13 bits each, raw)
       pkU = seedU(32) || B_15    (n*8 entries, 15-bit compressed)
  ct = ctS || ctU
       ctS = u_11 (k*256 coefficients, 11-bit compressed)
             || v_5  (256 coefficients, 5-bit compressed)
       ctU = U_13 (n*8 entries, 13-bit compressed) || C_12 (8*8 entries, 12-bit)
  sk = seedS(32) || skSeedS(32) || seedU(32) || skSeedU(32) || z(32)  (160 bytes)

The full message m_S (32 bytes) is encoded one bit per coefficient of v
(256 ring coefficients); the message m_U (32 bytes) is encoded as 64 chunks of
4 bits each, one chunk per entry of the 8x8 matrix M, at symbol spacing
q_U / 16, which is injective over Z_{2^16}.

------------------------------------------------------------------------------
THIS IS A REFERENCE IMPLEMENTATION, NOT A PRODUCTION LIBRARY:
  * it is NOT constant-time and NOT hardened against side channels;
  * DUOLITH-KEM is a new construction and has NOT received public cryptanalysis;
  * use it to study, test and review the design -- not to protect real secrets.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional, Tuple

import numpy as np

__version__ = "1.0"

# ----------------------------------------------------------------------------
# Parameter sets
# ----------------------------------------------------------------------------

PARAMETER_SETS: Dict[str, dict] = {
    "DUOLITH-128": dict(
        name="DUOLITH-128",
        # Wall-S (structured, Module-LWE)
        nS=256, kS=2, qS=7681, etaS=3, duS=12, dvS=6, dtS=13,
        # Wall-U (unstructured, plain LWE)
        nU=640, tU=8, qU=65536, etaU=8, duU=14, dvU=13, dtU=15, bU=4,
    ),
    "DUOLITH-192": dict(
        name="DUOLITH-192",
        # Wall-S (structured, Module-LWE)
        nS=256, kS=3, qS=7681, etaS=3, duS=12, dvS=6, dtS=13,
        # Wall-U (unstructured, plain LWE)
        nU=768, tU=8, qU=65536, etaU=8, duU=14, dvU=13, dtU=15, bU=4,
    ),
}

# Scaled-down parameter sets used ONLY by the test/analysis suite to validate
# the failure-probability machinery empirically.  NOT secure, do not use.
SCALED_SETS: Dict[str, dict] = {
    "SCALED-S": dict(
        name="SCALED-S",
        nS=16, kS=1, qS=769, etaS=3, duS=4, dvS=3, dtS=10,
        nU=96, tU=8, qU=4096, etaU=2, duU=7, dvU=8, dtU=11, bU=4,
    ),
    "SCALED-U": dict(
        name="SCALED-U",
        nS=16, kS=1, qS=769, etaS=3, duS=4, dvS=3, dtS=10,
        nU=96, tU=8, qU=4096, etaU=2, duU=7, dvU=8, dtU=11, bU=4,
    ),
}


def _P(name: str) -> dict:
    if name in PARAMETER_SETS:
        return PARAMETER_SETS[name]
    if name in SCALED_SETS:
        return SCALED_SETS[name]
    raise ValueError("unknown parameter set: %r" % (name,))


# ----------------------------------------------------------------------------
# Domain separation tags (all ASCII, versioned)
# ----------------------------------------------------------------------------

DOM_EXPAND_S = b"DUOLITH/EXPAND-S/v1/"   # Wall-S matrix from seed
DOM_EXPAND_U = b"DUOLITH/EXPAND-U/v1/"   # Wall-U matrix rows from seed
DOM_NOISE_S = b"DUOLITH/NOISE-S/v1/"     # Wall-S keygen noise
DOM_NOISE_U = b"DUOLITH/NOISE-U/v1/"     # Wall-U keygen noise
DOM_ENC_S = b"DUOLITH/ENC-S/v1/"         # Wall-S encryption randomness
DOM_ENC_U = b"DUOLITH/ENC-U/v1/"         # Wall-U encryption randomness
DOM_RAND_S = b"DUOLITH/RAND-S/v1/"       # FO re-encryption randomness (Wall-S)
DOM_RAND_U = b"DUOLITH/RAND-U/v1/"       # FO re-encryption randomness (Wall-U)
DOM_HASH = b"DUOLITH/H/v1/"              # H(pk), H(ct)
DOM_KDF = b"DUOLITH/KDF/v1/"             # shared-secret derivation
DOM_REJECT = b"DUOLITH/REJECT/v1/"       # implicit rejection key

SEED_LEN = 32
KEYGEN_SEED_LEN = 5 * SEED_LEN   # seedS, skSeedS, seedU, skSeedU, z
ENCAP_COINS_LEN = 2 * SEED_LEN   # mS, mU (encryption randomness is derived
                                 # from the messages and H(pk), Kyber/HHK-style)


# ----------------------------------------------------------------------------
# Symmetric primitives and simple encodings
# ----------------------------------------------------------------------------

def _sha3(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def _shake256(data: bytes, n: int) -> bytes:
    return hashlib.shake_256(data).digest(n)


def _shake128(data: bytes, n: int) -> bytes:
    return hashlib.shake_128(data).digest(n)


def bitpack(values: np.ndarray, bits: int) -> bytes:
    """Pack non-negative integers (< 2**bits) LSB-first into bytes."""
    v = np.asarray(values, dtype=np.int64)
    bits_mat = ((v[:, None] >> np.arange(bits, dtype=np.int64)[None, :]) & 1)
    return np.packbits(bits_mat.astype(np.uint8).reshape(-1),
                       bitorder="little").tobytes()


def bitunpack(buf: bytes, bits: int, count: int) -> np.ndarray:
    """Inverse of bitpack (must be byte-aligned: count*bits % 8 == 0)."""
    need = count * bits
    assert need % 8 == 0 and len(buf) == need // 8, "bit field not byte-aligned"
    bits_arr = np.unpackbits(np.frombuffer(buf, dtype=np.uint8),
                             bitorder="little", count=need)
    vals = bits_arr.reshape(count, bits).astype(np.int64) * \
        (np.int64(1) << np.arange(bits, dtype=np.int64))
    return vals.sum(axis=1)


def compress(x: np.ndarray, q: int, d: int) -> np.ndarray:
    """Compress x mod q to d bits: floor((x * 2^d + q/2) / q) mod 2^d."""
    return (((np.asarray(x, dtype=np.int64) * (1 << d)) + q // 2) // q) % (1 << d)


def decompress(y: np.ndarray, q: int, d: int) -> np.ndarray:
    """Decompress d-bit y to Z_q: floor((y * q + 2^(d-1)) / 2^d) mod q."""
    return (((np.asarray(y, dtype=np.int64) * q) + (1 << (d - 1))) // (1 << d)) % q


def cbd_from_stream(stream: bytes, eta: int, count: int) -> np.ndarray:
    """Centered binomial distribution CBD_eta from an LSB-first bit stream.

    Each coefficient consumes 2*eta bits (b_0..b_{2eta-1}, LSB-first) and is
    sum(b_{2i} - b_{2i+1}).  Result in [-eta, eta], dtype int64.
    """
    need = 2 * eta * count
    bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8),
                         bitorder="little", count=need).astype(np.int64)
    pairs = bits.reshape(count, 2 * eta)
    return pairs[:, 0::2].sum(axis=1) - pairs[:, 1::2].sum(axis=1)


def _prf_noise(seed: bytes, dom: bytes, nonce: int, nbytes: int) -> bytes:
    return _shake256(dom + seed + bytes([nonce]), nbytes)


# ----------------------------------------------------------------------------
# Wall-S: Module-LWE over Z_q[X]/(X^256 + 1)
# ----------------------------------------------------------------------------

def _poly_uniform(seed: bytes, i: int, j: int, n: int, q: int) -> np.ndarray:
    """Uniform polynomial in Z_q[X]/(X^n+1) via SHAKE-128 rejection sampling
    of 13-bit chunks (q = 7681 < 2^13)."""
    data = _shake128(DOM_EXPAND_S + seed + bytes([i, j]), 4096)
    chunks = np.unpackbits(np.frombuffer(data, dtype=np.uint8),
                           bitorder="little", count=len(data) * 8)
    nchunks = len(chunks) // 13
    vals = chunks[: nchunks * 13].reshape(nchunks, 13).astype(np.int64) * \
        (np.int64(1) << np.arange(13, dtype=np.int64))
    vals = vals.sum(axis=1)
    vals = vals[vals < q]
    assert len(vals) >= n, "uniform sampling stream exhausted"
    return vals[:n]


def _poly_mul(a: np.ndarray, b: np.ndarray, n: int, q: int) -> np.ndarray:
    """Negacyclic polynomial multiplication modulo (X^n + 1, q)."""
    c = np.convolve(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
    head = c[:n].copy()
    tail = c[n:2 * n - 1]
    head[: len(tail)] -= tail          # X^n = -1
    return head % q


def _poly_add(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    return (a + b) % q


def _wall_s_expand_matrix(seedS: bytes, p: dict) -> list:
    """k x k matrix of polynomials from seedS."""
    k, n, q = p["kS"], p["nS"], p["qS"]
    return [[_poly_uniform(seedS, i, j, n, q) for j in range(k)]
            for i in range(k)]


def _wall_s_secret(skSeedS: bytes, p: dict, offset: int = 0) -> list:
    """k secret/noise polynomials from skSeedS with nonce offset."""
    k, n, eta = p["kS"], p["nS"], p["etaS"]
    nbytes = 2 * eta * n // 8
    out = []
    for idx in range(k):
        stream = _prf_noise(skSeedS, DOM_NOISE_S, offset + idx, nbytes)
        out.append(cbd_from_stream(stream, eta, n))
    return out


def _wall_s_keypair_pke(seedS: bytes, skSeedS: bytes, p: dict):
    """Wall-S PKE keypair.  Returns (pkS bytes, s polynomials)."""
    k, n, q, dt = p["kS"], p["nS"], p["qS"], p["dtS"]
    A = _wall_s_expand_matrix(seedS, p)
    s = _wall_s_secret(skSeedS, p, offset=0)
    e = _wall_s_secret(skSeedS, p, offset=k)
    t = []
    for i in range(k):
        acc = np.zeros(n, dtype=np.int64)
        for j in range(k):
            acc = _poly_add(acc, _poly_mul(A[i][j], s[j], n, q), q)
        t.append(_poly_add(acc, e[i], q))
    pkS = seedS + b"".join(bitpack(ti, dt) for ti in t)   # raw 13-bit storage
    return pkS, s


def _wall_s_encrypt_pke(pkS: bytes, m: bytes, coins: bytes, p: dict):
    """Encrypt 32-byte message m under Wall-S PKE with 32-byte coins."""
    k, n, q, eta = p["kS"], p["nS"], p["qS"], p["etaS"]
    du, dv, dt = p["duS"], p["dvS"], p["dtS"]
    seedS = pkS[:SEED_LEN]
    t = [bitunpack(pkS[SEED_LEN + idx * n * dt // 8:
                       SEED_LEN + (idx + 1) * n * dt // 8],
                   dt, n) for idx in range(k)]
    A = _wall_s_expand_matrix(seedS, p)
    nb = 2 * eta * n // 8
    r = [cbd_from_stream(_prf_noise(coins, DOM_ENC_S, idx, nb), eta, n)
         for idx in range(k)]
    e1 = [cbd_from_stream(_prf_noise(coins, DOM_ENC_S, k + idx, nb), eta, n)
          for idx in range(k)]
    e2 = cbd_from_stream(_prf_noise(coins, DOM_ENC_S, 2 * k, nb), eta, n)
    # u = A^T r + e1
    u = []
    for j in range(k):
        acc = np.zeros(n, dtype=np.int64)
        for i in range(k):
            acc = _poly_add(acc, _poly_mul(A[i][j], r[i], n, q), q)
        u.append(_poly_add(acc, e1[j], q))
    # v = t^T r + e2 + Encode(m): one bit of m per coefficient, symbol h = (q+1)/2
    h = (q + 1) // 2
    bits = np.unpackbits(np.frombuffer(m, dtype=np.uint8), bitorder="little",
                         count=8 * len(m)).astype(np.int64)
    acc = np.zeros(n, dtype=np.int64)
    for i in range(k):
        acc = _poly_add(acc, _poly_mul(t[i], r[i], n, q), q)
    v = _poly_add(_poly_add(acc, e2, q), (bits * h) % q, q)
    ctS = b"".join(bitpack(compress(ui, q, du), du) for ui in u) + \
        bitpack(compress(v, q, dv), dv)
    return ctS


def _wall_s_decrypt_pke(pkS: bytes, s: list, ctS: bytes, p: dict) -> bytes:
    """Decrypt Wall-S PKE ciphertext back to the 32-byte message."""
    k, n, q = p["kS"], p["nS"], p["qS"]
    du, dv = p["duS"], p["dvS"]
    off = 0
    u = []
    for _ in range(k):
        raw = bitunpack(ctS[off: off + n * du // 8], du, n)
        u.append(decompress(raw, q, du))
        off += n * du // 8
    v = decompress(bitunpack(ctS[off: off + n * dv // 8], dv, n), q, dv)
    acc = np.zeros(n, dtype=np.int64)
    for j in range(k):
        acc = _poly_add(acc, _poly_mul(s[j], u[j], n, q), q)
    w = (v - acc) % q
    h = (q + 1) // 2
    thr = (h + 1) // 2
    # centered representative
    xc = np.where(w <= q // 2, w, w - q)
    bits = (np.abs(xc) >= thr).astype(np.uint8)
    return np.packbits(bits, bitorder="little").tobytes()


# ----------------------------------------------------------------------------
# Wall-U: plain (unstructured) LWE over Z_{2^16}, amortized matrix form
# ----------------------------------------------------------------------------

def _wall_u_expand_matrix(seedU: bytes, p: dict) -> np.ndarray:
    """n x n uniform matrix over Z_{qU} (qU = 2^16, exact power of two: the
    raw 16-bit stream is already uniform, no rejection or bias)."""
    n, q = p["nU"], p["qU"]
    assert q & (q - 1) == 0, "Wall-U modulus must be a power of two"
    A = np.empty((n, n), dtype=np.int64)
    for i in range(n):
        row = _shake128(DOM_EXPAND_U + seedU + i.to_bytes(2, "little"), 2 * n)
        A[i] = np.frombuffer(row, dtype="<u2").astype(np.int64)
    return A


def _wall_u_sample_columns(skSeedU: bytes, p: dict, offset: int) -> np.ndarray:
    """t columns of CBD_eta noise vectors (n x t matrix)."""
    n, t, eta = p["nU"], p["tU"], p["etaU"]
    nbytes = 2 * eta * n // 8
    out = np.empty((n, t), dtype=np.int64)
    for c in range(t):
        out[:, c] = cbd_from_stream(
            _prf_noise(skSeedU, DOM_NOISE_U, offset + c, nbytes), eta, n)
    return out


def _wall_u_keypair_pke(seedU: bytes, skSeedU: bytes, p: dict):
    """Wall-U PKE keypair: pkU = seedU || B_15, returns (pkU, S)."""
    n, t, q, dt = p["nU"], p["tU"], p["qU"], p["dtU"]
    A = _wall_u_expand_matrix(seedU, p)
    S = _wall_u_sample_columns(skSeedU, p, offset=0)
    E = _wall_u_sample_columns(skSeedU, p, offset=t)
    B = (A @ S + E) % q
    pkU = seedU + bitpack(compress(B.reshape(-1), q, dt), dt)
    return pkU, S


def _encode_mU(m: bytes, p: dict) -> np.ndarray:
    """Encode 32-byte m_U into the t x t message matrix.

    Entry (j, c) carries the 4-bit chunk g = 8*j + c of the LSB-first bit
    string of m, valued chunk * (qU / 2^bU).  Injective: qU is a power of two
    and chunk < 2^bU, so the 64 symbols are pairwise distinct mod qU.
    """
    t, q, b = p["tU"], p["qU"], p["bU"]
    assert t * t * b == 8 * len(m), "message length mismatch for Wall-U encoding"
    bits = np.unpackbits(np.frombuffer(m, dtype=np.uint8), bitorder="little",
                         count=8 * len(m)).astype(np.int64)
    chunks = bits.reshape(t * t, b) * (np.int64(1) << np.arange(b, dtype=np.int64))
    chunks = chunks.sum(axis=1)
    return (chunks.reshape(t, t) * (q >> b))


def _decode_mU(Mdec: np.ndarray, p: dict) -> bytes:
    """Inverse of _encode_mU: decode the t x t decrypted matrix to 32 bytes."""
    t, q, b = p["tU"], p["qU"], p["bU"]
    x = Mdec.reshape(-1).astype(np.int64)
    chunk = ((x * (1 << b) + q // 2) // q) % (1 << b)     # nearest-symbol decode
    bits = ((chunk[:, None] >> np.arange(b, dtype=np.int64)[None, :]) & 1)
    return np.packbits(bits.astype(np.uint8).reshape(-1),
                       bitorder="little").tobytes()


def _wall_u_encrypt_pke(pkU: bytes, m: bytes, coins: bytes, p: dict):
    """Encrypt 32-byte message m under Wall-U PKE with 32-byte coins."""
    n, t, q, eta = p["nU"], p["tU"], p["qU"], p["etaU"]
    du, dv, dt, b = p["duU"], p["dvU"], p["dtU"], p["bU"]
    seedU = pkU[:SEED_LEN]
    B = decompress(bitunpack(pkU[SEED_LEN:], dt, n * t), q, dt).reshape(n, t)
    A = _wall_u_expand_matrix(seedU, p)
    nb = 2 * eta * n // 8
    Spr = np.empty((n, t), dtype=np.int64)
    Epr = np.empty((n, t), dtype=np.int64)
    for c in range(t):
        Spr[:, c] = cbd_from_stream(_prf_noise(coins, DOM_ENC_U, c, nb), eta, n)
        Epr[:, c] = cbd_from_stream(_prf_noise(coins, DOM_ENC_U, t + c, nb), eta, n)
    E2 = cbd_from_stream(_prf_noise(coins, DOM_ENC_U, 2 * t, 2 * t * t // 8),
                         eta, t * t).reshape(t, t)
    U = (A.T @ Spr + Epr) % q                       # n x t
    M = _encode_mU(m, p)                            # t x t
    C = (B.T @ Spr + E2 + M) % q                    # t x t
    ctU = bitpack(compress(U.reshape(-1), q, du), du) + \
        bitpack(compress(C.reshape(-1), q, dv), dv)
    return ctU


def _wall_u_decrypt_pke(S: np.ndarray, ctU: bytes, p: dict) -> bytes:
    """Decrypt Wall-U PKE ciphertext back to the 32-byte message."""
    n, t, q = p["nU"], p["tU"], p["qU"]
    du, dv = p["duU"], p["dvU"]
    U = decompress(bitunpack(ctU[: n * t * du // 8], du, n * t), q, du).reshape(n, t)
    C = decompress(bitunpack(ctU[n * t * du // 8:], dv, t * t), q, dv).reshape(t, t)
    Mdec = (C - S.T @ U) % q
    return _decode_mU(Mdec, p)


# ----------------------------------------------------------------------------
# Serialization lengths
# ----------------------------------------------------------------------------

def lengths(name: str) -> dict:
    p = _P(name)
    nS, kS, nU, tU = p["nS"], p["kS"], p["nU"], p["tU"]
    pkS = SEED_LEN + kS * nS * p["dtS"] // 8
    pkU = SEED_LEN + nU * tU * p["dtU"] // 8
    ctS = kS * nS * p["duS"] // 8 + nS * p["dvS"] // 8
    ctU = nU * tU * p["duU"] // 8 + tU * tU * p["dvU"] // 8
    return dict(pkS=pkS, pkU=pkU, pk=pkS + pkU,
                ctS=ctS, ctU=ctU, ct=ctS + ctU,
                sk=KEYGEN_SEED_LEN, ss=32)


# ----------------------------------------------------------------------------
# KEM: keygen / encapsulate / decapsulate
# ----------------------------------------------------------------------------

def keygen(name: str, seed: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """Generate (pk, sk).  For KAT/deterministic use, pass 160 bytes of seed
    material; otherwise fresh randomness is drawn from os.urandom.

    sk is exactly the 160-byte keygen seed: seedS || skSeedS || seedU ||
    skSeedU || z.  The public key is rebuilt deterministically on decapsulate.
    """
    p = _P(name)
    if seed is None:
        seed = os.urandom(KEYGEN_SEED_LEN)
    assert len(seed) == KEYGEN_SEED_LEN, \
        "keygen seed must be %d bytes" % KEYGEN_SEED_LEN
    pk = _public_key(name, seed)
    return pk, seed  # sk is the 160-byte seed material itself


def _build_all(name: str, sk: bytes):
    p = _P(name)
    seedS = sk[0:SEED_LEN]
    skSeedS = sk[SEED_LEN:2 * SEED_LEN]
    seedU = sk[2 * SEED_LEN:3 * SEED_LEN]
    skSeedU = sk[3 * SEED_LEN:4 * SEED_LEN]
    pkS, sS = _wall_s_keypair_pke(seedS, skSeedS, p)
    pkU, S = _wall_u_keypair_pke(seedU, skSeedU, p)
    return pkS + pkU, pkS, sS, pkU, S, sk[4 * SEED_LEN:5 * SEED_LEN]


def _public_key(name: str, sk: bytes) -> bytes:
    return _build_all(name, sk)[0]


def public_key(name: str, sk: bytes) -> bytes:
    """Rebuild the public key from a secret key (deterministic)."""
    return _public_key(name, sk)


def encapsulate(name: str, pk: bytes,
                coins: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """Encapsulate against pk.  Returns (ct, ss).  For KAT use, pass 64 bytes
    of coins (mS, mU).  Encryption randomness for both walls is derived from
    the messages and H(pk), as required by the implicit-rejection FO transform
    (decapsulation re-encrypts and compares)."""
    p = _P(name)
    L = lengths(name)
    if len(pk) != L["pk"]:
        raise ValueError("public key length mismatch (got %d, want %d)"
                         % (len(pk), L["pk"]))
    if coins is None:
        coins = os.urandom(ENCAP_COINS_LEN)
    assert len(coins) == ENCAP_COINS_LEN
    mS, mU = coins[0:32], coins[32:64]
    pkS, pkU = pk[:L["pkS"]], pk[L["pkS"]:]
    h_pk = _sha3(DOM_HASH + pk)
    coinsS = _shake256(DOM_RAND_S + mS + h_pk, SEED_LEN)
    coinsU = _shake256(DOM_RAND_U + mU + h_pk, SEED_LEN)
    ctS = _wall_s_encrypt_pke(pkS, mS, coinsS, p)
    ctU = _wall_u_encrypt_pke(pkU, mU, coinsU, p)
    ct = ctS + ctU
    h_ct = _sha3(DOM_HASH + ct)
    ss = _sha3(DOM_KDF + mS + mU + h_pk + h_ct)
    return ct, ss


def decapsulate(name: str, sk: bytes, ct: bytes) -> bytes:
    """Decapsulate ct under sk.  Always returns a 32-byte key: the correct
    shared secret for honestly generated ciphertexts, otherwise the
    pseudorandom implicit-rejection key."""
    p = _P(name)
    L = lengths(name)
    if len(sk) != KEYGEN_SEED_LEN:
        raise ValueError("secret key length mismatch (got %d, want %d)"
                         % (len(sk), KEYGEN_SEED_LEN))
    pk, pkS, sS, pkU, S, z = _build_all(name, sk)
    h_pk = _sha3(DOM_HASH + pk)
    h_ct = _sha3(DOM_HASH + ct)

    reject = _sha3(DOM_REJECT + z + h_ct)
    if len(ct) != L["ct"]:
        return reject

    ctS, ctU = ct[:L["ctS"]], ct[L["ctS"]:]
    try:
        mS = _wall_s_decrypt_pke(pkS, sS, ctS, p)
        mU = _wall_u_decrypt_pke(S, ctU, p)
        # FO re-encryption with message-derived randomness (same derivation
        # as encapsulation: randomness = f(m, H(pk)))
        coinsS = _shake256(DOM_RAND_S + mS + h_pk, SEED_LEN)
        coinsU = _shake256(DOM_RAND_U + mU + h_pk, SEED_LEN)
        ctS2 = _wall_s_encrypt_pke(pkS, mS, coinsS, p)
        ctU2 = _wall_u_encrypt_pke(pkU, mU, coinsU, p)
    except Exception:                       # defensive: never raise on bad ct
        return reject
    if ctS2 + ctU2 == ct:
        return _sha3(DOM_KDF + mS + mU + h_pk + h_ct)
    return reject


# ----------------------------------------------------------------------------
# KAT helpers
# ----------------------------------------------------------------------------

def kat_material(name: str) -> Tuple[bytes, bytes]:
    """Deterministic KAT seed material and encapsulation coins for a set."""
    seed = _shake256(b"DUOLITH-KAT-v1/keygen/" + name.encode(), KEYGEN_SEED_LEN)
    coins = _shake256(b"DUOLITH-KAT-v1/encaps/" + name.encode(), ENCAP_COINS_LEN)
    return seed, coins


def kat_vectors(name: str) -> dict:
    """Full deterministic keygen/encapsulate/decapsulate vector for a set."""
    seed, coins = kat_material(name)
    pk, sk = keygen(name, seed=seed)
    ct, ss = encapsulate(name, pk, coins=coins)
    ss2 = decapsulate(name, sk, ct)
    assert ss2 == ss
    return dict(name=name, seed=seed.hex(), coins=coins.hex(),
                pk=pk.hex(), sk=sk.hex(), ct=ct.hex(), ss=ss.hex())


if __name__ == "__main__":
    import time
    for name in ("DUOLITH-128", "DUOLITH-192"):
        L = lengths(name)
        print(name, "pk=%d ct=%d sk=%d ss=32" % (L["pk"], L["ct"], L["sk"]))
        t0 = time.perf_counter()
        pk, sk = keygen(name)
        t1 = time.perf_counter()
        ct, ss = encapsulate(name, pk)
        t2 = time.perf_counter()
        ss2 = decapsulate(name, sk, ct)
        t3 = time.perf_counter()
        assert ss == ss2
        print("  keygen %.1f ms  encaps %.1f ms  decaps %.1f ms  OK"
              % ((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3))
