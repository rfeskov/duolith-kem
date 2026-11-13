"""
Test suite for the DUOLITH-KEM reference implementation.

Run:  python3 test_duolith_kem.py
"""

import os
import secrets
import time

import numpy as np

import duolith_kem as dk

REAL_SETS = ("DUOLITH-128", "DUOLITH-192")
PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  [PASS] %s" % name)
    else:
        FAIL += 1
        print("  [FAIL] %s %s" % (name, detail))


# ---------------------------------------------------------------------------
print("== 1. Serialization lengths ==")
EXPECTED = {
    "DUOLITH-128": dict(pkS=864, pkU=9632, pk=10496, ctS=960, ctU=9064,
                        ct=10024, sk=160, ss=32),
    "DUOLITH-192": dict(pkS=1280, pkU=11552, pk=12832, ctS=1344, ctU=10856,
                        ct=12200, sk=160, ss=32),
}
for name in REAL_SETS:
    L = dk.lengths(name)
    check("lengths %s" % name, L == EXPECTED[name], str(L))

# ---------------------------------------------------------------------------
print("== 2. Bit packing round-trips ==")
rng = np.random.default_rng(1)
ok = True
for bits in (1, 4, 5, 11, 12, 13, 15, 16):
    n = 64 * bits  # byte aligned
    vals = rng.integers(0, 1 << bits, size=n).astype(np.int64)
    buf = dk.bitpack(vals, bits)
    back = dk.bitunpack(buf, bits, n)
    ok &= np.array_equal(vals, back)
check("bitpack/bitunpack widths", ok)

# ---------------------------------------------------------------------------
print("== 3. Compress/Decompress bounds ==")
ok = True
for q, d in ((7681, 12), (7681, 6), (65536, 15), (65536, 14), (65536, 13)):
    x = rng.integers(0, q, size=20000).astype(np.int64)
    y = dk.compress(x, q, d)
    assert y.min() >= 0 and y.max() < (1 << d)
    back = dk.decompress(y, q, d)
    err = (back - x) % q                      # error as an element of Z_q
    err_c = np.minimum(err, q - err)          # centered magnitude
    w = q / float(1 << d)
    ok &= bool(err_c.max() <= np.ceil(w / 2) + 1)
    ok &= bool(np.array_equal(dk.decompress(dk.compress(back, q, d), q, d), back))
check("compress/decompress round-trip + error bound", ok)

# ---------------------------------------------------------------------------
print("== 4. CBD sampling statistics ==")
ok = True
for eta in (3, 8):
    stream = dk._shake256(b"cbd-test", 2 * eta * 200000 // 8)
    v = dk.cbd_from_stream(stream, eta, 200000)
    ok &= bool(v.min() >= -eta and v.max() <= eta)
    ok &= abs(v.mean()) < 0.05
    var = eta / 2.0
    ok &= abs(v.var() - var) < 0.03 * var
check("CBD mean/variance/support", ok)

ok = True
try:
    dk.cbd_from_stream(b"\x00" * 16, 8, 64)   # 128 bits for 1024 needed
    ok = False
except ValueError:
    pass
check("CBD rejects a too-short stream", ok)

# (with this guard, any under-sized noise stream -- like the old Wall-U E''
# call -- raises inside encryption and fails the round-trip tests below)

# ---------------------------------------------------------------------------
print("== 5. Wall-level PKE round-trips (real parameters) ==")
for name in REAL_SETS:
    p = dk._P(name)
    okS = okU = True
    for _ in range(20):
        m_s = os.urandom(p["nS"] // 8)
        m_u = os.urandom(p["tU"] * p["tU"] * p["bU"] // 8)
        coins = os.urandom(32)
        pkS, s = dk._wall_s_keypair_pke(os.urandom(32), os.urandom(32), p)
        ctS = dk._wall_s_encrypt_pke(pkS, m_s, coins, p)
        okS &= dk._wall_s_decrypt_pke(pkS, s, ctS, p) == m_s
        pkU, S = dk._wall_u_keypair_pke(os.urandom(32), os.urandom(32), p)
        ctU = dk._wall_u_encrypt_pke(pkU, m_u, coins, p)
        okU &= dk._wall_u_decrypt_pke(S, ctU, p) == m_u
    check("%s Wall-S PKE roundtrip x20" % name, okS)
    check("%s Wall-U PKE roundtrip x20" % name, okU)

# ---------------------------------------------------------------------------
print("== 6. KEM round-trips ==")
for name in REAL_SETS:
    ok = True
    for _ in range(30):
        pk, sk = dk.keygen(name)
        ct, ss = dk.encapsulate(name, pk)
        ss2 = dk.decapsulate(name, sk, ct)
        ok &= ss == ss2 and len(ss) == 32
    check("%s KEM encaps/decaps x30" % name, ok)

# ---------------------------------------------------------------------------
print("== 7. Determinism and coin sensitivity ==")
name = "DUOLITH-128"
pk, sk = dk.keygen(name)
c1 = os.urandom(dk.ENCAP_COINS_LEN)
ct1, ss1 = dk.encapsulate(name, pk, coins=c1)
ct2, ss2 = dk.encapsulate(name, pk, coins=c1)
check("same coins -> same ct/ss", ct1 == ct2 and ss1 == ss2)
_, ss3 = dk.encapsulate(name, pk, coins=os.urandom(dk.ENCAP_COINS_LEN))
check("different coins -> different ss", ss1 != ss3)

# ---------------------------------------------------------------------------
print("== 8. pk rebuild from sk ==")
for name in REAL_SETS:
    pk, sk = dk.keygen(name)
    check("public_key(sk) == pk (%s)" % name, dk.public_key(name, sk) == pk)

# ---------------------------------------------------------------------------
print("== 9. Tamper detection (implicit rejection) ==")
name = "DUOLITH-192"
L = dk.lengths(name)
pk, sk = dk.keygen(name)
ct, ss = dk.encapsulate(name, pk)
regions = [("ctS-u", 10), ("ctS-v", L["ctS"] - 5),
           ("ctU-U", L["ctS"] + 10), ("ctU-C", L["ct"] - 3)]
ok = True
for tag, pos in regions:
    bad = bytearray(ct)
    bad[pos] ^= 0x01
    bad = bytes(bad)
    k1 = dk.decapsulate(name, sk, bad)
    k2 = dk.decapsulate(name, sk, bad)
    check("tamper %s -> reject (deterministic)" % tag,
          k1 != ss and k1 == k2)
# wrong keypair
pk_other, sk_other = dk.keygen(name)
kw = dk.decapsulate(name, sk_other, ct)
check("wrong sk -> reject key", kw != ss)

# ---------------------------------------------------------------------------
print("== 10. Malformed ciphertexts ==")
ok = True
for bad in (ct[:-1], ct + b"\x00", b"", ct[:L["ctS"]]):
    k = dk.decapsulate(name, sk, bad)
    ok &= k == dk.decapsulate(name, sk, bad) and k != ss and len(k) == 32
check("malformed ct -> deterministic reject key", ok)

# ---------------------------------------------------------------------------
print("== 11. API input validation ==")
try:
    dk.encapsulate(name, pk[:-1])
    check("encapsulate rejects short pk", False)
except ValueError:
    check("encapsulate rejects short pk", True)
try:
    dk.decapsulate(name, sk[:-1], ct)
    check("decapsulate rejects short sk", False)
except ValueError:
    check("decapsulate rejects short sk", True)

# ---------------------------------------------------------------------------
print("== 12. Known-answer tests (deterministic seeds/coins) ==")
for name in REAL_SETS:
    v1 = dk.kat_vectors(name)
    v2 = dk.kat_vectors(name)
    check("KAT reproducibility %s" % name,
          all(v1[k] == v2[k] for k in ("pk", "sk", "ct", "ss")))
    print("    %s ss = %s" % (name, v1["ss"]))

# ---------------------------------------------------------------------------
print("== 13. Multi-recipient independence ==")
name = "DUOLITH-128"
pka, ska = dk.keygen(name)
pkb, skb = dk.keygen(name)
cta, ssa = dk.encapsulate(name, pka)
ctb, ssb = dk.encapsulate(name, pkb)
check("recipient A", dk.decapsulate(name, ska, cta) == ssa)
check("recipient B", dk.decapsulate(name, skb, ctb) == ssb)
check("cross-decapsulation rejected", dk.decapsulate(name, ska, ctb) != ssb)

print("\n%d passed, %d failed" % (PASS, FAIL))
if FAIL:
    raise SystemExit(1)
