"""PyTorch as a backend, measured rather than claimed (issue #95).

``docs/design.md`` lists PyTorch in five places and ``ops/basic.py`` says arithmetic "must
already work on JAX (or torch) blocks"; this module is what makes those statements
enforceable. Every array touch in ``src/`` goes through ``ar.do``, so the coverage
here is **breadth of ops** — every public op once on torch blocks — rather than
breadth of fixtures: the numerics are bit-identical to NumPy's, which is asserted
with ``np.array_equal`` and not ``allclose``, so parametrizing the other 20
``to_backend`` test modules over a third backend would prove nothing new.

Scope is **eager**: ``.backward()`` through ``get_params``/``set_params``.
``torch.compile``, ``torch.func`` and a torch analogue of ``tenet.pytree`` are out,
and ``tenet.ad`` stays JAX-only — pinned below as an executable fact rather than a
docstring hope.
"""

import math
import pathlib
import re

import autoray as ar
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import from_matrices, to_matrices
from tenet.ops.dense import to_dense
from tenet.symmetry import (
    SU2,
    U1,
    FZ2Sector,
    ProductProvider,
    ProductSector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    fZ2,
)

torch = pytest.importorskip("torch")

# ``tests/conftest.py``'s ``jax_enable_x64`` argument, in torch's spelling: at
# float32 a central difference at h = 1e-6 has no usable digits, and the
# bit-identical comparisons against NumPy's float64 would be meaningless. It is
# process-global in the same way; it lives here rather than in ``conftest.py``
# only because, unlike JAX's flag, it is genuinely reversible and no other module
# needs it.
torch.set_default_dtype(torch.float64)


# --- fixtures --------------------------------------------------------------------

T_TRIV = tuple(GradedSpace.new(Trivial, {TrivialSector(): m}) for m in (2, 3, 4, 2))
Q1 = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
Q2 = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 3})
V = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 3})
W = GradedSpace.new(SU2, {SU2Sector(1): 2, SU2Sector(2): 1})
X = GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(2): 2})
F1 = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 3})
F2 = GradedSpace.new(fZ2, {FZ2Sector(0): 3, FZ2Sector(1): 1})

UU = ProductProvider((U1, U1))


def uu(a: int, b: int) -> ProductSector:
    return ProductSector((U1Sector(a), U1Sector(b)))


P1 = GradedSpace.new(UU, {uu(0, 0): 2, uu(1, 0): 2, uu(0, 1): 1})
P2 = GradedSpace.new(UU, {uu(0, 0): 2, uu(1, 0): 1})

# Four legs with interleaved sides, as in tests/ops/test_blocks.py.
LEGS = {
    "trivial": tuple(Leg(s, side) for s, side in zip(T_TRIV, (OUT, IN, OUT, IN), strict=True)),
    "u1": (Leg(Q1, OUT), Leg(Q2, IN), Leg(Q2, OUT), Leg(Q1, IN)),
    "su2": (Leg(V, OUT), Leg(W, IN), Leg(X, OUT), Leg(V, IN)),
    "fz2": (Leg(F1, OUT), Leg(F2, IN), Leg(F2, OUT), Leg(F1, IN)),
    # Both OUT legs in the codomain, domain mirroring it in order. That shape was
    # forced when a product forwarded no BendingCoefficients (#40); since #312 it does,
    # and the arrangement is kept as an ordinary fixture choice rather than a
    # constraint -- the two spaces still differ, which is what the axes below read.
    "product": (Leg(P1, OUT), Leg(P2, OUT), Leg(P1, IN), Leg(P2, IN)),
}
PROVIDERS = list(LEGS)

# Square maps, for eigh / expm / eig / trace of an endomorphism.
SQUARE = {
    "u1": (Leg(Q1, OUT), Leg(Q1, IN)),
    "su2": (Leg(V, OUT), Leg(V, IN)),
    "fz2": (Leg(F1, OUT), Leg(F1, IN)),
}


def nt(legs, seed=0):
    """A random NumPy-backed float64 tensor — the reference for every comparison."""
    return SymmetricTensor.random(legs, seed=seed)


def tt(legs, seed=0):
    """The same tensor, on torch."""
    return nt(legs, seed).to_backend("torch")


def complex_tensor(legs, seed=0, backend="numpy"):
    a, b = nt(legs, seed), nt(legs, seed + 100)
    t = SymmetricTensor(
        a.structure,
        tuple((x + 1j * y).astype("complex128") for x, y in zip(a.blocks, b.blocks, strict=True)),
    )
    return t.to_backend(backend)


def resolved(b):
    """``torch.conj`` returns a *view* with the conjugate bit set; ``.numpy()`` refuses it.

    autoray's ``torch_to_numpy`` is a bare ``x.detach().cpu().numpy()``, so a
    conjugated complex torch block cannot be handed to NumPy until it is
    materialized. Nothing in ``tenet`` is wrong here — ``conj`` is one ``ar.do``
    and the view is torch's own optimization — and it is not worked around in
    library code; it is a torch/autoray interop fact, pinned by
    ``test_conj_returns_a_lazy_view`` below.
    """
    return b.resolve_conj() if isinstance(b, torch.Tensor) else b


def npb(t):
    """The blocks as NumPy arrays, whatever backend they are on."""
    return [ar.to_numpy(resolved(b)) for b in t.blocks]


def same(a, b):
    """Bit-identical blocks, structure included. ``array_equal``, never ``allclose``."""
    assert a.structure == b.structure
    for x, y in zip(npb(a), npb(b), strict=True):
        assert np.array_equal(x, y)


def close(a, b, atol=1e-12):
    """Whenever the value comes out of a backend *kernel* rather than out of ``tenet``.

    ``same`` is for what the library computes itself — data movement and the
    categorical coefficients — and it holds exactly on every platform. A ``matmul``
    (``compose``, ``tensordot``, ``einsum``, ``trace``) sums in the backend's order,
    and torch's vectorized ``sqrt`` is not the correctly-rounded one NumPy calls, so
    those differ in the last ulp: measured 0.0 on macOS arm64 and nonzero on the
    linux CI runner, i.e. a BLAS/kernel property and not a tenet one.
    """
    assert a.structure == b.structure
    for x, y in zip(npb(a), npb(b), strict=True):
        np.testing.assert_allclose(x, y, rtol=0, atol=atol)


def is_torch(t):
    return t.backend == "torch" and all(isinstance(b, torch.Tensor) for b in t.blocks)


# --- the tensor surface ----------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_to_backend_round_trip(name):
    a = nt(LEGS[name], seed=1)
    t = a.to_backend("torch")
    assert is_torch(t)
    assert t.dtype == torch.float64
    assert str(t.device) == "cpu"  # a plain getattr on the block; CPU-only by design
    same(t.to_backend("numpy"), a)


def test_complex128_round_trip():
    a = complex_tensor(LEGS["su2"], seed=2)
    t = a.to_backend("torch")
    assert t.dtype == torch.complex128
    same(t.to_backend("numpy"), a)


@pytest.mark.parametrize("name", PROVIDERS)
def test_arithmetic(name):
    a, b = tt(LEGS[name], seed=3), tt(LEGS[name], seed=4)
    ra, rb = a.to_backend("numpy"), b.to_backend("numpy")
    for got, want in (
        (tenet.add(a, b), tenet.add(ra, rb)),
        (tenet.subtract(a, b), tenet.subtract(ra, rb)),
        (tenet.negative(a), tenet.negative(ra)),
        (tenet.multiply(a, 2.5), tenet.multiply(ra, 2.5)),
        (tenet.divide(a, 4.0), tenet.divide(ra, 4.0)),
        (tenet.conj(a), tenet.conj(ra)),
        (a + b, ra + rb),
        (a - b, ra - rb),
        (-a, -ra),
        (a * 2.5, ra * 2.5),
        (a / 4.0, ra / 4.0),
    ):
        assert is_torch(got)
        same(got, want)


@pytest.mark.parametrize("name", PROVIDERS)
def test_norm_allclose_and_eq(name):
    t = tt(LEGS[name], seed=5)
    n = tenet.norm(t)
    assert isinstance(n, torch.Tensor)
    # ``float(n)`` to a few ulp here; the *exact* claim is the SU(2) one below, which
    # is the case the issue measured at 0.0 (a summation order can differ elsewhere).
    assert float(n) == pytest.approx(float(tenet.norm(t.to_backend("numpy"))), rel=1e-14)
    assert tenet.allclose(t, t.copy())
    assert not tenet.allclose(t, t * 2.0)
    assert t == t.copy()
    assert t != t * 2.0


def test_conj_of_complex_blocks():
    t = complex_tensor(LEGS["u1"], seed=6, backend="torch")
    same(tenet.conj(t), tenet.conj(t.to_backend("numpy")))


def test_conj_returns_a_lazy_view():
    """The one torch wrinkle found: ``ar.to_numpy`` of a conjugated complex block raises.

    ``torch.conj`` is a view carrying a conjugate *bit*; autoray's ``to_numpy`` calls
    ``.numpy()`` on it, which torch refuses until ``.resolve_conj()``. It affects
    ``ar.do("to_numpy", conj(t))`` and ``tenet.save`` of such a tensor, and nothing
    else — every arithmetic and linalg path consumes the view happily. Left as an
    upstream fact rather than materialized inside ``tenet.conj``, which would copy
    every conjugate for the benefit of one export path.
    """
    b = tenet.conj(complex_tensor(LEGS["u1"], seed=6, backend="torch")).blocks[0]
    assert b.is_conj()
    with pytest.raises(RuntimeError, match="conjugate bit"):
        ar.to_numpy(b)
    assert isinstance(ar.to_numpy(b.resolve_conj()), np.ndarray)


# --- structural ops --------------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_transpose(name):
    """SU(2) braiding (F/R symbols) and fZ2 Koszul signs on torch blocks."""
    t = tt(LEGS[name], seed=7)
    perm = (0, 2, 1, 3)  # interleaving only: legal without a within-side braid
    got = tenet.transpose(t, perm)
    assert is_torch(got)
    same(got, tenet.transpose(t.to_backend("numpy"), perm))


@pytest.mark.parametrize("name", PROVIDERS)
def test_braid(name):
    """The crossing a planar embedding forces: a swap gate on torch blocks.

    Identity permutation, two levels inverted — the legs do not move, and every
    block whose two crossed lines are both odd changes sign (nothing changes on a
    bosonic provider). ``numpy`` is the oracle, as everywhere in this file.
    """
    t = tt(LEGS[name], seed=7)
    got = tenet.braid(t, (0, 1, 2, 3), (1, 0, 2, 3))
    assert is_torch(got)
    assert got.legs == t.legs
    same(got, tenet.braid(t.to_backend("numpy"), (0, 1, 2, 3), (1, 0, 2, 3)))


@pytest.mark.parametrize("name", PROVIDERS)
def test_twist(name):
    """The ribbon twist on torch blocks: one real scalar per block, no axis moves.

    ``theta`` is ``1`` on a bosonic provider, where the tensor comes back untouched,
    and ``(-1)^parity`` on a fermion-parity grading -- so the involution below is the
    statement that carries content on both. ``numpy`` is the oracle, as everywhere here.
    """
    t = tt(LEGS[name], seed=7)
    got = tenet.twist(t, (0, 2))
    assert is_torch(got)
    assert got.legs == t.legs
    same(got, tenet.twist(t.to_backend("numpy"), (0, 2)))
    same(tenet.twist(got, (0, 2)), t.to_backend("numpy"))


@pytest.mark.parametrize("name", PROVIDERS)
def test_flip(name):
    """#142's duality flip on torch blocks — the a1 hole of #146.

    A single flip of a non-dual leg is a pure relabel (no phase), so the double
    flip is asserted too: it multiplies each block by ``chi * theta`` — ``-1`` on
    an SU(2) half-integer or odd-fermion line — which is the arithmetic worth
    seeing on a torch block. One scalar per block, tenet's own: ``same``.
    """
    t = tt(LEGS[name], seed=42)
    r = t.to_backend("numpy")
    got = tenet.flip_dual(t, 0)
    assert is_torch(got)
    same(got, tenet.flip_dual(r, 0))
    same(tenet.flip_dual(got, 0), tenet.flip_dual(tenet.flip_dual(r, 0), 0))


@pytest.mark.parametrize("name", PROVIDERS)
def test_repartition_fuse_unfuse_adjoint(name):
    t = tt(LEGS[name], seed=8)
    r = t.to_backend("numpy")
    m, mr = tenet.repartition(t, (0, 1), (2, 3)), tenet.repartition(r, (0, 1), (2, 3))
    same(m, mr)
    # fuse needs same-side axes, which the repartition above arranges
    fused, fused_r = tenet.fuse(m, (0, 1)), tenet.fuse(mr, (0, 1))
    assert is_torch(fused)
    same(fused, fused_r)
    legs = m.legs[:2]
    same(tenet.unfuse(fused, 0, legs), tenet.unfuse(fused_r, 0, legs))
    same(tenet.adjoint(t), tenet.adjoint(r))


@pytest.mark.parametrize("name", PROVIDERS)
def test_embed(name):
    t = tt(LEGS[name], seed=9)
    bigger = tuple(
        Leg(GradedSpace.new(leg.space.provider, {a: m + 1 for a, m in leg.space.sectors}), leg.side)
        for leg in t.legs
    )
    got = tenet.embed(t, bigger)
    assert is_torch(got)
    same(got, tenet.embed(t.to_backend("numpy"), bigger))


@pytest.mark.parametrize("name", PROVIDERS)
def test_restrict_and_direct_sum(name):
    """The other two ``np.promote_types(block.dtype, ...)`` sites, fixed with #95's."""
    t = tt(LEGS[name], seed=10)
    r = t.to_backend("numpy")
    smaller = tuple(
        Leg(
            GradedSpace.new(leg.space.provider, {a: max(m - 1, 1) for a, m in leg.space.sectors}),
            leg.side,
        )
        for leg in t.legs
    )
    got = tenet.restrict(t, smaller, atol=math.inf)  # default atol is checked below
    assert is_torch(got)
    same(got, tenet.restrict(r, smaller, atol=math.inf))

    summed = tenet.direct_sum(t, tt(LEGS[name], seed=11), 0)
    assert is_torch(summed)
    same(summed, tenet.direct_sum(r, tt(LEGS[name], seed=11).to_backend("numpy"), 0))


def test_restrict_with_the_default_atol():
    """``_refuse_discarded``'s default tolerance goes through ``ar.get_dtype_name`` too."""
    t = tt(LEGS["u1"], seed=12)
    legs = t.legs
    assert tenet.restrict(t, legs).backend == "torch"  # a no-op restriction: nothing dropped


@pytest.mark.parametrize("name", PROVIDERS)
def test_apply_blocks_sqrt_power(name):
    t = tenet.apply_blocks(tt(LEGS[name], seed=13), lambda b: ar.do("abs", b))
    assert is_torch(t)
    r = t.to_backend("numpy")
    close(tenet.block_sqrt(t), tenet.block_sqrt(r), atol=1e-12)  # torch's vectorized sqrt, to a ulp
    close(
        tenet.block_power(t, 3), tenet.block_power(r, 3), atol=1e-12
    )  # `pow` is torch's, to a ulp
    same(t.apply_blocks(lambda b: b * 2), r.apply_blocks(lambda b: b * 2))
    u = tt(LEGS[name], seed=15)
    same(
        tenet.zip_blocks(t, u, lambda x, y: x + y),
        tenet.zip_blocks(r, u.to_backend("numpy"), lambda x, y: x + y),
    )


def test_map_diagonal():
    """The reduced-basis diagonal is a per-block ``einsum``, so torch must reach it too."""
    legs = (Leg(V, OUT), Leg(W, OUT), Leg(V, IN), Leg(W, IN))
    t = tt(legs, seed=16)
    d = tenet.map_diagonal(t)
    assert is_torch(d)
    same(d, tenet.map_diagonal(t.to_backend("numpy")))


def test_cast():
    """SU(2) -> U(1) forgetting: ``to_dense`` out, ``from_dense`` back in, on torch."""
    t = tt((Leg(V, OUT), Leg(V, IN)), seed=14)
    got = tenet.to_symmetry(t, U1)
    assert is_torch(got)
    same(got, tenet.to_symmetry(t.to_backend("numpy"), U1))


def test_save_and_load(tmp_path):
    """``save`` writes ``ar.to_numpy(b)``; ``load`` returns NumPy blocks, as documented."""
    t = tt(LEGS["su2"], seed=15)
    path = tmp_path / "t.npz"
    tenet.save(t, path)
    back = tenet.load(path)
    assert back.backend == "numpy"
    same(back, t.to_backend("numpy"))


# --- contraction -----------------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_compose_and_tensordot(name):
    """SU(2) braiding and fZ2 Koszul signs again, this time inside a contraction."""
    a, b = tt(LEGS[name], seed=16), tt(LEGS[name], seed=17)
    ra, rb = a.to_backend("numpy"), b.to_backend("numpy")
    # leg 0 against leg 3 is a contractible pair everywhere but the product row, whose
    # fixture puts two *different* spaces on those axes; it takes the
    # composition-shaped pattern instead (a's IN axes against b's OUT axes). Not a
    # capability limit any more -- #312 forwarded bending -- a fixture one.
    axes = ((2, 3), (0, 1)) if name == "product" else ((0,), (3,))
    got = tenet.tensordot(a, b, axes)
    assert is_torch(got)
    # `close`, not `same`: a contraction is a `matmul`, and its summation order is
    # the backend's. The bit-identical claims are the eight of #95's table, below.
    close(got, tenet.tensordot(ra, rb, axes))

    m, rm = tenet.repartition(a, (0, 1), (2, 3)), tenet.repartition(ra, (0, 1), (2, 3))
    n, rn = tenet.repartition(b, (0, 1), (2, 3)), tenet.repartition(rb, (0, 1), (2, 3))
    # `m @ n` is only composable when m's domain is n's codomain; use m @ adjoint(m)
    close(m @ tenet.adjoint(m), rm @ tenet.adjoint(rm))
    close(tenet.compose(n, tenet.adjoint(n)), tenet.compose(rn, tenet.adjoint(rn)))


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_einsum_two_and_three_operands(name):
    a, b, c = (tt(LEGS[name], seed=s) for s in (18, 19, 20))
    ra, rb, rc = (x.to_backend("numpy") for x in (a, b, c))
    two = tenet.einsum("abcd,defg->abcefg", a, b)
    assert is_torch(two)
    close(two, tenet.einsum("abcd,defg->abcefg", ra, rb))  # a matmul: see `close`
    # three operands: the opt_einsum path (#67)
    three = tenet.einsum("abcd,defg,ghij->abcefhij", a, b, c)
    assert is_torch(three)
    close(three, tenet.einsum("abcd,defg,ghij->abcefhij", ra, rb, rc))


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_einsum_chain_fuses_on_torch(name):
    """The chain writes through ``out=``, which torch has and JAX does not."""
    a, b, c = (tt(LEGS[name], seed=s) for s in (21, 22, 23))
    ra, rb, rc = (x.to_backend("numpy") for x in (a, b, c))
    eqs = ("abcd,defg->abcefg", "abcefg,ghij->abcefhij")
    chained = tenet.einsum_chain([(eqs[0], a, b, ""), (eqs[1], None, c, "")])
    assert is_torch(chained)
    close(chained, tenet.einsum_chain([(eqs[0], ra, rb, ""), (eqs[1], None, rc, "")]))


@pytest.mark.parametrize("name", ["u1", "su2"])
@pytest.mark.parametrize("axes", [(1, 0), (0, 3)], ids=["same-side", "out-in"])
def test_trace_stays_on_torch(name, axes):
    """Regression, bug 1 of #95: ``trace`` used to leave torch for NumPy.

    Before the fix ``ops/map.py::identity`` hard-coded ``like="numpy"``, so
    ``trace``'s ``tensordot(t, identity(...))`` contracted torch blocks against
    NumPy ones and torch refused:

        TypeError: matmul(): argument 'other' (position 2) must be Tensor,
                   not numpy.ndarray

    The identical call on **JAX** passed and returned a JAX tensor, because
    ``jnp.matmul`` silently promotes a NumPy operand — so this was NumPy/JAX
    leniency hiding a latent bug in ``identity``, not a torch defect. ``(1, 0)`` is
    the same-side pair, i.e. the bend path; ``(0, 3)`` the OUT/IN one.
    """
    # axis 1 is the dual object of axis 0, so the same-side pair contracts
    legs = (Leg(Q1, OUT), Leg(Q1, OUT, dual=True), Leg(Q2, IN), Leg(Q1, IN))
    if name == "su2":
        legs = (Leg(V, OUT), Leg(V, OUT, dual=True), Leg(W, IN), Leg(V, IN))
    t = tt(legs, seed=21)
    got = tenet.trace(t, axes)
    assert is_torch(got)
    close(got, tenet.trace(t.to_backend("numpy"), axes))  # a matmul: see `close`


@pytest.mark.parametrize("name", PROVIDERS)
def test_full_trace_and_inner(name):
    """The scalar exits of #126 on torch blocks — the a2 hole of #146.

    ``full_trace`` needs a square map, which ``LEGS``' interleaved fixtures are
    not, so each provider gets leg 0 against its IN partner. Both scalars are
    backend sums (a ``trace``, a contraction), hence ``approx`` and not
    ``array_equal`` — the same reasoning as ``close``.
    """
    legs = (LEGS[name][0], Leg(LEGS[name][0].space, IN))
    t = tt(legs, seed=43)
    ft = tenet.full_trace(t)
    assert isinstance(ft, torch.Tensor)
    assert float(ft) == pytest.approx(float(tenet.full_trace(t.to_backend("numpy"))), rel=1e-14)
    a, b = tt(legs, seed=44), tt(legs, seed=45)
    # Until M62 the product row asserted a CapabilityError: the old `inner` bent
    # every leg past the first through `adjoint`, and a product provider forwards
    # no BendingCoefficients (#40). The coefficient-space `inner` bends nothing,
    # so the refusal fell away — a strict capability widening, and the row now
    # takes the same path as every other provider.
    got = tenet.inner(a, b)
    assert isinstance(got, torch.Tensor)
    want = tenet.inner(a.to_backend("numpy"), b.to_backend("numpy"))
    assert float(got) == pytest.approx(float(want), rel=1e-14)


def test_identity_still_defaults_to_numpy():
    """The ``like=`` default is today's behaviour, byte for byte."""
    legs = (Leg(Q1, OUT), Leg(Q2, IN))
    assert tenet.identity(legs).backend == "numpy"
    assert tenet.isometry(legs, legs).backend == "numpy"
    assert tenet.random_isometry(legs, legs, seed=0).backend == "numpy"


def test_identity_on_a_torch_reference():
    block = tt(LEGS["su2"], seed=22).blocks[0]
    legs = (Leg(V, OUT), Leg(V, IN))
    i = tenet.identity((legs[0],), dtype=ar.get_dtype_name(block), like=block)
    assert is_torch(i)
    same(i, tenet.identity((legs[0],)))
    t = tt((Leg(V, OUT), Leg(V, IN)), seed=23)
    same(i @ t, t)


# --- the map view ----------------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_to_matrices_and_from_matrices(name):
    t = tenet.repartition(tt(LEGS[name], seed=24), (0, 1), (2, 3))
    mats = to_matrices(t)
    assert all(isinstance(m, torch.Tensor) for m in mats.values())
    same(from_matrices(t.structure, mats), t)


# --- dense -----------------------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_to_dense_is_bit_identical(name):
    d = to_dense(tt(LEGS[name], seed=25))
    assert isinstance(d, torch.Tensor)
    assert np.array_equal(ar.to_numpy(d), to_dense(nt(LEGS[name], seed=25)))


def test_to_dense_complex_is_bit_identical():
    legs = LEGS["su2"]
    got = to_dense(complex_tensor(legs, seed=26, backend="torch"))
    want = to_dense(complex_tensor(legs, seed=26))
    assert np.array_equal(ar.to_numpy(got), want)


@pytest.mark.parametrize("name", PROVIDERS)
def test_from_dense_with_the_default_atol(name):
    """Regression, bug 2 of #95: the default-``atol`` dtype line.

    ``np.promote_types(getattr(dense, "dtype", np.float64), np.float32)`` raised

        TypeError: Cannot interpret 'torch.float64' as a data type

    because a torch tensor's ``.dtype`` is not a NumPy dtype, where a JAX array's
    *is* (``numpy.dtypes.Float32DType``) — which is why JAX never saw this.
    ``atol=math.inf`` already worked, localizing the failure to that one line;
    ``ar.get_dtype_name(dense)`` is the portable spelling ``ops/map.py::compose``
    already uses.
    """
    t = tt(LEGS[name], seed=27)
    back = SymmetricTensor.from_dense(to_dense(t), t.legs)  # default atol
    assert is_torch(back)
    close(back, t, atol=1e-12)


def test_a_block_less_operand_takes_the_other_one_s_backend():
    """A tensor whose legs cannot couple carries no block, hence no backend of its own.

    Contracting it against a torch operand still has to produce torch zeros wherever the
    output structure admits blocks — the rule that ``ops.map.block_ref`` states.
    """
    zero = GradedSpace.new(SU2, {SU2Sector(0): 1})
    one = GradedSpace.new(SU2, {SU2Sector(2): 1})
    empty = SymmetricTensor.random((Leg(zero, OUT), Leg(one, IN)), seed=0)
    assert empty.blocks == ()
    b = tt((Leg(one, OUT), Leg(one, IN), Leg(one, IN)), seed=31)

    c = tenet.tensordot(empty, b, axes=((1,), (0,)))
    assert is_torch(c)
    assert all(not block.any() for block in c.blocks)
    assert is_torch(tenet.einsum("ab,bcd->acd", empty, b))
    assert is_torch(tenet.einsum_chain([("ab,bcd->acd", empty, b, "")]))
    assert is_torch(empty @ b)


# --- linalg ----------------------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_svd_qr_lq_polar(name):
    t = tt(LEGS[name], seed=28)
    r = t.to_backend("numpy")
    axes = ((0, 1), (2, 3))
    u, s, vh = tenet.linalg.svd(t, axes)
    assert is_torch(u) and is_torch(s) and is_torch(vh)
    # the *bit-identical* singular-value claim is #95's U(1) one, asserted below;
    # here the point is breadth, and a LAPACK driver may differ in the last ulp
    close(s, tenet.linalg.svd(r, axes)[1], atol=1e-12)
    close(u @ s @ vh, tenet.repartition(t, *axes))

    q, rr = tenet.linalg.qr(t, axes)
    assert is_torch(q)
    close(q @ rr, tenet.repartition(t, *axes))
    lo, lq_q = tenet.linalg.lq(t, axes)
    assert is_torch(lq_q)
    close(lo @ lq_q, tenet.repartition(t, *axes))
    w, p = tenet.linalg.polar(t, axes)
    assert is_torch(w)
    close(w @ p, tenet.repartition(t, *axes))


@pytest.mark.parametrize("name", list(SQUARE))
def test_eigh_expm_eig_eigvals(name):
    t = tt(SQUARE[name], seed=29)
    herm = from_matrices(
        t.structure, {c: (m + ar.do("conj", m).T) / 2 for c, m in to_matrices(t).items()}
    )
    w, v = tenet.linalg.eigh(herm)
    assert is_torch(w) and is_torch(v)
    close(v @ w @ tenet.adjoint(v), herm, atol=1e-10)

    e = tenet.linalg.expm(t, alpha=0.5)
    assert is_torch(e)
    close(e, tenet.linalg.expm(t.to_backend("numpy"), alpha=0.5), atol=1e-10)

    vals, vecs = tenet.linalg.eig(t)
    assert is_torch(vals) and is_torch(vecs)
    # `eig` is complex always, and torch refuses a real-against-complex `matmul`
    # where NumPy silently promotes; the promotion is the caller's to spell.
    complex_t = t.apply_blocks(lambda b: ar.do("astype", b, "complex128"))
    close(complex_t @ vecs, vecs @ vals, atol=1e-10)
    close(tenet.linalg.eigvals(t), vals, atol=1e-10)


@pytest.mark.parametrize("name", ["u1", "su2", "fz2"])
def test_left_and_right_null(name):
    """A tall map for the cokernel and its adjoint (wide) for the kernel."""
    legs = {
        "u1": (Leg(Q1, OUT), Leg(Q2, IN)),
        "su2": (Leg(V, OUT), Leg(W, IN)),
        "fz2": (Leg(F1, OUT), Leg(F2, IN)),
    }[name]
    t = tt(legs, seed=30)
    n = tenet.linalg.left_null(t)
    assert is_torch(n)
    assert float(tenet.norm(tenet.adjoint(n) @ t)) < 1e-10
    wide = tenet.adjoint(t)
    m = tenet.linalg.right_null(wide)
    assert is_torch(m)
    assert float(tenet.norm(wide @ tenet.adjoint(m))) < 1e-10


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_svd_truncated_and_bond(name):
    t = tt(LEGS[name], seed=31)
    axes = ((0, 1), (2, 3))
    u, s, vh = tenet.linalg.svd_truncated(t, axes, max_bond=4)
    assert is_torch(u)
    bond = s.structure.legs[0].space
    # same kept structure on both backends, values to a ulp (the driver's)
    reference = tenet.linalg.svd_truncated(t.to_backend("numpy"), axes, max_bond=4)[1]
    close(s, reference, atol=1e-12)
    projected = tenet.linalg.svd(t, axes, bond=bond)
    assert is_torch(projected[0])
    same(projected[1], s)


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_select_bond_decides_the_same_bond_on_torch(name):
    """#209's two-call form on torch blocks: the decision needs Python floats out of a
    torch tensor, which is the only backend-facing thing ``select_bond`` does."""
    t = tt(LEGS[name], seed=31)
    axes = ((0, 1), (2, 3))
    selection = tenet.linalg.select_bond(t, axes, max_bond=4)
    assert isinstance(selection, tenet.linalg.BondSelection)
    reference = tenet.linalg.select_bond(t.to_backend("numpy"), axes, max_bond=4)
    assert selection.bond == reference.bond
    assert selection.discarded_weight == pytest.approx(reference.discarded_weight, rel=1e-12)
    # and it is the bond svd_truncated picks on the same blocks
    assert selection.bond == tenet.linalg.svd_truncated(t, axes, max_bond=4)[1].legs[0].space


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_eigh_truncated_and_bond(name):
    """#205's pair on torch blocks. The gather is `argsort` over `abs`, which is the
    only backend call `eigh_truncated` makes that `svd_truncated` does not."""
    t = tt(LEGS[name], seed=31)
    axes = ((0, 1), (2, 3))
    m = tenet.repartition(t, *axes)
    h = m @ tenet.adjoint(m)
    ax = ((0, 1), (2, 3))

    w, v = tenet.linalg.eigh_truncated(h, ax, max_bond=4)
    assert is_torch(w) and is_torch(v)
    bond = w.structure.legs[0].space
    reference = tenet.linalg.eigh_truncated(h.to_backend("numpy"), ax, max_bond=4)[0]
    close(w, reference, atol=1e-12)

    projected = tenet.linalg.eigh(t=h, axes=ax, bond=bond)
    assert is_torch(projected[0])
    same(projected[0], w)


# --- the bit-identical table (issue #95, all eight measured at 0.0) ---------------


def test_numerics_are_bit_identical_to_numpy():
    """``np.array_equal``, deliberately: 0.0 is what was measured, so 0.0 is asserted."""
    axes = ((0, 1), (2, 3))
    perm = (0, 2, 1, 3)
    for name in ("su2", "fz2"):
        legs = LEGS[name]
        t, r = tt(legs, seed=32), nt(legs, seed=32)
        # 1-2: to_dense
        assert np.array_equal(ar.to_numpy(to_dense(t)), to_dense(r))
        # 4: transpose -> to_dense (the braided / Koszul path)
        assert np.array_equal(
            ar.to_numpy(to_dense(tenet.transpose(t, perm))),
            to_dense(tenet.transpose(r, perm)),
        )
        # 5: fuse -> unfuse -> to_dense (on the repartitioned map: fuse wants one side)
        m, mr = tenet.repartition(t, *axes), tenet.repartition(r, *axes)
        rt = tenet.unfuse(tenet.fuse(m, (0, 1)), 0, m.legs[:2])
        rr = tenet.unfuse(tenet.fuse(mr, (0, 1)), 0, mr.legs[:2])
        assert np.array_equal(ar.to_numpy(to_dense(rt)), to_dense(rr))
        # 6: repartition -> to_dense
        assert np.array_equal(
            ar.to_numpy(to_dense(tenet.repartition(t, *axes))),
            to_dense(tenet.repartition(r, *axes)),
        )
    # 3: complex128 SU(2) to_dense
    legs = LEGS["su2"]
    assert np.array_equal(
        ar.to_numpy(to_dense(complex_tensor(legs, seed=33, backend="torch"))),
        to_dense(complex_tensor(legs, seed=33)),
    )
    # 7: svd singular values, U(1). The one entry of #95's table that is *not*
    # exact everywhere: it was measured at 0.0 on linux/OpenBLAS, and on macOS
    # arm64 (NumPy on Accelerate against torch's own LAPACK) the two drivers
    # differ in the last ulp. Everything above is arithmetic tenet does itself,
    # so it is exact on any platform; this one is a library call on each side.
    t, r = tt(LEGS["u1"], seed=34), nt(LEGS["u1"], seed=34)
    for x, y in zip(
        npb(tenet.linalg.svd(t, axes)[1]), npb(tenet.linalg.svd(r, axes)[1]), strict=True
    ):
        np.testing.assert_allclose(x, y, rtol=1e-14, atol=1e-14)
    # 8: norm on SU(2), with the qdim weight
    t, r = tt(LEGS["su2"], seed=35), nt(LEGS["su2"], seed=35)
    assert np.array_equal(float(tenet.norm(t)), float(tenet.norm(r)))


# --- parameters and dispatch -----------------------------------------------------


def test_get_params_and_set_params():
    t = tt(LEGS["su2"], seed=36)
    params = t.get_params()
    assert all(isinstance(b, torch.Tensor) for b in params)
    rebuilt = t.set_params(tuple(b * 1.0 for b in params))
    assert is_torch(rebuilt)
    same(rebuilt, t)


def test_autoray_entry_points_on_a_torch_tensor():
    """``ar.do("to_numpy", ...)`` gives an ndarray; ``linalg.norm`` a torch scalar."""
    t = tt(LEGS["u1"], seed=37)
    assert isinstance(ar.do("to_numpy", t), np.ndarray)
    assert isinstance(ar.do("linalg.norm", t), torch.Tensor)


@pytest.mark.parametrize("name", ["svd", "sqrt", "exp"])
def test_the_closed_list_is_unchanged_with_torch_installed(name):
    """Invariant 11: dispatch on a *SymmetricTensor* is a closed registration list.

    Unrelated: ``ar.do("svd", <bare torch array>)`` *does* resolve, because torch has
    a bare ``svd`` where NumPy does not. That is autoray reaching torch's own
    namespace for an ordinary array, not this refusal weakening.
    """
    with pytest.raises(Exception):  # noqa: B017  the exception *type* is autoray's business
        ar.do(name, tt(LEGS["u1"], seed=38))
    assert ar.do("svd", torch.eye(3)) is not None  # the lookalike, recorded


# --- eager autograd, through get_params/set_params and nothing else ---------------


def differentiable(t):
    """The whole idiom: no library code, no pytree registration, no ``torch.func``."""
    return t.set_params(tuple(b.detach().clone().requires_grad_(True) for b in t.get_params()))


def grads_are_finite(t, objective):
    """The gradient arrives on the leaves ``differentiable`` made, which are the params.

    ``get_params`` and not ``blocks``: the parameters are the coupled-sector matrices,
    and a block is a view cut out of one of them. A view is a non-leaf with no ``.grad``
    of its own, so asking a block for one would be asking the wrong array -- the same
    reason ``tenet.pytree`` gives JAX the matrices as leaves.
    """
    x = differentiable(t)
    objective(x).backward()
    for leaf, source in zip(x.get_params(), t.get_params(), strict=True):
        assert leaf.grad is not None
        assert leaf.grad.shape == source.shape
        assert bool(torch.isfinite(leaf.grad).all())


@pytest.mark.parametrize("name", ["u1", "su2"])
@pytest.mark.parametrize(
    "objective",
    [
        pytest.param(tenet.norm, id="norm"),
        pytest.param(lambda t: tenet.norm(tenet.linalg.svd(t)[1]), id="svd"),
        pytest.param(lambda t: tenet.norm(tenet.linalg.qr(t)[1]), id="qr"),
        pytest.param(lambda t: tenet.norm(tenet.linalg.eigh(t)[0]), id="eigh"),
        pytest.param(lambda t: ar.do("sum", ar.do("abs", to_dense(t))), id="dense"),
        pytest.param(lambda t: tenet.norm(tenet.transpose(t, (1, 0))), id="transpose"),
        pytest.param(
            lambda t: tenet.norm(tenet.fuse(tenet.repartition(t, (0, 1), ()), (0, 1))), id="fuse"
        ),
    ],
)
def test_eager_backward(name, objective):
    grads_are_finite(tt(SQUARE[name], seed=39), objective)


@pytest.mark.parametrize("dtype", [None, "float64"])
def test_to_backend_same_backend_keeps_the_graph(dtype):
    """``t.to_backend("torch")`` on a torch tensor must not detach it.

    ``ar.do("array", x, like="torch")`` is ``torch.tensor(x)``, which copies and
    detaches, so the rebuild-unconditionally version zeroed every gradient routed
    through a "move" that moves nothing.
    """
    grads_are_finite(
        tt(SQUARE["su2"], seed=41), lambda x: tenet.norm(x.to_backend("torch", dtype=dtype))
    )


def test_eager_backward_through_embed():
    t = tt(SQUARE["u1"], seed=40)
    bigger = tuple(
        Leg(GradedSpace.new(leg.space.provider, {a: m + 1 for a, m in leg.space.sectors}), leg.side)
        for leg in t.legs
    )
    grads_are_finite(t, lambda x: tenet.norm(tenet.embed(x, bigger)))


def test_finite_differences_on_the_singular_values():
    """Central differences at h = 1e-6 in float64, every entry of every block.

    Measured agreement 1.2e-11; the criterion is 1e-9.
    """
    t = tt(SQUARE["u1"], seed=41)

    def objective(x):
        return tenet.norm(tenet.linalg.svd(x)[1])

    x = differentiable(t)
    objective(x).backward()

    # over the parameters, which are the coupled-sector matrices: they are the leaves
    # ``differentiable`` made and the only arrays that carry a ``.grad``
    h = 1e-6
    params = t.get_params()
    for i, matrix in enumerate(params):
        want = np.zeros(tuple(matrix.shape))
        for idx in np.ndindex(want.shape):

            def shifted(delta, i=i, idx=idx):
                ms = [m.clone() for m in params]
                ms[i][idx] += delta
                return float(objective(t.set_params(ms)))

            want[idx] = (shifted(h) - shifted(-h)) / (2 * h)
        np.testing.assert_allclose(ar.to_numpy(x.get_params()[i].grad), want, rtol=0, atol=1e-9)


# --- tenet.ad stays JAX-only -----------------------------------------------------


def degenerate_torch_block():
    """Every singular value 1: the degeneracy #76 broadens for JAX and not for torch."""
    return torch.eye(3, dtype=torch.float64, requires_grad=True)


def svd_recon_grad(block):
    u, s, vh = ar.do("linalg.svd", block, full_matrices=False)
    (u @ ar.do("diag", s) @ vh).sum().backward()
    return block.grad


def test_ad_leaves_torch_alone():
    """Documented non-support, as an executable fact: NaN before *and* after install().

    ``tenet.ad.install()`` patches autoray's dispatch for the **jax** backend only.
    The torch equivalent is a ``torch.autograd.Function`` or a ``torch.library``
    registration — a different mechanism, a second implementation of #76's
    ``_safe_inverse`` reasoning, and no caller. See #95's "out of scope".
    """
    ad = pytest.importorskip("tenet.ad")  # needs JAX; the pin is about torch either way
    assert ad._NAMES == ("linalg.svd", "linalg.eigh")

    before = svd_recon_grad(degenerate_torch_block())
    assert not bool(torch.isfinite(before).all())
    try:
        ad.install()
        # `_FUNCS` is also autoray's resolution *cache*, so ("torch", ...) keys are
        # there simply because this module used torch; what proves the point is
        # whose function sits behind them (the shape test_ad.py uses for jax).
        for fn in ad._NAMES:
            assert not ar.get_lib_fn("torch", fn).__module__.startswith("tenet")
        after = svd_recon_grad(degenerate_torch_block())
        assert not bool(torch.isfinite(after).all())
    finally:
        ad.uninstall()


# --- astype / to_backend(dtype=) / keyed blocks (#207, #208) ----------------------


@pytest.mark.parametrize("spelling", ["complex128", np.complex128])
def test_astype_takes_a_numpy_dtype_or_a_name(spelling):
    """autoray's torch ``astype`` wants a dtype *name*, so ``astype`` normalizes to one.

    Without that, ``t.astype(np.complex128)`` — the spelling that works on NumPy and
    JAX — is a ``TypeError`` on torch, and the method would mean two different things
    depending on where the blocks live.
    """
    t = tt(LEGS["su2"], seed=1)
    c = t.astype(spelling)
    assert is_torch(c)
    assert c.dtype == torch.complex128
    for got, want in zip(c.blocks, t.blocks, strict=True):
        assert torch.equal(got.real, want)


def test_to_backend_dtype_lands_on_torch():
    a = nt(LEGS["su2"], seed=1)
    t = a.to_backend("torch", dtype=np.complex128)
    assert is_torch(t)
    assert t.dtype == torch.complex128
    for got, want in zip(t.blocks, a.to_backend("torch").blocks, strict=True):
        assert torch.equal(got.real, want)


@pytest.mark.parametrize("name", PROVIDERS)
def test_from_blocks_and_with_blocks_stay_on_torch(name):
    t = tt(LEGS[name], seed=1)
    key = t.structure.block_order[0]

    built = SymmetricTensor.from_blocks(t.legs, {key: t.block(key)})
    assert is_torch(built)  # the zero fill follows the supplied block's backend
    assert built.dtype == torch.float64
    assert torch.equal(built.block(key), t.block(key))
    assert all(not bool(b.any()) for k, b in built.items() if k != key)

    zeroed = t.with_blocks({key: torch.zeros_like(t.block(key))})
    assert is_torch(zeroed)
    assert not bool(zeroed.block(key).any())
    for other, block in zeroed.items():
        if other != key:
            assert torch.equal(block, t.block(other))


# --- the core still imports neither backend --------------------------------------


def test_only_this_module_imports_torch():
    """``tests/array/test_dispatch.py`` polices ``src/``; this policices ``tests/``."""
    root = pathlib.Path(__file__).parents[1]
    offenders = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*.py")
        # an actual import statement, not the string literal four other modules
        # assert the *absence* of in `src/`
        if re.search(
            r"^\s*(import torch\b|from torch\b|torch = pytest\.importorskip)",
            p.read_text(),
            re.MULTILINE,
        )
    }
    assert offenders == {"backends/test_torch.py"}


# --- the batched plan applier (#316) ----------------------------------------------


@pytest.mark.parametrize("provider", ["u1", "su2"])
def test_torch_takes_the_loop_and_gets_the_same_numbers(provider):
    """Torch is outside the batching gate, and its result is the looped one to the bit.

    [_batches][tenet.ops.repartition._batches] admits NumPy only, and Torch is out for
    want of a measurement rather than for a measured loss — this is the module where
    that measurement would be made. Until it is, the claim under test is the one every
    backend owes: the plan's execution equals the term-by-term one.
    """
    from ops.test_batch import assert_bit_identical, bend_plan_of, tensor

    from tenet.ops.repartition import _batches

    t = tensor(provider, "ragged", 5).to_backend("torch")
    assert not _batches(t)
    got = assert_bit_identical(t, *bend_plan_of(t))
    assert got.backend == "torch"
