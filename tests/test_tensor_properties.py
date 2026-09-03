"""Tests for the array-style properties of SymmetricTensor — issue #19."""

import dataclasses
import os
import subprocess
import sys

import numpy as np
import pytest

from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.symmetry import SU2, U1, CapabilityError, SU2Sector, Trivial, TrivialSector, U1Sector

HALF, ONE = SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {HALF: 4, ONE: 3})  # dim 4*2 + 3*3 = 17, reduced_dim 7
W = GradedSpace.new(SU2, {HALF: 2})
TRIV = GradedSpace.new(Trivial, {TrivialSector(): 3})
Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})

SU2_LEGS = (Leg(V, OUT), Leg(W, IN), Leg(V, OUT))
U1_LEGS = (Leg(Q, OUT), Leg(Q, IN), Leg(Q, OUT))
TRIV_LEGS = (Leg(TRIV, OUT), Leg(TRIV, IN))

# one OUT leg carrying only spin-1/2 and one IN leg carrying only spin-1 share
# no coupled sector, so block_order is genuinely empty
BLOCK_FREE_LEGS = (
    Leg(GradedSpace.new(SU2, {HALF: 2}), OUT),
    Leg(GradedSpace.new(SU2, {ONE: 2}), IN),
)


def keyed(legs, blocks):
    """Every block, keyed in ``block_order``."""
    return dict(zip(TensorStructure(tuple(legs)).block_order, blocks, strict=True))


@dataclasses.dataclass(frozen=True, slots=True)
class Bare:
    """A provider with fusion data but no ClebschGordanData capability."""

    name: str = "Bare"

    @property
    def unit(self):
        return TrivialSector()

    def dual(self, a):
        return a

    def fusion(self, a, b):
        return (TrivialSector(),)

    def n_symbol(self, a, b, c):
        return 1


# --- shape / reduced_shape ------------------------------------------------------


@pytest.mark.parametrize("legs", [TRIV_LEGS, U1_LEGS, SU2_LEGS])
def test_shape_is_the_dense_shape(legs):
    t = SymmetricTensor.random(legs, seed=0)
    assert t.shape == tuple(leg.space.dim for leg in legs)
    assert t.shape == t.to_dense().shape


def test_su2_shape_reduced_shape_and_block_extents_differ():
    t = SymmetricTensor.random(SU2_LEGS, seed=1)
    assert t.shape[0] == 4 * 2 + 3 * 3 == 17
    assert t.reduced_shape[0] == 4 + 3 == 7
    assert {b.shape[0] for b in t.blocks} <= {4, 3}


@pytest.mark.parametrize("legs", [TRIV_LEGS, U1_LEGS])
def test_abelian_shape_equals_reduced_shape(legs):
    t = SymmetricTensor.zeros(legs)
    assert t.shape == t.reduced_shape


def test_shape_requires_clebsch_gordan_but_reduced_shape_does_not():
    space = GradedSpace.new(Bare(), {TrivialSector(): 2})
    t = SymmetricTensor.zeros((Leg(space, OUT), Leg(space, IN)))
    with pytest.raises(CapabilityError, match="Bare.*ClebschGordanData"):
        _ = t.shape
    assert t.reduced_shape == (2, 2)


def test_ndim_agrees_with_both_shapes():
    t = SymmetricTensor.zeros(SU2_LEGS)
    assert t.ndim == len(t.shape) == len(t.reduced_shape) == len(t.legs) == 3


# --- dtype / backend / device ---------------------------------------------------


def test_dtype_is_the_shared_block_dtype():
    t = SymmetricTensor.random(SU2_LEGS, seed=2, dtype=np.float32)
    assert t.dtype == np.float32
    assert all(b.dtype == t.dtype for b in t.blocks)
    assert SymmetricTensor.zeros(SU2_LEGS).dtype == np.float64


def test_backend_and_device_for_numpy():
    t = SymmetricTensor.zeros(SU2_LEGS)
    assert t.backend == "numpy"
    assert str(t.device) == "cpu"


def test_core_never_imports_jax_or_torch():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "tenet"
    for path in root.rglob("*.py"):
        if path.name in ("pytree.py", "ad.py"):
            continue  # the opt-in modules (#41, #76); core never imports either
        text = path.read_text()
        assert "import jax" not in text and "import torch" not in text, path


# --- to_backend -----------------------------------------------------------------


def test_to_backend_jax_keeps_structure_and_values():
    pytest.importorskip("jax")
    # float32: JAX disables x64 by default, so it is the dtype that survives a move
    t = SymmetricTensor.random(SU2_LEGS, seed=3, dtype=np.float32)
    j = t.to_backend("jax")
    assert j.backend == "jax"
    assert j.structure == t.structure
    assert j.shape == t.shape and j.reduced_shape == t.reduced_shape
    assert j.dtype == t.dtype
    assert j.device is not None
    for a, b in zip(t.blocks, j.blocks, strict=True):
        np.testing.assert_array_equal(a, np.asarray(b))


def test_to_backend_round_trips_through_jax():
    pytest.importorskip("jax")
    t = SymmetricTensor.random(SU2_LEGS, seed=4, dtype=np.float32)
    back = t.to_backend("jax").to_backend("numpy")
    assert back.backend == "numpy"
    assert back == t


def test_to_backend_to_the_current_backend_is_a_no_op():
    # not just equal: the same block objects. A copy here is what detaches a torch
    # tensor from its autograd graph (see tests/backends/test_torch.py).
    t = SymmetricTensor.random(SU2_LEGS, seed=6)
    assert t.to_backend("numpy").data == t.data

    cast = t.to_backend("numpy", dtype=np.complex128)
    assert cast.dtype == np.dtype(np.complex128)
    assert cast.structure == t.structure
    for a, b in zip(t.blocks, cast.blocks, strict=True):
        np.testing.assert_array_equal(a, b.real)


# --- block-free tensor ----------------------------------------------------------


def test_block_free_tensor_has_structure_but_no_array_properties():
    t = SymmetricTensor.zeros(BLOCK_FREE_LEGS)
    assert t.blocks == ()
    assert t.shape == (2 * 2, 2 * 3)
    assert t.reduced_shape == (2, 2)
    assert t.ndim == 2
    for get in (lambda: t.dtype, lambda: t.backend, lambda: t.device):
        with pytest.raises(ValueError, match="no blocks"):
            get()


# --- equality, hashing, repr ----------------------------------------------------


def test_equality_is_exact_and_total():
    a = SymmetricTensor.random(SU2_LEGS, seed=5)
    b = SymmetricTensor.from_blocks(SU2_LEGS, keyed(SU2_LEGS, [x.copy() for x in a.blocks]))
    assert a == b

    blocks = [x.copy() for x in a.blocks]
    blocks[0] = blocks[0] + 1e-15
    assert a != SymmetricTensor.from_blocks(SU2_LEGS, keyed(SU2_LEGS, blocks))

    assert (a == SymmetricTensor.zeros(U1_LEGS)) is False
    assert (a == "not a tensor") is False


def test_unhashable_but_structure_is_hashable():
    t = SymmetricTensor.zeros(SU2_LEGS)
    with pytest.raises(TypeError):
        hash(t)
    hash(t.structure)
    assert {t.structure: 1}[t.structure] == 1


def test_repr_is_single_line_and_block_free():
    big = GradedSpace.new(SU2, {HALF: 20, ONE: 20})
    t = SymmetricTensor.random((Leg(big, OUT), Leg(big, IN)), seed=6)
    assert sum(b.size for b in t.blocks) > 500  # a default dataclass repr would be huge
    r = repr(t)
    assert "\n" not in r
    assert len(r) < 200
    assert "ndim=2" in r
    assert f"shape={t.shape}" in r
    assert "dtype=float64" in r
    assert "backend='numpy'" in r
    assert f"blocks={len(t.blocks)}" in r


def test_repr_of_block_free_tensor_does_not_raise():
    assert "blocks=0" in repr(SymmetricTensor.zeros(BLOCK_FREE_LEGS))


# --- astype / to_backend(dtype=) -- issue #207 -----------------------------------


def test_astype_round_trips_float64_and_complex128_keeping_structure_and_legs():
    t = SymmetricTensor.random(SU2_LEGS, seed=6)
    assert t.dtype == np.float64

    c = t.astype(np.complex128)
    assert c.dtype == np.complex128
    assert all(b.dtype == np.complex128 for b in c.blocks)
    assert c.structure == t.structure and c.legs == t.legs

    back = c.astype(np.float64)
    assert back.dtype == np.float64
    assert back.structure == t.structure and back.legs == t.legs
    assert back == t  # exact: the cast is lossless in both directions


def test_astype_leaves_the_original_alone():
    t = SymmetricTensor.random(SU2_LEGS, seed=6)
    t.astype(np.complex128)
    assert t.dtype == np.float64


def test_astype_and_to_backend_dtype_refuse_a_block_free_tensor():
    t = SymmetricTensor.zeros(BLOCK_FREE_LEGS)
    with pytest.raises(ValueError, match="no blocks"):
        t.astype(np.complex128)
    with pytest.raises(ValueError, match="no blocks"):
        t.to_backend("numpy", dtype=np.complex128)
    assert t.to_backend("numpy").blocks == ()  # the dtype-free form still passes it through


def test_to_backend_without_dtype_is_unchanged():
    t = SymmetricTensor.random(SU2_LEGS, seed=7, dtype=np.float32)
    moved = t.to_backend("numpy")
    assert moved.dtype == np.float32
    assert moved == t


def test_to_backend_dtype_casts_after_the_move():
    t = SymmetricTensor.random(SU2_LEGS, seed=7, dtype=np.float32)
    c = t.to_backend("numpy", dtype=np.complex128)
    assert c.backend == "numpy"
    assert all(b.dtype == np.complex128 for b in c.blocks)
    for a, b in zip(t.blocks, c.blocks, strict=True):
        np.testing.assert_allclose(np.asarray(b).real, a, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(np.asarray(b).imag, np.zeros_like(a))


def test_to_backend_jax_with_dtype_keeps_float64_under_x64():
    """``conftest.py`` enables ``jax_enable_x64`` session-wide; see the no-x64 test below."""
    pytest.importorskip("jax")
    t = SymmetricTensor.random(SU2_LEGS, seed=8, dtype=np.float32)
    j = t.to_backend("jax", dtype=np.float64)
    assert j.backend == "jax"
    assert all(b.dtype == np.float64 for b in j.blocks)
    np.testing.assert_allclose(np.asarray(j.blocks[0]), t.blocks[0], rtol=0.0, atol=0.0)


# Run in a fresh interpreter: ``tests/conftest.py`` turns x64 on session-wide and
# JAX makes that switch irreversible, so the unset state is only reachable here.
_NO_X64_PROBE = """
import warnings
warnings.simplefilter("ignore")
import numpy as np
import jax
assert not jax.config.jax_enable_x64
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import SU2, SU2Sector
V = GradedSpace.new(SU2, {SU2Sector(1): 2})
legs = (Leg(V, OUT), Leg(V, IN))
t = SymmetricTensor.random(legs, seed=0)
print(t.to_backend("jax", dtype=np.float64).dtype,
      t.to_backend("jax", dtype=np.complex128).dtype)
"""


def test_to_backend_jax_dtype_cannot_defeat_a_disabled_x64():
    """Pins the sharp edge ``to_backend``'s Notes names, in a fresh process without x64.

    JAX has no per-array escape from ``jax_enable_x64``: ``astype(np.float64)``
    truncates to float32 exactly as ``array(...)`` does, so ``dtype=`` overrides
    the backend's *choice* of dtype and not its *refusal*.
    """
    pytest.importorskip("jax")
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    out = subprocess.run(
        [sys.executable, "-c", _NO_X64_PROBE], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.split() == ["float32", "complex64"]
