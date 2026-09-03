"""The public tensor type: a [TensorStructure][tenet.TensorStructure] plus its sector matrices.

Fields are ``(structure, data)``, never ``(legs, blocks)``: the structure is the
static, hashable half -- the JAX treedef -- and ``data`` is the clean parameter tree of
dynamic leaves, one dense matrix per coupled sector in ``map_layout(structure).sectors``
order (invariant 8).
[blocks][tenet.SymmetricTensor.blocks] keeps its meaning -- ``blocks[i]`` belongs to
``structure.block_order[i]`` -- and is the public *view* of that storage, cut out of the
matrices on demand and memoized. ``T.legs``, ``T.domain``, ``T.codomain``,
``T.block(key)`` and ``T.items()`` are derived views; ``from_blocks`` builds one from
public legs.

Why the storage is the matrices and not the blocks: every categorical operation is
already defined on the coupled-sector matrices -- composition is one matmul per sector,
``svd``/``qr``/``eigh`` one call per sector -- so holding the blocks meant gathering them
into matrices and cutting them back out around every operation. Holding the matrices
deletes that round trip. A block is then a two-dimensional slice reshaped, which is a
view on every backend tenet reaches, JAX included; a flat vector with a stride table --
TensorKit's storage -- would not be, since JAX has no ``as_strided``.

[to_dense][tenet.SymmetricTensor.to_dense] and [from_dense][tenet.SymmetricTensor.from_dense]
are the only way to cross into the dense basis — there is deliberately no ``__array__``
(invariant 9) — and both are thin delegations to ``tenet.ops.dense``, which
owns the layout convention, the plan cache and the only NumPy in the boundary.
That convention, fixed once and depended on downstream: axis ``i`` has length
``legs[i].space.dim``; sectors occupy contiguous slabs in the space's canonical
order; within sector ``a``'s slab the index is ``alpha * d_a + m``.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import autoray as ar
import numpy as np

from tenet.leg import OUT, Leg
from tenet.structure import FusionBlockKey, TensorStructure
from tenet.symmetry.base import _DualFusionRules

if TYPE_CHECKING:
    from types import ModuleType

    from tenet.map_view import TensorMapView

__all__ = ["SymmetricTensor"]

Array = Any
"""A backend array, whatever ``autoray`` dispatches on."""


# A FusionBlockKey reprs to ~200 characters, so a structure of any size cannot have
# its whole block_order in an exception message; name a few and say where the rest is.
_KEYS_IN_MESSAGE = 3


def _reject_foreign_keys(
    structure: TensorStructure, blocks: "Mapping[FusionBlockKey, Array]"
) -> None:
    """Raise naming the legal keys if any key of ``blocks`` is not in ``block_order``."""
    legal = structure.block_order
    foreign = [key for key in blocks if key not in set(legal)]
    if not foreign:
        return
    shown = "; ".join(map(str, legal[:_KEYS_IN_MESSAGE]))
    rest = "" if len(legal) <= _KEYS_IN_MESSAGE else f" (and {len(legal) - _KEYS_IN_MESSAGE} more)"
    raise KeyError(
        f"{len(foreign)} key(s) foreign to this structure, the first being {foreign[0]}. "
        f"The legal keys are TensorStructure(legs).block_order, which here begins "
        f"{shown}{rest}."
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SymmetricTensor:
    """A symmetric tensor: categorical structure plus one dense matrix per coupled sector.

    Parameters
    ----------
    structure : TensorStructure
        The static, hashable half: legs and everything derived from them.
    blocks : tuple of array
        One reduced block per key, in ``structure.block_order``, all sharing
        one dtype. They are checked and then gathered into the coupled-sector
        matrices the tensor stores; [blocks][tenet.SymmetricTensor.blocks] reads
        them back as views.

    Raises
    ------
    ValueError
        If the number of blocks does not match ``block_order``, a block's shape
        does not match ``structure.block_shape(key)``, or the blocks do not
        share one dtype.

    Examples
    --------
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> t.ndim, len(t.blocks)
    (2, 2)
    >>> t.shape, t.reduced_shape
    ((3, 3), (3, 3))
    >>> t.backend
    'numpy'
    """

    structure: TensorStructure
    _data: tuple[Array, ...] | None
    _views: tuple[Array, ...] | None

    def __init__(self, structure: TensorStructure, blocks: "Sequence[Array]") -> None:
        from tenet.map_view import gather

        blocks = tuple(blocks)
        order = structure.block_order
        if len(blocks) != len(order):
            raise ValueError(f"expected {len(order)} blocks, got {len(blocks)}")
        # ``block_shapes``, not ``block_shape(key)`` per block: this loop runs on every
        # tensor construction, and the per-key spelling would pay two structure-keyed
        # cache lookups -- each a deep hash -- for every block (#307).
        shapes = structure.block_shapes
        for i, (key, block) in enumerate(zip(order, blocks, strict=True)):
            expected = shapes[i]
            if tuple(block.shape) != expected:
                raise ValueError(
                    f"block {i} has shape {tuple(block.shape)}, expected {expected} for {key}"
                )
        dtypes = {block.dtype for block in blocks}
        if len(dtypes) > 1:
            raise ValueError(f"blocks must share one dtype, got {sorted(map(str, dtypes))}")
        object.__setattr__(self, "structure", structure)
        data, cut = gather(structure, blocks)
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_views", blocks if cut is None else cut)

    @classmethod
    def from_data(cls, structure: TensorStructure, data: "Sequence[Array]") -> "SymmetricTensor":
        """Wrap coupled-sector matrices as a tensor, unchecked and zero-copy.

        Parameters
        ----------
        structure : TensorStructure
            The structure the matrices belong to.
        data : sequence of array
            One matrix per coupled sector, in ``map_layout(structure).sectors`` order
            and of the shape that layout gives.

        Returns
        -------
        SymmetricTensor
            The tensor holding exactly those arrays.

        Notes
        -----
        The storage constructor, for a caller that already *has* the matrices --
        [from_matrices][tenet.from_matrices], which checks them, and the plan appliers,
        which built them from the structure they are being handed. The public
        constructor stays the trust boundary and keeps validating; this one takes the
        arrays as given, since re-deriving what the layout just produced would be
        checking our own arithmetic.
        """
        t = object.__new__(cls)
        object.__setattr__(t, "structure", structure)
        object.__setattr__(t, "_data", tuple(data))
        object.__setattr__(t, "_views", None)
        return t

    @property
    def data(self) -> tuple[Array, ...]:
        """One dense matrix per coupled sector, in ``map_layout(structure).sectors`` order.

        Returns
        -------
        tuple of array
            The tensor's coupled-sector matrices -- its storage, and its pytree leaves.

        Notes
        -----
        **Gathered on demand where gathering is not free.** On a mutable backend the
        constructor gathers immediately: it is one strided copy per block and it hands
        back every block of the result as a view into the matrix it just wrote, so the
        two forms cost one pass between them and a write through either reaches the
        other. On an immutable backend there is no memory to alias and no destination to
        write through, so a gather is a fresh array built by concatenation -- and under a
        JAX trace, one graph node per block with a backward pass of its own. There it is
        deferred until something actually asks for a matrix, which the pytree does at
        every traced boundary and a contraction does at every lowering, and which a
        transpose or an elementwise map never does at all.

        Either way the two forms hold the same values and the tensor is the same tensor;
        what the backend decides is only which of them is built first.
        """
        if self._data is None:
            from tenet.map_view import assemble

            # ``_views`` is what the constructor stored when it deferred the gather
            blocks: tuple[Array, ...] = self._views  # ty: ignore[invalid-assignment]
            object.__setattr__(self, "_data", assemble(self.structure, blocks))
        return self._data  # ty: ignore[invalid-return-type]  # set just above

    @property
    def blocks(self) -> tuple[Array, ...]:
        """One reduced block per key, in ``structure.block_order``. Views into ``data``.

        Returns
        -------
        tuple of array
            ``blocks[i]`` belongs to ``structure.block_order[i]``, in public axis order.

        Notes
        -----
        The storage contract, unchanged in meaning and changed in mechanism: each block
        is a two-dimensional slice of its coupled-sector matrix, reshaped into the
        block's axes and permuted back into public axis order. Nothing is copied, and
        writing into a block writes into the tensor. The cut is memoized, so a caller
        that reads ``blocks`` in a loop pays for it once.

        **A block is a live view and is usually not C-contiguous.** Two consequences for
        a caller that wants a block of its own. ``numpy.array(block, copy=True)`` and
        ``numpy.ascontiguousarray(block)`` are both the wrong tool: the first defaults to
        order ``"K"`` and keeps the block's layout, so ``.reshape(-1)`` on the result
        hands back another copy and a write into it is lost; the second does not copy at
        all when the block already happens to be contiguous, so a write into it reaches
        this tensor. ``numpy.array(block, order="C", copy=True)`` is the one that is both
        a copy and flat.
        """
        if self._views is None:
            from tenet.map_view import views

            object.__setattr__(self, "_views", views(self.structure, self.data))
        return self._views  # ty: ignore[invalid-return-type]  # set just above

    # --- constructors ---------------------------------------------------------

    @classmethod
    def from_blocks(
        cls, legs: Sequence[Leg], blocks: "Mapping[FusionBlockKey, Array]"
    ) -> "SymmetricTensor":
        """Build from public legs by **naming** fusion-block keys; absent keys are zero.

        Only the blocks the caller has an opinion about are named; the keys come from
        ``TensorStructure(legs).block_order`` — no throwaway tensor is needed to
        discover the layout.

        Parameters
        ----------
        legs : sequence of Leg
            The legs, in public axis order.
        blocks : mapping of FusionBlockKey to array
            The blocks to set. Every key must belong to
            ``TensorStructure(legs).block_order``; keys left out are filled with
            zeros of the supplied blocks' dtype and backend.

        Returns
        -------
        SymmetricTensor
            The assembled tensor; the constructor validates every block's shape.

        Raises
        ------
        KeyError
            If a key is foreign to the structure. The message names the legal
            keys (the first few, plus the count) and where to read the rest.
        ValueError
            If ``blocks`` is empty — the zero fill has no dtype or backend to
            take from. [zeros][tenet.SymmetricTensor.zeros] is that tensor.
            Also if a supplied block has the wrong shape, from the ordinary
            constructor, naming the expected shape and the key.

        Notes
        -----
        **An absent key is zero, not an error.** The convenience is not the
        argument — strictness would be worth the inconvenience if it caught
        mistakes, and here it does not: a mistyped key is an *unknown* key, not
        a missing one, so it raises either way. Demanding every key would only
        penalise the case the constructor exists for, which is naming one block
        of many.

        ``docs/tutorials/symmetric-tensors.md`` is this constructor at length:
        reading a key, the U(1) and SU(3) operators built with it, and what the
        foreign-key refusal catches.

        Examples
        --------
        >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
        >>> from tenet.symmetry import SU2, SU2Sector
        >>> V = GradedSpace.new(SU2, {SU2Sector(1): 1})       # one spin-1/2 multiplet
        >>> legs = (Leg(V, OUT), Leg(V, OUT, dual=True))      # the evaluation cup
        >>> structure = TensorStructure(legs)
        >>> key, = structure.block_order                      # exactly one fusion channel
        >>> t = SymmetricTensor.from_blocks(legs, {key: np.ones(structure.block_shape(key))})
        >>> t.blocks
        (array([[1.]]),)
        """
        structure = TensorStructure(tuple(legs))
        _reject_foreign_keys(structure, blocks)
        if not blocks:
            raise ValueError(
                "from_blocks needs at least one block to take a dtype and backend from; "
                "for an all-zero tensor use SymmetricTensor.zeros(legs, dtype)"
            )
        ref = next(iter(blocks.values()))
        return cls(
            structure,
            tuple(
                blocks[key] if key in blocks else ar.do("zeros", shape, dtype=ref.dtype, like=ref)
                for key, shape in zip(structure.block_order, structure.block_shapes, strict=True)
            ),
        )

    @classmethod
    def zeros(cls, legs: Sequence[Leg], dtype: Any = np.float64) -> "SymmetricTensor":
        """All-zero blocks over ``legs``.

        Parameters
        ----------
        legs : sequence of Leg
            The legs, in public axis order.
        dtype : dtype, optional
            The blocks' dtype. Default ``np.float64``.

        Returns
        -------
        SymmetricTensor
            The zero tensor, NumPy blocks.
        """
        structure = TensorStructure(tuple(legs))
        return cls(
            structure,
            tuple(np.zeros(shape, dtype) for shape in structure.block_shapes),
        )

    @classmethod
    def random(
        cls, legs: Sequence[Leg], *, seed: int | None = None, dtype: Any = np.float64
    ) -> "SymmetricTensor":
        """Standard-normal blocks from ``np.random.default_rng(seed)``, reproducible.

        Parameters
        ----------
        legs : sequence of Leg
            The legs, in public axis order.
        seed : int or None, optional
            The RNG seed; ``None`` (the default) draws fresh entropy.
        dtype : dtype, optional
            The blocks' dtype. Default ``np.float64``.

        Returns
        -------
        SymmetricTensor
            The random tensor, NumPy blocks.
        """
        structure = TensorStructure(tuple(legs))
        rng = np.random.default_rng(seed)
        # Simplification: real draws cast to dtype; give complex dtypes a real+imag draw
        # if a test ever needs genuinely complex random data.
        return cls(
            structure,
            tuple(rng.standard_normal(shape).astype(dtype) for shape in structure.block_shapes),
        )

    # --- derived views --------------------------------------------------------

    @property
    def legs(self) -> tuple[Leg, ...]:
        """The structure's legs, in public axis order.

        Returns
        -------
        tuple of Leg
            ``self.structure.legs``.
        """
        return self.structure.legs

    @property
    def ndim(self) -> int:
        """Number of legs.

        Returns
        -------
        int
            ``self.structure.ndim``.
        """
        return self.structure.ndim

    @property
    def provider(self) -> _DualFusionRules:
        """The legs' shared symmetry provider.

        Returns
        -------
        provider
            ``self.structure.provider``.
        """
        return self.structure.provider

    @property
    def codomain(self) -> tuple[Leg, ...]:
        """The OUT legs, in public axis order.

        Returns
        -------
        tuple of Leg
            The OUT legs. [ProductSpace][tenet.ProductSpace] is the fused view of
            the same legs, when one is wanted.
        """
        return tuple(leg for leg in self.legs if leg.side is OUT)

    @property
    def domain(self) -> tuple[Leg, ...]:
        """IN legs in public axis order.

        Returns
        -------
        tuple of Leg
            The IN legs.
        """
        return tuple(leg for leg in self.legs if leg.side is not OUT)

    def block(self, key: FusionBlockKey) -> Array:
        """The stored block for ``key`` — the array itself, not a copy.

        Parameters
        ----------
        key : FusionBlockKey
            A key of this tensor's structure.

        Returns
        -------
        array
            The block, in public axis order.

        Raises
        ------
        KeyError
            If ``key`` is foreign to the structure.
        """
        return self.blocks[self.structure.index_of(key)]

    def items(self) -> Iterator[tuple[FusionBlockKey, Array]]:
        """Iterate ``(key, block)`` pairs in ``block_order``.

        Yields
        ------
        tuple of (FusionBlockKey, array)
            Each key with its stored block.
        """
        return zip(self.structure.block_order, self.blocks, strict=True)

    def with_blocks(self, blocks: "Mapping[FusionBlockKey, Array]") -> "SymmetricTensor":
        """Same structure, the named blocks replaced and the rest carried over.

        The immutable spelling of assigning to one block: ``self`` is untouched
        and a new tensor is returned. The keys are this tensor's own, from
        ``self.structure.block_order`` or [items][tenet.SymmetricTensor.items].

        Parameters
        ----------
        blocks : mapping of FusionBlockKey to array
            The blocks to replace. Keys absent from the mapping keep the block
            they already have; an empty mapping is a no-op copy.

        Returns
        -------
        SymmetricTensor
            A new tensor over the same structure.

        Raises
        ------
        KeyError
            If a key is foreign to this tensor's structure — the same message
            [from_blocks][tenet.SymmetricTensor.from_blocks] raises.
        ValueError
            From the ordinary constructor, if a replacement has the wrong shape
            (naming the expected shape and the key) or a dtype the other blocks
            do not share.

        Examples
        --------
        >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
        >>> from tenet.symmetry import U1, U1Sector
        >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
        >>> t = SymmetricTensor.zeros((Leg(V, OUT), Leg(V, IN)))
        >>> key = t.structure.block_order[0]
        >>> u = t.with_blocks({key: np.ones(t.structure.block_shape(key))})
        >>> u.block(key)
        array([[1., 1.],
               [1., 1.]])
        >>> t.block(key).any()          # the original is untouched
        np.False_
        """
        _reject_foreign_keys(self.structure, blocks)
        return SymmetricTensor(
            self.structure, tuple(blocks.get(key, block) for key, block in self.items())
        )

    # --- array-style properties -----------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Full **physical** dimension per public axis: ``Σ_a m_a d_a``.

        Equal to ``self.to_dense().shape``.

        Returns
        -------
        tuple of int
            One dense dimension per public axis.

        Raises
        ------
        CapabilityError
            If the provider lacks ``ClebschGordanData`` (via ``GradedSpace.dim``)
            — a provider with non-integer quantum dimensions has no physical
            shape, and silently returning
            [reduced_shape][tenet.SymmetricTensor.reduced_shape] would violate
            invariant 11.
        """
        return tuple(leg.space.dim for leg in self.legs)

    @property
    def reduced_shape(self) -> tuple[int, ...]:
        """Degeneracy dimension per public axis: ``Σ_a m_a``. Any provider.

        The storage-facing shape: what the reduced blocks are made of.

        Returns
        -------
        tuple of int
            One degeneracy dimension per public axis.
        """
        return tuple(leg.space.reduced_dim for leg in self.legs)

    @property
    def dtype(self) -> Any:
        """The single dtype shared by all blocks (``__post_init__`` validates it).

        Returns
        -------
        dtype
            The first block's dtype.

        Raises
        ------
        ValueError
            If the tensor has no blocks — the dtype is then undefined.
        """
        return self._first_block().dtype

    @property
    def backend(self) -> str:
        """``"numpy"`` / ``"jax"`` / ``"torch"``, inferred from the first block.

        Returns
        -------
        str
            The autoray backend name.

        Raises
        ------
        ValueError
            If the tensor has no blocks — the backend is then undefined.

        Notes
        -----
        One tensor uses one backend; construction does not re-check every block,
        since ``to_backend`` is the only sanctioned way to move them.
        """
        return ar.infer_backend(self._first_block())

    @property
    def device(self) -> Any:
        """The first block's own ``.device`` (``None`` if it has none).

        Returns
        -------
        device or None
            Whatever the backend exposes.

        Raises
        ------
        ValueError
            If the tensor has no blocks — the device is then undefined.

        Notes
        -----
        A plain ``getattr``: autoray exposes no portable device accessor, and
        NumPy >= 2 arrays already carry ``.device == 'cpu'``.
        """
        return getattr(self._first_block(), "device", None)

    def _first_block(self) -> Array:
        """The first coupled-sector matrix -- what dtype/backend/device are read off.

        A block and its matrix share all three, and the matrix is there without cutting
        the views out, so the storage is what these questions are asked of.
        """
        # whichever form the tensor already holds: a block and its matrix share dtype,
        # backend and device, and forcing the other one to read a dtype would be a
        # gather (or a cut) for nothing
        held = self._views if self._data is None else self._data
        if not held:
            raise ValueError(
                "tensor has no blocks: dtype/backend/device are undefined "
                f"(structure with legs {self.legs})"
            )
        return held[0]

    def astype(self, dtype: Any) -> "SymmetricTensor":
        """Same structure and backend, every block cast to ``dtype``.

        Parameters
        ----------
        dtype : dtype
            The target dtype. Any spelling NumPy recognizes — ``np.complex128``,
            ``"complex128"``, ``np.dtype("complex128")`` — means the same thing on
            every backend; a backend-native dtype object (``torch.complex128``)
            is passed through untouched.

        Returns
        -------
        SymmetricTensor
            A new tensor whose blocks all carry ``dtype``. ``self`` is untouched.

        Raises
        ------
        ValueError
            If the tensor has no blocks — there is nothing to cast, and the
            result's dtype would be undefined, as for
            [dtype][tenet.SymmetricTensor.dtype].

        Notes
        -----
        Blockwise ``ar.do("astype", b, dtype)``, so the backend's own casting
        rules apply: JAX truncates a request for a dtype its ``jax_enable_x64``
        setting does not admit, exactly as it does on any other array.

        Examples
        --------
        >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
        >>> from tenet.symmetry import U1, U1Sector
        >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
        >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
        >>> c = t.astype(np.complex128)
        >>> c.dtype
        dtype('complex128')
        >>> c.structure == t.structure and c.legs == t.legs
        True
        >>> bool(np.allclose(np.asarray(c.blocks[0]).real, t.blocks[0]))
        True
        """
        self._first_block()  # the empty tensor has no dtype to cast to; same refusal
        # autoray's torch `astype` routes the dtype through `to_backend_dtype`, which
        # wants the *name*; NumPy and JAX take the name too, so normalizing every
        # NumPy-recognized spelling to it is what makes `astype(np.complex128)` mean
        # one thing on all three backends. A backend-native dtype object is left alone.
        try:
            dtype = np.dtype(dtype).name
        except TypeError:
            pass
        return SymmetricTensor.from_data(
            self.structure, tuple(ar.do("astype", m, dtype) for m in self.data)
        )

    def to_backend(self, backend: str, dtype: Any = None) -> "SymmetricTensor":
        """Same structure, blocks converted with ``ar.do("array", b, like=backend)``.

        Parameters
        ----------
        backend : str
            The target autoray backend, e.g. ``"jax"``.
        dtype : dtype or None, optional
            Cast the blocks to this dtype **after** the move, via
            [astype][tenet.SymmetricTensor.astype]. ``None`` (the default) is
            today's behaviour: whatever dtype the backend chose is kept.

        Returns
        -------
        SymmetricTensor
            A new tensor on ``backend``.

        Notes
        -----
        A move to the backend the tensor already lives on is the identity on the
        blocks — no copy, and, on torch, no ``torch.tensor(x)``, which would
        detach them from the autograd graph. An explicit ``dtype`` still runs.

        The target backend's own dtype policy applies to the move (JAX demotes
        float64 to float32 unless ``jax_enable_x64`` is set). ``dtype`` runs
        after that move rather than as part of it, so it overrides a backend's
        *choice* — e.g. a real tensor moved to JAX and asked for
        ``np.complex128`` arrives complex — but it cannot override a backend's
        *refusal*: JAX truncates ``np.float64`` to float32 in ``astype`` too
        when ``jax_enable_x64`` is unset, because in that mode the dtype does
        not exist for it to produce. Enabling ``jax_enable_x64`` is the only fix
        for that one, and it is process-global.

        Examples
        --------
        >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
        >>> from tenet.symmetry import U1, U1Sector
        >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
        >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
        >>> t.to_backend("numpy", dtype=np.complex128).dtype
        dtype('complex128')
        """
        if self.data and self.backend == backend:
            # a move to where the tensor already is: no copy, and on torch no
            # `torch.tensor(x)`, which would detach the blocks from the autograd
            # graph. `backend` needs a block to be read off, hence `self.data`.
            moved = self
        else:
            moved = SymmetricTensor.from_data(
                self.structure, tuple(ar.do("array", m, like=backend) for m in self.data)
            )
        return moved if dtype is None else moved.astype(dtype)

    # --- parameter protocol (quimb / autoray) ---------------------------------

    def get_params(self) -> tuple[Array, ...]:
        """The coupled-sector matrices — a pytree of backend arrays. See ``data``.

        Returns
        -------
        tuple of array
            ``self.data``, the identity.

        Notes
        -----
        The identity, deliberately: ``data`` is an ordered tuple, so no dict ordering
        and no key hashing ever enters the dynamic data.

        These are the same leaves [tenet.pytree][] hands JAX, and for the same reason:
        they are the arrays the tensor is made of. A block is a *view* cut out of one of
        them, so a torch leaf handed back as a block would be a non-leaf with no ``.grad``
        of its own, and a JAX leaf handed back as a block would put the cut in the graph.
        The optimizer differentiates the storage; ``blocks`` is how the result is read.
        """
        return self.data

    def set_params(self, params: Sequence[Array]) -> "SymmetricTensor":
        """Same structure, new numerical data. A **new** tensor; ``self`` is untouched.

        Parameters
        ----------
        params : sequence of array
            The new coupled-sector matrices, in the order
            [get_params][tenet.SymmetricTensor.get_params] returns them.

        Returns
        -------
        SymmetricTensor
            A new tensor over the same structure.

        Raises
        ------
        ValueError
            If the count, a shape or the dtypes do not match the structure's layout --
            [from_matrices][tenet.from_matrices]'s refusals, which are the trust
            boundary for arrays given as matrices.

        """
        # Simplification: quimb's ``inject_variables`` may expect this to mutate in place.
        # quimb is not a dependency, so this is not guessed at here — the frozen structure
        # is what the JAX story rests on. If M8 finds quimb needs it, a thin mutable
        # adapter belongs there, not in the core type.
        from tenet.map_view import from_matrices, map_layout

        sectors = map_layout(self.structure).sectors
        return from_matrices(self.structure, dict(zip(sectors, params, strict=True)))

    def copy(self) -> "SymmetricTensor":
        """A new instance sharing the same structure and stored arrays.

        Returns
        -------
        SymmetricTensor
            The shallow copy.
        """
        return SymmetricTensor.from_data(self.structure, self.data)

    # --- value semantics ------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Exact equality: same ``structure`` and every stored matrix exactly equal.

        The matrices carry exactly the blocks, cell for cell and with no cell left over,
        so this is block equality read off the storage.

        Never raises on a structure mismatch. Under a JAX trace ``bool()`` of a
        traced comparison does raise, correctly: ``==`` is a concrete-value
        question (invariants 9/10).
        """
        if not isinstance(other, SymmetricTensor):
            return NotImplemented
        if self.structure != other.structure:
            return False
        return all(bool(ar.do("all", a == b)) for a, b in zip(self.data, other.data, strict=True))

    __hash__ = None  # type: ignore[assignment]  # holds arrays; T.structure is the hashable half

    # --- arithmetic -----------------------------------------------------------
    # One-line wrappers over tenet.ops.basic, with function-local imports so the
    # dependency edge stays one-way (ops -> tensor). No __iadd__: the tensor is
    # frozen and Python's rebinding fallback is the intended behaviour.

    def _ops(self) -> "ModuleType":
        from tenet.ops import basic

        return basic

    def __add__(self, other: "SymmetricTensor") -> "SymmetricTensor":
        if not isinstance(other, SymmetricTensor):
            raise TypeError(
                f"cannot add {type(other).__name__} to a SymmetricTensor: adding a scalar "
                "is not equivariant and would break the symmetry"
            )
        return self._ops().add(self, other)

    def __sub__(self, other: "SymmetricTensor") -> "SymmetricTensor":
        if not isinstance(other, SymmetricTensor):
            raise TypeError(
                f"cannot subtract {type(other).__name__} from a SymmetricTensor: "
                "scalar shifts are not equivariant"
            )
        return self._ops().subtract(self, other)

    def __neg__(self) -> "SymmetricTensor":
        return self._ops().negative(self)

    def __mul__(self, s: Any) -> "SymmetricTensor":
        """Scalar multiplication only; a ``SymmetricTensor`` operand is a ``TypeError``."""
        return self._ops().multiply(self, s)

    __rmul__ = __mul__

    def __truediv__(self, s: Any) -> "SymmetricTensor":
        return self._ops().divide(self, s)

    def __matmul__(self, other: "SymmetricTensor") -> "SymmetricTensor":
        """Morphism composition ``a ∘ b``. See [tenet.compose][]."""
        if not isinstance(other, SymmetricTensor):
            raise TypeError(
                f"@ composes two SymmetricTensors, got {type(other).__name__}: "
                "for scalar multiplication write `t * s`"
            )
        from tenet.ops import map as map_ops

        return map_ops.compose(self, other)

    # --- map view -------------------------------------------------------------

    def as_map(self) -> "TensorMapView":
        """View this tensor as a morphism. Zero-copy.

        Returns
        -------
        TensorMapView
            The semantic view; see [TensorMapView][tenet.TensorMapView].
        """
        from tenet.map_view import as_map

        return as_map(self)

    def conj(self) -> "SymmetricTensor":
        """Conjugate the blocks; ``legs`` unchanged. See [tenet.conj][].

        Returns
        -------
        SymmetricTensor
            The blockwise complex conjugate.
        """
        return self._ops().conj(self)

    def adjoint(self) -> "SymmetricTensor":
        """``T†``: every leg's ``side`` flips, blocks are conjugated and key-swapped.

        Returns
        -------
        SymmetricTensor
            The adjoint. Not ``conj()`` (which touches no leg) and not a
            dualization. See [tenet.adjoint][].
        """
        from tenet.ops import map as map_ops

        return map_ops.adjoint(self)

    def norm(self) -> Any:
        """qdim-weighted Frobenius norm (a backend scalar). See [tenet.norm][].

        Returns
        -------
        scalar
            The norm, on this tensor's backend.
        """
        return self._ops().norm(self)

    # --- elementwise block maps (coefficient space, not dense space) -----------

    def apply_blocks(self, fn: Any) -> "SymmetricTensor":
        """``fn`` on each reduced block. See [tenet.apply_blocks][] for the caveat.

        Parameters
        ----------
        fn : callable
            Applied to each block; must preserve shape and be backend-generic.

        Returns
        -------
        SymmetricTensor
            The mapped tensor, same structure.
        """
        from tenet.ops import blocks

        return blocks.apply_blocks(self, fn)

    def block_sqrt(self) -> "SymmetricTensor":
        """Blockwise ``sqrt`` — *not* ``sqrt(self.to_dense())``. See [tenet.block_sqrt][].

        Returns
        -------
        SymmetricTensor
            The blockwise square root.
        """
        from tenet.ops import blocks

        return blocks.block_sqrt(self)

    def block_power(self, p: Any) -> "SymmetricTensor":
        """Blockwise ``self ** p`` for a scalar ``p``. See [tenet.block_power][].

        Parameters
        ----------
        p : scalar
            The exponent.

        Returns
        -------
        SymmetricTensor
            The blockwise power.
        """
        from tenet.ops import blocks

        return blocks.block_power(self, p)

    # --- serialization --------------------------------------------------------

    def save(self, path: Any, *, compress: bool = False) -> None:
        """Write to ``path`` as a single ``.npz``. See [tenet.save][].

        Parameters
        ----------
        path : path-like
            Where to write.
        compress : bool, optional
            Compress the archive. Default ``False``.
        """
        from tenet.serialize import save

        save(self, path, compress=compress)

    @classmethod
    def load(cls, path: Any) -> "SymmetricTensor":
        """Read a file written by [save][tenet.SymmetricTensor.save]; NumPy blocks.

        See [tenet.load][].

        Parameters
        ----------
        path : path-like
            The ``.npz`` file to read.

        Returns
        -------
        SymmetricTensor
            The loaded tensor, NumPy blocks.
        """
        from tenet.serialize import load

        return load(path)

    def transpose(self, *axes: Any) -> "SymmetricTensor":
        """``T.transpose(2, 0, 1)``, ``T.transpose((2, 0, 1))`` or ``T.transpose()``.

        Parameters
        ----------
        *axes : int or sequence of int
            The new axis order; empty (or ``None``) reverses all axes (NumPy
            convention).

        Returns
        -------
        SymmetricTensor
            The permuted tensor; see [tenet.transpose][] — no leg changes
            ``side``.
        """
        from tenet.ops import permutation

        if len(axes) == 1 and (axes[0] is None or isinstance(axes[0], Sequence)):
            axes = tuple(axes[0] or ())
        return permutation.transpose(self, axes or None)

    def repartition(self, outputs: Sequence[int], inputs: Sequence[int]) -> "SymmetricTensor":
        """``T.repartition(outputs=(0, 1), inputs=(2,))``. See ``tenet.repartition``.

        Parameters
        ----------
        outputs : sequence of int
            The public axes to place on the OUT side.
        inputs : sequence of int
            The public axes to place on the IN side.

        Returns
        -------
        SymmetricTensor
            The repartitioned tensor.

        Notes
        -----
        Every leg that crosses sides is *bent*: its ``side`` and its ``dual`` both
        flip. Requires ``BendingCoefficients`` unless no leg crosses.
        """
        from tenet.ops.repartition import repartition

        return repartition(self, outputs, inputs)

    # --- fusion ---------------------------------------------------------------

    def fuse(self, *axes: Any) -> "SymmetricTensor":
        """``T.fuse(0, 1)`` or ``T.fuse((0, 1))``. See [tenet.fuse][].

        Parameters
        ----------
        *axes : int or sequence of int
            The adjacent axes to fuse into one.

        Returns
        -------
        SymmetricTensor
            The fused tensor.
        """
        from tenet.ops import fusion

        if len(axes) == 1 and not isinstance(axes[0], int):
            axes = tuple(axes[0])
        return fusion.fuse(self, axes)

    def unfuse(self, axis: int, legs: Sequence[Leg]) -> "SymmetricTensor":
        """Split ``axis`` into ``legs``. See [tenet.unfuse][].

        Parameters
        ----------
        axis : int
            The fused axis to split.
        legs : sequence of Leg
            The constituent legs the axis splits into.

        Returns
        -------
        SymmetricTensor
            The unfused tensor.
        """
        from tenet.ops import fusion

        return fusion.unfuse(self, axis, legs)

    def embed(self, legs: Sequence[Leg]) -> "SymmetricTensor":
        """Zero-pad into larger, containing legs. See ``tenet.embed``.

        Parameters
        ----------
        legs : sequence of Leg
            The target legs, one per axis, each containing the current leg.

        Returns
        -------
        SymmetricTensor
            The embedded tensor.
        """
        from tenet.ops.embed import embed

        return embed(self, legs)

    def to_symmetry(
        self, target: _DualFusionRules, *, atol: float | None = None
    ) -> "SymmetricTensor":
        """Restrict to a smaller symmetry, e.g. SU(2) -> U(1). See ``tenet.to_symmetry``.

        Parameters
        ----------
        target : provider
            The smaller symmetry's provider.
        atol : float or None, optional
            Symmetry-check tolerance; ``None`` (the default) uses the default.

        Returns
        -------
        SymmetricTensor
            The tensor over ``target``.
        """
        from tenet.ops.cast import to_symmetry

        return to_symmetry(self, target, atol=atol)

    def restrict(self, legs: Sequence[Leg], *, atol: float | None = None) -> "SymmetricTensor":
        """Slice down to smaller, contained legs. See [tenet.restrict][].

        Parameters
        ----------
        legs : sequence of Leg
            The target legs, one per axis, each contained in the current leg.
        atol : float or None, optional
            Tolerance for the discarded weight check; ``None`` (the default)
            skips it.

        Returns
        -------
        SymmetricTensor
            The restricted tensor.
        """
        from tenet.ops.embed import restrict

        return restrict(self, legs, atol=atol)

    def direct_sum(self, other: "SymmetricTensor", axes: int | Sequence[int]) -> "SymmetricTensor":
        """``self ⊕ other`` along ``axes``. See [tenet.direct_sum][].

        Parameters
        ----------
        other : SymmetricTensor
            The other summand.
        axes : int or sequence of int
            The axes along which the spaces are summed.

        Returns
        -------
        SymmetricTensor
            The direct sum.
        """
        from tenet.ops.embed import direct_sum

        return direct_sum(self, other, axes)

    def __repr__(self) -> str:
        def safe(get: Callable[[], Any]) -> Any:
            try:
                return get()
            except Exception:
                return "?"

        return (
            f"SymmetricTensor(ndim={self.ndim}, shape={safe(lambda: self.shape)}, "
            f"dtype={safe(lambda: self.dtype)}, backend={safe(lambda: self.backend)!r}, "
            f"blocks={self.structure.num_blocks})"
        )

    # --- dense expansion ------------------------------------------------------

    def to_dense(self) -> Array:
        """``T = Σ_τ A^(τ) ⊗ C^(τ)`` expanded into a dense array of ``self``'s backend.

        Returns
        -------
        array
            The dense carrier-basis array, shape
            [shape][tenet.SymmetricTensor.shape].

        Raises
        ------
        CapabilityError
            If the provider lacks ``ClebschGordanData``; a leg with
            ``dual=True`` additionally requires ``DualBasis``.

        Notes
        -----
        Explicit by design (invariant 9). See
        ``tenet.ops.dense.to_dense`` — traceable and differentiable.
        """
        from tenet.ops.dense import to_dense

        return to_dense(self)

    @classmethod
    def from_dense(
        cls, dense: Array, legs: Sequence[Leg], *, atol: float | None = None
    ) -> "SymmetricTensor":
        """Project a dense carrier-basis array onto the symmetric subspace of ``legs``.

        Parameters
        ----------
        dense : array
            The dense array, in the layout convention of the module docstring.
        legs : sequence of Leg
            The legs describing each dense axis.
        atol : float or None, optional
            Tolerance for the symmetry check; ``None`` (the default) uses the
            default.

        Returns
        -------
        SymmetricTensor
            The projected tensor.

        Notes
        -----
        The inverse of [to_dense][tenet.SymmetricTensor.to_dense]; non-symmetric
        input is refused rather than silently projected. See
        ``tenet.ops.dense.from_dense``.
        """
        from tenet.ops.dense import from_dense

        return from_dense(dense, legs, atol=atol)


def _unchecked(structure: TensorStructure, blocks: tuple[Array, ...]) -> SymmetricTensor:
    """Assemble a tensor **without** running ``__post_init__``'s checks.

    Private to ``tenet.ops`` and ``tenet.map_view``, and deliberately not
    exported: it is the one construction path that trusts its caller.

    The blocks are still *gathered* into the coupled-sector matrices the tensor
    stores -- that is the storage, not a check --
    and [from_data][tenet.SymmetricTensor.from_data] is the door for a caller
    that already has the matrices and needs no gathering either.

    Parameters
    ----------
    structure : TensorStructure
        The structure the blocks were computed against.
    blocks : tuple of array
        One block per key of ``structure.block_order``, already a ``tuple``,
        each of shape ``structure.block_shapes[i]``, all of one dtype. Nothing
        here checks any of that.

    Returns
    -------
    SymmetricTensor
        The tensor, byte for byte what the ordinary constructor would have
        built from the same arguments.

    Notes
    -----
    ``__post_init__`` is the trust boundary -- it is what makes
    ``SymmetricTensor(structure, blocks)`` safe to hand arbitrary arrays to, and
    it stays exactly as it is. But a caller that *derived* its blocks from
    ``structure``'s own tables is not untrusted input; it is us, re-checking our
    own arithmetic once per block. On a rank-8 SU(2) intermediate of 613,468
    blocks that re-check is a fifth of the contraction it sits in. So the
    boundary stops being crossed internally rather than being weakened, and the
    check is never conditioned on how the interpreter was started -- ``python -O``
    would switch off the boundary too, which is the copy of the check that
    matters.

    Use this only where the shapes provably came from the same ``structure``:
    a plan applier writing one block per key of the structure the plan was built
    from, or an elementwise map over an already-validated tensor's blocks. A
    caller that lets a *user* function decide a block's shape, or that assumes
    two structures enumerate their keys in the same order, is not such a place
    and must keep the ordinary constructor.

    **The dtype check goes too, and is not offered as a separate entry point.**
    It is the cheaper of the two -- one pass building ``{block.dtype for block in
    blocks}`` against the shape loop's tuple comparison per block, about 7.5 ms
    against 27 ms at 613,468 blocks -- but it is cheaper by a constant, not by a
    shape: it is still one touch per block. It *is* separable, though, and where
    it is still wanted it separates in a better direction than a second
    constructor would give it. ``map_view.from_matrices`` takes user matrices and
    hands out blocks that are views into them, so a dtype disagreement there is a
    real refusal and a shape disagreement is not; it keeps the dtype check and
    spells it over the *matrices*, one touch per coupled sector instead of one per
    block. That is the general shape of the answer -- check the untrusted thing,
    which is never the block list -- and it is why no trusted caller needs a
    dtype-only door into this function.
    """
    from tenet.map_view import gather

    data, cut = gather(structure, blocks)
    t = object.__new__(SymmetricTensor)
    object.__setattr__(t, "structure", structure)
    object.__setattr__(t, "_data", data)
    object.__setattr__(t, "_views", cut)
    return t


def _relabelled(t: SymmetricTensor, structure: TensorStructure) -> SymmetricTensor:
    """``t``'s storage under a different but layout-identical ``structure``.

    Private to ``tenet.ops``, like [_unchecked][tenet.tensor._unchecked], and trusting
    its caller the same way.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor whose storage is reused. Untouched.
    structure : TensorStructure
        A structure with the same ``block_order`` and the same
        [map_layout][tenet.map_layout] bands as ``t.structure``. Nothing here checks
        that.

    Returns
    -------
    SymmetricTensor
        The same arrays under ``structure``.

    Notes
    -----
    Some operations change only what a leg is *called*: ``tenet.flip_dual`` on a grading
    whose Frobenius-Schur indicator and twist are both 1 relabels every sector through
    ``provider.dual`` and moves no element. Routing that through either constructor would
    make the tensor commit to one of the two forms -- ``from_data`` forces a gather that
    on an immutable backend is a fresh array per block, ``_unchecked`` forces a cut --
    for an operation with nothing to compute. Carrying both fields over leaves the
    tensor in whichever form it already had, so a relabel costs one object.
    """
    relabelled = object.__new__(SymmetricTensor)
    object.__setattr__(relabelled, "structure", structure)
    object.__setattr__(relabelled, "_data", t._data)
    object.__setattr__(relabelled, "_views", t._views)
    return relabelled
