# symtenet

[![tests](https://github.com/Ryo-wtnb11/symtenet/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryo-wtnb11/symtenet/actions/workflows/ci.yml)
[![coverage](https://ryo-wtnb11.github.io/symtenet/coverage-badge.svg)](https://github.com/Ryo-wtnb11/symtenet/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Ryo-wtnb11/symtenet/graph/badge.svg)](https://codecov.io/gh/Ryo-wtnb11/symtenet)
[![docs](https://img.shields.io/badge/docs-online-blue.svg)](https://ryo-wtnb11.github.io/symtenet/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/Ryo-wtnb11/symtenet/blob/main/LICENSE)
[![python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](https://github.com/Ryo-wtnb11/symtenet/blob/main/pyproject.toml)

**A Python library for symmetric tensor networks: NumPy-ish API on the surface, category theory under the hood.**

`tenet` gives you block-sparse tensors that carry a symmetry exactly — SU(N), SU(2),
U(1), Z2, fermion parity and their products — through exact recoupling coefficients.
A tensor carries legs, contracts with `tenet.einsum` and factorizes with `tenet.linalg`,
the way an ndarray does. Blocks live in NumPy, JAX or PyTorch arrays through
[`autoray`](https://github.com/jcmgray/autoray), so the same tensor runs under
`jax.jit` and `jax.grad` with its symmetry structure intact. On top of the tensor layer,
`tenet.network` ships finite DMRG and CTMRG.

## Acknowledgments and upstream work

symtenet builds on the mathematical and implementation work of
[TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl), developed by
Lukas Devos, Jutho Haegeman, and contributors. TensorKit provides the primary
foundation for its tensor-map semantics, fusion-tree bases, duality, and
non-Abelian block linear algebra. Its architecture also draws on
[TeNeT](https://github.com/Ryo-wtnb11/TeNeT), whose reduced-block design and
execution approach are informed by
[QSpace](https://bitbucket.org/qspace4u/), developed by Andreas Weichselbaum
and contributors. [symmray](https://github.com/jcmgray/symmray), developed by
Johnnie Gray and contributors, provides an important reference for the
NumPy-style tensor interface and backend-native blocks through `autoray`.
We gratefully acknowledge these mathematical, architectural, and implementation
foundations. The [design document](docs/design.md) describes the roles of these
and other reference libraries.

**If you use symtenet in research, please cite TensorKit, QSpace, and symmray,
as well as symtenet.** See [Citation](#citation) below.

## Install

```sh
uv add symtenet         # or: pip install symtenet
```

The core install pulls `numpy`, `autoray`, `opt-einsum` and `racah-py`, and every
symmetry works on it. Two optional extras:

```sh
uv add "symtenet[jax]"      # jax>=0.10 — pytrees, jit, grad
uv add "symtenet[torch]"    # torch>=2.0 — eager blocks
```

## Quickstart

A 20-site spin-1/2 Heisenberg chain, U(1)-graded by `2 S^z`, to its ground state:

```python
from tenet.models import spin_half
from tenet.network import MPO, MPS, dmrg_
from tenet.symmetry import U1Sector

site, n = spin_half(), 20
terms = []
for i in range(n - 1):
    terms.append((1.0, [(site.ops["Sz"], i), (site.ops["Sz"], i + 1)]))
    terms.append((0.5, [(site.ops["S+"], i), (site.ops["S-"], i + 1)]))
    terms.append((0.5, [(site.ops["S-"], i), (site.ops["S+"], i + 1)]))

h = MPO.from_terms(n, terms)
psi = MPS.product(site.phys, [U1Sector(1 if i % 2 else -1) for i in range(n)])
out = dmrg_(psi, h, chi=64)

print(out.sweeps, out.energy)          # 6 -8.682473334398...
print(out.psi.entanglement_entropy())  # {bond: S}, in nats
```

The Néel product state's own charges put the run in the `S^z_tot = 0` sector, and the
site tensors' invariance keeps it there — no projector, no penalty term.
[Getting started](https://ryo-wtnb11.github.io/symtenet/getting-started/) reads this
example line by line.

With the `jax` extra, the same tensors are JAX pytrees: `jit`, `grad` and `vmap` reach
through to the blocks while the symmetry structure stays static.

```python
import jax
import tenet

tenet.enable_jax()

t = out.psi[0].to_backend("jax")
g = jax.jit(jax.grad(lambda x: tenet.norm(x) ** 2))(t)
assert g.legs == t.legs
```

## What it supports

- **Symmetries.** SU(N), SU(2), U(1), Z2, fermion parity (`fZ2`) and Deligne products
  of any of them. Non-Abelian sectors are multiplets with exact Clebsch-Gordan,
  F- and R-symbols; fermionic wires carry their Koszul signs.
- **Tensors.** `SymmetricTensor` over a flat tuple of `Leg`s, one reduced block per
  allowed fusion channel. `einsum`, `tensordot`, `compose`, `transpose`, `fuse`,
  `repartition`, `trace`, and `tenet.linalg`'s `svd`, `qr`, `lq`, `eigh`, `eig`,
  `polar`, `expm`, `left_null`, plus the truncating `svd_truncated` / `eigh_truncated`.
- **Algorithms.** Finite two-site DMRG (`dmrg_`, schedules, noise, excited states) with
  MPS/MPO containers, environment caches and measurement; CTMRG on the directional
  `EnvCTM` and its C4v specialization `EnvCTMc4v`, with a differentiable fixed-bond move.
- **Backends.** NumPy, JAX and PyTorch blocks through `autoray`. `tenet.enable_jax()`
  registers `SymmetricTensor` as a JAX pytree, so `jit`, `grad` and `vmap` reach the
  blocks while the structure stays static metadata.

## Docs

- [Getting started](https://ryo-wtnb11.github.io/symtenet/getting-started/) — install and the first example.
- [User guide](https://ryo-wtnb11.github.io/symtenet/guide/tensors-legs-spaces/) — tensors, symmetries, contraction, Hamiltonians, DMRG, truncation, JAX, files.
- [Tutorials](https://ryo-wtnb11.github.io/symtenet/tutorials/dmrg/) — DMRG, fermions, SU(2), quantum chemistry, CTMRG, VMC.
- [Examples](https://ryo-wtnb11.github.io/symtenet/examples/) — runnable files with their committed output.
- [API reference](https://ryo-wtnb11.github.io/symtenet/api/tenet/) — every public name.
- [`docs/design.md`](https://ryo-wtnb11.github.io/symtenet/design/) — the categorical model underneath.
- [`REPOSITORY_RULES.md`](https://github.com/Ryo-wtnb11/symtenet/blob/main/REPOSITORY_RULES.md) — process rules for contributing.

## Citation

**If you use symtenet in research, please cite TensorKit, QSpace, and symmray,
as well as symtenet.** This requests scholarly credit for the work on which
symtenet builds; it does not add a condition to the software license.

- **TensorKit:** Lukas Devos and Jutho Haegeman, *TensorKit.jl: A Julia package
  for large-scale tensor computations, with a hint of category theory* (2025),
  [doi:10.48550/arXiv.2508.10076](https://doi.org/10.48550/arXiv.2508.10076).
  See TensorKit's [official citation metadata](https://github.com/QuantumKitHub/TensorKit.jl/blob/main/CITATION.cff)
  for its preferred citation and software record.
- **QSpace:** Andreas Weichselbaum, *QSpace — An open-source tensor library for
  Abelian and non-Abelian symmetries*, SciPost Physics Codebases **40** (2024),
  [doi:10.21468/SciPostPhysCodeb.40](https://doi.org/10.21468/SciPostPhysCodeb.40).
  When using QSpace directly, also cite the release used, following the
  [QSpace publication's citation guidance](https://scipost.org/SciPostPhysCodeb.40).
- **symmray:** Yang Gao et al., *Fermionic tensor network contraction for arbitrary
  geometries*, Physical Review Research **7**, 023193 (2025),
  [doi:10.1103/PhysRevResearch.7.023193](https://doi.org/10.1103/PhysRevResearch.7.023193),
  as requested in symmray's [official citation guidance](https://github.com/jcmgray/symmray/blob/main/docs/references.md#citing-symmray).

For symtenet itself:


```bibtex
@software{symtenet,
  author  = {Watanabe, Ryo},
  title   = {{symtenet}: a {Python} library for symmetric tensor networks
             --- a {NumPy}-style {API} on the surface, category theory under
             the hood},
  url     = {https://github.com/Ryo-wtnb11/symtenet},
  license = {Apache-2.0},
  year    = {2026}
}
```

The same metadata lives in
[`CITATION.cff`](https://github.com/Ryo-wtnb11/symtenet/blob/main/CITATION.cff) —
GitHub's "Cite this repository" button renders it as BibTeX or APA.

## License

Apache License 2.0 — see [`LICENSE`](https://github.com/Ryo-wtnb11/symtenet/blob/main/LICENSE).
