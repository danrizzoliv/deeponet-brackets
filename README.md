# A DeepONet Surrogate Model for Stress Fields in Parametric Bracket Families: The Cost of Geometric Generalization

Code accompanying the monograph of the same title, submitted to the MBA in
Artificial Intelligence and Big Data of the Institute of Mathematical and
Computer Sciences (ICMC), University of São Paulo, 2026.

- **Author:** Danilo Rizzo de Oliveira
- **Advisor:** Prof. Dr. João Luís Garcia Rosa
- **Co-advisor:** Prof. Dr. Alberto Costa Nogueira Junior
- **Monograph:** TODO: link to the deposited version
- **Dataset and trained model:** https://doi.org/10.5281/zenodo.22981969

The version of the code used in the monograph is tagged **`tcc-2026`**.
Later commits may extend it; to reproduce the reported results, check out
that tag.

## What this is

A DeepONet trained to predict the full von Mises stress field of a
mechanical bracket from its geometry, load and material, over a catalogue
of six structurally distinct parametric families (L, Z, U, T, O and G).
Every training case is a finite element solution produced by the pipeline
in this repository, verified for discretization against a criterion fixed
in advance.

The main results reported in the monograph:

| | relative L² field error |
|---|---|
| Final model, varying geometry (8,370 validation cases) | **0.252** |
| Same architecture, geometry held fixed | 0.145 |

- Geometric variation accounts for roughly **half** of the total error.
  With the geometry fixed the model converges; with it varying, it begins
  to overfit.
- A single model over six families is **16% more accurate** on the U family
  than a specialist trained on the U family alone.
- The peak stress is **systematically underestimated**, by 22% on average.
  The model is suitable for design **screening**, not for verification.

## Repository contents

| Script | Role |
|---|---|
| `gen_simples.py` | Parametric generation, meshing (gmsh) and finite element solution (FEniCSx) of the brackets. Writes one `.npz` per case. |
| `compactar_dataset.py` | Builds the reduced dataset used for training: drops the mesh connectivity and writes float32. |
| `dataset_simples.py` | Branch and trunk encodings, normalization statistics, and the PyTorch dataset. |
| `train_simples.py` | The DeepONet and its training loop, with checkpoint resume. |
| `treino_colab.ipynb` | Notebook that trained the final model on a Colab GPU. |
| `inspecionar_simples.py` | Per-case inspection and figures; with `--lista`, evaluates every case and prints one line per case. |
| `analisa_lista.py` | Aggregates the `--lista` output into the per-family table of the monograph. |
| `ablacao_p2.py` | The single-variable architectural ablation on 300 quadratic-element cases. |
| `bench_inferencia.py` | Inference timing against the solver. |
| `figs_fixo.py`, `figs_final.py`, `figs_par.py`, `figs_bc_malha.py`, `fig_hero.py`, `render_familias.py` | The field, boundary-condition, mesh and geometry figures of the monograph. |

## Requirements

The generator needs FEniCSx (dolfinx) and gmsh; everything else needs
PyTorch, NumPy and, for the figures, PyVista.

TODO: add the exact environment, for example with
`conda env export --no-builds > environment.yml` from the environment the
work was run in.

## Data

The dataset holds **55,803** cases, archived at the DOI above in two forms:
the complete set, and a reduced set without the mesh connectivity, about
five times smaller and sufficient for training.

| Family | L | Z | U | T | O | G | Total |
|---|---|---|---|---|---|---|---|
| Cases | 9,999 | 10,000 | 9,999 | 8,751 | 8,486 | 8,568 | **55,803** |

The families are not balanced: generation was stopped before every family
reached the same count. The monograph discusses the effect of this.

Each `sample_NNNNNN.npz` contains:

| Field | Content |
|---|---|
| `nodes`, `cells` | Mesh vertices and tetrahedron connectivity |
| `u`, `stress`, `von_mises` | Displacement, stress tensor and von Mises stress at the vertices |
| `E`, `nu` | Elastic constants |
| `loads`, `moments`, `l_ref` | Applied force and moment resultants, and the lever reference |
| `familia` | Family index: 0=L, 1=Z, 2=U, 3=T, 4=O, 5=G |
| `geo_params` | The eleven geometric parameters `[A, B, C, W, T, RF, RB, DF, DL, NF, NL]` |
| `furos_fix`, `furos_carga` | Fixation and load holes, one row per hole: `[cx, cy, cz, ax, ay, az, r, half]` |
| `load_centers`, `fixed_centers` | Hole centers |
| `n_loads`, `n_nodes`, `n_cells`, `max_vm`, `max_disp` | Counts and summary values |

## Reproducing the results

The scripts assume the project lives in `~/projetos/tcc_brackets`; several
of them set that path as `BASE`. Adjust it if yours differs.

**1. Generate the data.** One run per family, with quadratic elements:

```bash
python gen_simples.py --out-dir exp_L --fam L --seed 0 --i0 0 --n 2000 --grau 2
```

The fixed-geometry datasets were generated with `--geo-fixa <seed>`, which
freezes the part and varies only load and material. TODO: record the seed
used for each family.

Generation is deterministic in its sampling: the geometry, load and
material of each case derive from the seed and the case index. It is **not**
reproducible at the level of the mesh, because gmsh optimizes the mesh
without a fixed thread count and the fillet operation may fall back to a
smaller radius. Regenerating yields a statistically equivalent dataset,
not an identical one; exact reproduction of the figures requires the
archived files.

**2. Train.** The final model was trained on a Colab T4 with
`treino_colab.ipynb`, using:

```bash
python -u train_simples.py --dataset dataset_v4_treino --device cuda \
  --batch 32 --n-query 1024 --repeats 1 \
  --dist-furos --decoder prod --eval-every 10 --save-every 10 \
  --final-eval 400 --resume --out deeponet_6fam_55k.pt
```

It ran for **3,187 epochs**: the first 989 at the default learning rate of
10⁻³, then resumed with `--lr 3e-4`, which restarts the rate and the
scheduler. Running the notebook alone does not reproduce that restart;
it must be done as a second run with `--resume --lr 3e-4`.

**3. Evaluate by family.**

```bash
python inspecionar_simples.py --ckpt deeponet_6fam_55k.pt \
  --dataset dataset_v4_p2 --lista > lista_55k.txt
python analisa_lista.py lista_55k.txt
```

**4. Ablation, figures and timing.** `ablacao_p2.py`, the `figs_*.py`
scripts and `bench_inferencia.py`, each documented at its top.

## A note on names

The code is in English, but some names were deliberately kept in Portuguese
because they are part of the interface with the published data and model,
and changing them would break compatibility:

- **Command-line options**, such as `--caso`, `--lista`, `--salvar` and
  `--dist-furos`. They are also saved inside the checkpoint.
- **The `.npz` field names**, such as `familia`, `furos_fix` and
  `furos_carga`, which are written in the 55,803 archived files.
- **The checkpoint key** `split["treino"]`.
- **The training log lines**, such as `época` and `relL2 treino méd=`, which
  the notebook and `ablacao_p2.py` parse, and which appear in the archived
  training log.
- **The file names** of the scripts.

A short glossary: *caso* = case, *lista* = listing, *salvar* = save,
*furo* = hole, *carga* = load, *fixação* = fixation, *família* = family,
*treino* = training, *época* = epoch, *méd* = mean.

Some module docstrings describe an earlier stage of the work — for
example, three families instead of six, or a dot-product decoder — and
have not been updated. The monograph is the authoritative description of
the final configuration.

## Citation

TODO: fill in once the monograph is deposited.

```bibtex
@misc{oliveira2026deeponet,
  author = {Oliveira, Danilo Rizzo de},
  title  = {A DeepONet Surrogate Model for Stress Fields in Parametric
            Bracket Families: The Cost of Geometric Generalization},
  year   = {2026},
  note   = {MBA monograph, Instituto de Ci{\^e}ncias Matem{\'a}ticas e de
            Computa{\c{c}}{\~a}o, Universidade de S{\~a}o Paulo}
}
```

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0, as stated on the Zenodo record.
