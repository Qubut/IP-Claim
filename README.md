# ip-claim

Self-supervised **Soft Structural Vocabulary (SSV)** for patents, plus a
**prior-art collision** readout that ranks cited partners against random
controls using covering intensities from the trunk.

Given HUPD filings (and optional EPO–HUPD citation pairs), you can:

1. Train a graph-conditioned masked LM trunk that learns soft entity and
   relation codebooks over CPC and claim structure.
2. Encode a corpus and score prior-art collision with saturation covering.
3. Inspect overlays, ablate the graph prefix, and publish run artefacts to
   Hugging Face if you want.

This README is the operator guide: setup, data, train, and evaluate on your
own machines or GPU servers.

---

## Architecture (short)

| Piece | Role |
| ----- | ---- |
| **Ingestion** | HUPD JSON → typed `Patent` / claim value objects |
| **SSV trunk** | Soft banks, HGT / CompGCN encode, LoRA host MLM, constraint duals |
| **Collision head** | Covering scores, ranking metrics, encode shards, diagnose |
| **patent-ate** | Multi-word term extraction (C-value / termhood) used at ingress |
| **epo-processor** | Builds the linked EPO–HUPD citation Parquet used for collision eval |

Entry points (after install):

```bash
python -m ip_claim.ssv --help
python -m ip_claim.collision --help
# or: ip-claim-ssv / ip-claim-collision
```

Configs live under [`configs/`](configs/). Override any field on the CLI or
with your own YAML.

---

## Requirements

- Linux workstation or GPU server
- [Nix](https://nixos.org/download/) + [devenv](https://devenv.sh/) (recommended), **or** Python 3.12–3.13 + [uv](https://github.com/astral-sh/uv)
- NVIDIA GPU + drivers for real train / encode (CPU smoke is possible with tiny hosts)
- Disk for HUPD JSON (large) and optional EPO archives
- Hugging Face token when you pull gated hosts or publish runs (`HF_TOKEN`)

External tools used by this stack:

- [Qubut/patent-ate](https://github.com/Qubut/patent-ate) — termhood / occupy ingress (`patent-ate` is a declared dependency)
- [Qubut/epo-processor](https://github.com/Qubut/epo-processor) — image `ghcr.io/qubut/epo-processor` for citation-pair Parquet

---

## Setup

### 1. Clone and enter the environment

```bash
git clone https://github.com/Qubut/ip-claim.git
cd ip-claim

# Recommended: Nix + devenv (locks Python, uv, ruff, ansible, …)
devenv shell

# Inside the shell, the project is installed editable from pyproject.toml / uv.lock
```

Without devenv:

```bash
uv sync
source .venv/bin/activate   # or: uv run …
```

### 2. Secrets and local env

```bash
cp .env.example .env
# Edit .env — never commit it
```

At minimum for Hub downloads / publish:

```bash
HF_TOKEN=hf_…
# or HUGGING_FACE_HUB_TOKEN=…
```

Optional remote-GPU SSH (only if you use the ansible playbooks): put the
GPU server host, user, and jump settings in `.env`. The playbooks read
`SERVER` / `SERVER_USER` (and optional `SERVER_SSH_KEY` / `SERVER_SSH_ARGS`)
from the controller environment. Map your own names into those variables
before running ansible; do not hard-code site aliases into the repo.

### 3. Sanity check

```bash
devenv shell -- python -m ip_claim.ssv --help
devenv shell -- python -m ip_claim.collision --help
devenv shell -- pytest tests/unit/test_hf_hub_publish.py -q
```

---

## Data preparation

Collision eval and full training need two corpora. Smoke training can use
the tiny HUPD fixtures under `tests/fixtures/hupd/`.

### A. HUPD JSON (required for train / encode)

1. Obtain the [HUPD](https://huggingface.co/datasets/HUPD/hupd) dump
   (operators often use the all-years tar from the dataset card).
2. Extract so you have a tree of `*.json` patent files.
3. Point configs / CLI at that root with `--hupd-dir` or `runtime.hupd_dir`.

Example layout:

```text
/data/hupd/
  2018/
    ….json
  2019/
    ….json
```

The loader walks recursively. Document id is `application_number`.

### B. Citation pairs (required for collision eval)

Use [epo-processor](https://github.com/Qubut/epo-processor) to build a linked
EPO–HUPD Parquet (`analyze --dataset`). The smoke collision config expects a
path like:

```yaml
dataset: /data/epo-data/epo_hupd_dataset.parquet
hupd_dir: /data/hupd
```

Locally or on a server you can run the published image:

```bash
# Example — adjust mounts to your disks
podman run --rm -v /data/epo-data:/data:rw \
  ghcr.io/qubut/epo-processor:latest \
  analyze --dataset …
```

This repo also ships ansible under [`ops/ansible/`](ops/ansible/) to prepare
data on a GPU server (`playbooks/epo_data_prepare.yml` tags: `prepare`,
`start_hupd`, `start_process`, `start_analyze`, `status`). From
`ops/ansible` inside devenv:

```bash
devenv shell -- ansible-navigator run playbooks/epo_data_prepare.yml -- --tags prepare,status
```

### C. Termhood store (recommended for production train)

Production configs set `runtime.termhood_store_path` to a directory produced
by [patent-ate](https://github.com/Qubut/patent-ate) (Ray / full-HUPD extract).
Ingress occupy uses those artefacts so soft assignment is driven by extracted
terms rather than raw boilerplate.

Smoke configs omit termhood and use the tiny host
`hf-internal-testing/tiny-random-bert` so you can verify the pipe without a
full extract.

---

## Training the SSV trunk

### Smoke (local, tiny host)

Uses [`configs/ssv_train.smoke.yaml`](configs/ssv_train.smoke.yaml):

```bash
devenv shell -- python -m ip_claim.ssv train \
  --local \
  --config configs/ssv_train.smoke.yaml \
  --hupd-dir tests/fixtures/hupd \
  --checkpoint-dir artifacts/ssv-smoke
```

Drop `--local` (or pass `--ray`) to use Ray TorchTrainer with
[`configs/ssv_ray.yaml`](configs/ssv_ray.yaml).

### Production (GPU)

1. Copy or edit [`configs/ssv_train.prod.yaml`](configs/ssv_train.prod.yaml).
2. Set paths that belong to **your** disks:

   - `runtime.hupd_dir` — HUPD JSON root (or pass `--hupd-dir`)
   - `runtime.termhood_store_path` — patent-ate output (or leave unset only if you accept ingress without a store)
   - `runtime.init_weights` — optional warm-start Lightning ckpt; set to `null` for a cold start
   - `fit.checkpoint_dir` — where `ssv.ckpt` / step checkpoints land
   - `fit.num_devices`, `batch_size`, `accumulate_grad_batches` — match your GPUs / VRAM

3. Train:

```bash
devenv shell -- python -m ip_claim.ssv train \
  --config configs/ssv_train.prod.yaml \
  --hupd-dir /data/hupd \
  --checkpoint-dir /outputs/ssv
```

Useful CLI overrides: `--max-steps`, `--batch-size`, `--host`, `--d-model`,
`--publish-to-hub`, `--hub-repo-id`.

### Ablation and inspect

```bash
# Paired MLM NLL: real graph prefix vs zeroed prefix
devenv shell -- python -m ip_claim.ssv ablate-graph-prefix --help

# Soft-entity overlay HTML for a checkpoint on a HUPD peek
devenv shell -- python -m ip_claim.ssv inspect-graph --help

# Join extract parquet to committed termhood
devenv shell -- python -m ip_claim.ssv probe-ingress --help
```

### Remote train via ansible (optional)

With `SERVER` / `SERVER_USER` set in the environment and the repo mirrored on
the GPU host:

```bash
cd ops/ansible
devenv shell -- ansible-navigator run playbooks/ssv_train.yml -- --list-tags
devenv shell -- ansible-navigator run playbooks/ssv_train.yml -- --tags provision,image,train
devenv shell -- ansible-navigator run playbooks/ssv_train.yml -- --tags status
```

Tag names follow the playbook; use `--list-tags` on your checkout rather than
memorizing a stale list. Follow until GPU util and ledger mtimes prove the
job is live — a green PLAY RECAP alone is not acceptance.

---

## Evaluating prior-art collision

Collision needs:

- a trained trunk checkpoint (`checkpoint` in YAML)
- HUPD JSON (`hupd_dir`)
- citation Parquet from epo-processor (`dataset`)
- an SSV train YAML that matches the checkpoint host / arch (`ssv_config`)

### Smoke eval

Edit paths in [`configs/collision_eval.smoke.yaml`](configs/collision_eval.smoke.yaml), then:

```bash
devenv shell -- python -m ip_claim.collision eval \
  --config configs/collision_eval.smoke.yaml \
  --output-dir /outputs/collision-smoke
```

This encodes patents (Lightning predict → per-rank shards), ranks with
saturation covering, and writes `covering.json` (plus optional explain
artefacts).

### Full eval

Same entrypoint with [`configs/collision_eval.yaml`](configs/collision_eval.yaml).
Typical knobs:

| Field | Meaning |
| ----- | ------- |
| `prefix_mode` | `trunk` (graph-conditioned) vs host-only ablations |
| `query_limit` | Cap queries (smoke / debug) |
| `recall_k` | Recall@K cutoffs |
| `covering.*` | Saturation σ and optional keep sparsifiers |
| `num_devices` / `encode_batch_size` | Encode throughput |

### Offline tools (no re-encode)

```bash
# Score existing shards for letter / X–A diagnostics
devenv shell -- python -m ip_claim.collision diagnose --help

# Sweep keep_grid on existing shards
devenv shell -- python -m ip_claim.collision rank --help

# Disclosure length histogram
devenv shell -- python -m ip_claim.collision measure-disclosure --help
```

### Publish a run

```bash
devenv shell -- python -m ip_claim.ssv publish --help
devenv shell -- python -m ip_claim.collision publish --help
```

Requires `HF_TOKEN` and a dataset repo id.

---

## Configuration map

| File | Purpose |
| ---- | ------- |
| `configs/ssv_train.smoke.yaml` | Tiny host, few steps, fixture-friendly |
| `configs/ssv_train.prod.yaml` | ModernBERT-large trunk defaults |
| `configs/ssv_train.prod.*.yaml` | Ablation / mask-ratio / dest-compare variants |
| `configs/ssv_ray.yaml` | Ray scaling |
| `configs/collision_eval.smoke.yaml` | Small collision smoke |
| `configs/collision_eval.yaml` | Full collision eval |
| `configs/collision_eval.disclosure*.yaml` | Disclosure probes |

All configs are OmegaConf YAML validated into Pydantic models
(`SsvTrainConfig`, `CollisionEvalConfig`). Secrets use
`${oc.env:HF_TOKEN,…}` at load time.

---

## Project layout

```text
src/ip_claim/
  shared/       # errors, Hub publish, local Ray session
  ingestion/    # Patent models + HUPD JSON adapters
  ssv/          # trunk library, Lightning module, train CLI
  collision/    # covering, encode job, eval CLI
  app/          # dependency-injector composition roots
configs/        # job YAML
containers/     # GPU runtime image (Nix / Containerfile)
ops/ansible/    # data prep + SSV remote jobs
tests/          # unit + integration
experiments/    # covering-objective falsification campaign (optional)
```

---

## Tests

```bash
devenv shell -- ruff check src
devenv shell -- ruff format --check src
devenv shell -- pytest
```

Targeted examples:

```bash
devenv shell -- pytest tests/unit/ssv -q
devenv shell -- pytest tests/unit/test_covering.py tests/unit/test_collision_ranking.py -q
```

---

## License

MIT (see package metadata in `pyproject.toml`). Respect HUPD and host-model
licenses for any redistributed weights or derived corpora.
