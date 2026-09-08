# BUNDLE.md — the artifact that joins the two planes

*[English] · [`README.md`](README.md) · [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`EVALUATION.md`](EVALUATION.md)*

> A `RagBundle` is the answer to one question: **"which exact system produced the
> numbers I am about to believe?"** Everything in it exists because some part of
> that question was once unanswerable.

---

## 1. What a bundle is

A directory under [`bundles/`](bundles/) containing a `manifest.json`, plus a
one-line [`bundles/CURRENT`](bundles/CURRENT) pointer naming the version that is
live.

```
bundles/
  CURRENT                    ← "0.2.1"
  rag-bundle-v0.1.0/manifest.json
  rag-bundle-v0.2.0/manifest.json
  rag-bundle-v0.2.1/manifest.json
```

It is **immutable** (a change means a new version), **checksummed**, and it
records both *how the system is configured* and *what that configuration
measured*.

`CURRENT` is a single pointer read by both `serving/api/app.py` and
`pipeline/eval/smoke.py`. Before `W5-10` there were two independent answers to
"which bundle is current", which is a divergence waiting to happen — the eval
would grade one bundle while serving ran another, and both would look correct.

---

## 2. What is inside

```jsonc
{
  "bundle_version": "0.2.1",
  "created_at": "2026-09-05T04:36:27Z",
  "git_sha": "87f912b8ae",

  "components": {
    "chunking":   { "strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100,
                    "contextual": true, "chunking_fingerprint": "c7ca3e6fc4da29a5" },
    "embedding":  { "model": "BAAI/bge-m3", "dim": 1024, "normalize": true,
                    "revision": null },
    "index":      { "backend": "qdrant", "collection": "rag_bgem3_ctx",
                    "fingerprint": "ff0828fe…", "n_chunks": 15814, "n_documents": 60 },
    "retrieval":  { "mode": "hybrid", "top_k": 20, "options": { "k": 1, "candidate_k": 20 } },
    "rerank":     { "model": "BAAI/bge-reranker-v2-m3", "candidates": 50, "max_length": 512 },
    "prompt":     { "id": "chat-system", "version": 2, "hash": "ae5ea143…" },
    "generation": { "primary": "deepseek-v4-flash", "max_tokens": 1024, "temperature": 0.0 },
    "retriever_name": "reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20-w1:0.25]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n50"
  },

  "eval": {
    "golden_set": "golden_v1",
    "n_queries": 209,
    "evaluated_with_generator": "deepseek-v4-flash",
    "judge": { "model": "deepseek-v4-flash", "temperature": 0.0,
               "kappa_vs_human": 0.7368, "reasoning": false, "cache_digest": "79d7df51…" },
    "retrieval_metrics":  { "ndcg@10": 0.7079, "recall@10": 0.8022, "mrr": 0.7047, "...": "..." },
    "generation_metrics": { "faithfulness": 0.9877, "citation_coverage": 0.6186,
                            "answer_relevancy": 0.7479 }
  }
}
```

### Why each group is there

| group | the question it answers | what went wrong without it |
|---|---|---|
| `chunking` + `chunking_fingerprint` | is this the same chunking the labels were resolved against? | a chunk-size change silently re-points every positional id |
| `embedding.model` + `dim` | which vectors are in that collection? | a mismatched embedder returns plausible nonsense, not an error |
| `index.collection` + `fingerprint` | which index, and is it the one that was measured? | serving picked a collection by config; the eval measured another |
| `retrieval` + `rerank` | the exact ranking configuration | "we use a reranker" is not a configuration |
| `prompt.id/version/hash` | which prompt text produced those generation numbers? | a prompt edit changes every generation metric with nothing in the diff of the metrics |
| `generation.primary` | which model wrote the answers | see `W5-11` below |
| `eval.judge` | who graded, at what temperature, agreeing with humans how much | swapping only the judge moves a metric **7.5 points** |
| `git_sha` | which code | — |

**`retriever_name` is a single string that reconstructs the whole ranking
stack**, device and dtype included. It is what gets logged, so two runs can be
compared by eye without opening two manifests.

Caveat: **`eval.judge.kappa_vs_human` lives in the bundle, not only in a report.** A
generation metric without its judge's agreement score is a number without a unit.

---

## 3. How a bundle is minted

```bash
make index BUNDLE=bgem3                                   # build/refresh the index
make eval-retrieval BUNDLE=bgem3 MODE=hybrid RUN=cand     # retrieval metrics
make gate BUNDLE=0.2.2                                    # candidate vs champion
```

`make gate` ([`pipeline/eval/gate.py`](pipeline/eval/gate.py)) reads thresholds
from YAML, compares the candidate against the current champion, writes an HTML
report, and **exits non-zero on FAIL**. Exit codes are distinct on purpose:

| exit | meaning |
|---:|---|
| 0 | PASS — the candidate may be promoted |
| 1 | FAIL — a threshold was not met |
| 2 | **INCOMPARABLE** — the label digests differ, so no comparison is defined |

Exit 2 is not a softer failure, it is a different one. FAIL means "measured and
worse". INCOMPARABLE means "these two numbers are not on the same scale" — the
case where a comparison table looks perfectly normal and means nothing.

`make nightly` chains candidate → gate → promote, with `make nightly-dry` running
everything **except** minting the bundle and moving the pointer.

---

## 4. How a bundle is activated — three rules

Activation happens at runtime via `POST /admin/bundle/reload`; the process is not
restarted. Each rule blocks one failure mode
([`serving/core/registry.py`](serving/core/registry.py) ·
[`w4-02-bundle-reload.md`](plans/reports/tasks/w4-02-bundle-reload.md)):

### Rule 1 — build the new runtime completely, *then* swap

Loading a bundle means: read manifest → verify checksum → build retriever →
connect Qdrant → load reranker. Every step can fail. If the old bundle were
removed first, one bad reload would turn `/chat` from *"serving the previous
version"* into *"serving nothing"* — an operation meant to improve the system
becomes the one that takes it down. The reference assignment is the **last**
thing that happens; a failure anywhere before it leaves the old runtime
untouched.

### Rule 2 — a request holds a snapshot, not a reference to "current"

Reading `registry.active.retriever` twice in one request can return two different
runtimes if a reload lands in between — and then the answer cites chunks from one
index with scores from another, with nothing red anywhere. The fix is not a lock
but a **type**: `active` returns an immutable `ActiveBundle` that the caller holds
for the whole request. The swap is a single attribute assignment, atomic under the
GIL, so a snapshot already handed out never changes underneath its holder.

### Rule 3 — never close the old runtime on swap

Calling `close()` on the previous retriever to reclaim connections would tear down
exactly the requests rule 2 just protected. The old runtime lives until nothing
references it and GC collects it.

Caveat: The cost: keeping the previous version for rollback means holding **two**
runtimes, and a cross-encoder is 2.2 GB. This is usually free because consecutive
bundles typically share models and `RuntimeBuilder` shares instances by model
identity — but when two bundles use *different* models the memory genuinely
doubles, which is why the history keeps exactly **one** previous version.

---

## 5. Rollback is not "reload the old one"

```bash
curl -X POST /admin/bundle/rollback     # runtime: re-activate the previous object
make bundle-rollback BUNDLE_ROLLBACK=0.2.0   # pointer: move CURRENT on disk
```

Reloading from disk can fail — the disk changed, the network dropped, the GPU is
full. A rollback mechanism is only useful when things are already broken, so **it
must not be able to fail**. `rollback()` therefore re-activates the *same runtime
object* that was serving before, building nothing. That is the reason rule 3
exists.

Rollback deliberately does **not** re-warm the model. Re-warming makes the
emergency path slower, and the emergency path is the whole point.

---

## 6. Identity checks at load

On activation the runtime compares what the manifest declares against what
actually loaded — model name, device, dtype, `max_length` — and refuses on a
mismatch. `GET /admin/bundle` exposes the result as `runtime_drift`.

Caveat: **Known blind spot, and it bit once.** The check does not compare *library*
versions. In [`w5-01-generation-eval.md`](plans/reports/tasks/w5-01-generation-eval.md) an image had drifted to `transformers 5.16.1 /
sentence-transformers 6.0.1 / torch 2.14.0` while the lockfile pinned `5.15.0 /
5.7.0 / 2.13.0`. `runtime_drift` reported `null` throughout, and every request
that touched retrieval returned 503 because the cross-encoder loaded with mixed
dtypes. `runtime_drift: null` is therefore **not** sufficient to say "this is the
system that was measured".

The gap is closed outside the bundle, by an e2e test that asks the running
container for its library versions and compares them to `uv.lock`
(`tests/e2e/test_smoke.py`). It is the only check in the project that connects
"what was measured" to "what is running" at the library level — which is also why
it must not be allowed to fail for environmental reasons and be learned as noise.

---

## 7. Cache namespacing — the bundle's other job

The semantic cache key is a **namespace**, not a hash of the question. It carries:

```
bundle_version · prompt_version · top_k · provider:model · provider_base_url
```

**`provider:model` was missing until
[`exp-003-generator.md`](plans/reports/tasks/exp-003-generator.md)**, and its absence made a generator
ablation compare DeepSeek with itself: switching `CHAT_PROVIDER` to GLM meant the
GLM arm received DeepSeek's cached answers verbatim, while every number in the
table looked plausible.

**`provider_base_url` was added in
[`w6-01-web-ui.md`](plans/reports/tasks/w6-01-web-ui.md)**, after a server pointed at the real
DeepSeek replayed, word for word, an answer generated by the `W6-05` load-test
stub — and the `done` frame named a model that had never written it. The URL must
sit *outside* the `generator` field, because `generator` doubles as the failover
signal parsed with `split(":", 1)[-1]`, which a URL containing `:` would break.

Two additions to one key, both found by measurement rather than by review. The
general shape: **any input that changes the answer must appear in the cache key,
and the ones that get forgotten are the ones that are not part of the request.**

---

## 8. Current bundle

| | |
|---|---|
| version | **0.2.1** |
| index | `rag_bgem3_ctx` — 15,814 chunks from 60 documents |
| retrieval | BGE-M3 → hybrid RRF `k=1` → cross-encoder over 50 candidates |
| generation | `deepseek-v4-flash`, `temperature=0`, `max_tokens=1024` |
| nDCG@10 / Recall@10 / MRR | 0.7079 / 0.8022 / 0.7047 |
| faithfulness / citations / relevancy | 0.9877 / 0.6186 / 0.7479 |
| judge | `deepseek-v4-flash`, κ vs human 0.7368 ([`judge-calibration.md`](plans/reports/tasks/judge-calibration.md)) |

Read it directly: [`bundles/rag-bundle-v0.2.1/manifest.json`](bundles/rag-bundle-v0.2.1/manifest.json).
