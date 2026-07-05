# siglip-onnx-node

Run **SigLIP image embeddings in pure Node.js** — no Python service at inference time — and build a small **product recognizer** on top: k-NN / centroid classification with **abstention gating**.

Extracted from a production retail-execution SaaS, where this exact pipeline recognises FMCG products (brand + variant) from shelf-photo crops inside a Next.js server on Azure App Service. Client catalog and images are not included — point it at your own reference photos.

```
one-time (Python)                     runtime (Node.js only)
─────────────────                     ─────────────────────────────────────────────
HF siglip-base ──export──> ONNX  ──>  sharp preprocessing ──> onnxruntime-node
      │                                        │
      └─ parity check torch↔onnx               └──> 768-d embedding ──> nearest
         (cosine ~1.0)                               reference (centroid rule)
                                                     ──> {brand, sku, cosine}
                                                         or ABSTAIN below gate
```

## Why ONNX-in-Node

The production app is a Next.js monolith. A Python inference microservice would mean a second runtime to deploy, monitor and pay for — for a model that runs ~85 ms/crop on CPU. Exporting `google/siglip-base-patch16-224`'s vision tower to ONNX and running it with `onnxruntime-node` keeps everything in one process. The subtle part is **replicating the Hugging Face processor exactly** in JS:

- resize to 224×224, **bicubic**, `fit: 'fill'` (square, no letterbox — matches `SiglipImageProcessor`)
- rescale `1/255`, then normalize with mean = std = 0.5 → `x/255 * 2 − 1`
- CHW layout, L2-normalize the output embedding

Get any of those wrong and embeddings silently land in a slightly different space — cosines still *look* plausible, accuracy quietly degrades. That's why the repo verifies parity twice: the export script checks **torch ↔ ONNX** (cosine ~1.0 on random inputs), and you can spot-check **Node ↔ Python** by embedding the same image in both.

## Quick start

```bash
# 0. install
npm install
pip install torch transformers onnx onnxruntime   # one-time, export only

# 1. export the vision tower → models/siglip_vision.onnx (prints parity check)
python python/export_siglip_onnx.py

# 2. put reference images in reference/<brand>/<sku>/*.jpg  (3+ images per SKU helps)
# 3. build the reference index
npm run build-index

# 4. recognize any image
npm run recognize -- path/to/crop.jpg
npm run recognize -- path/to/crop.jpg --rule knn --min-cos 0.7
```

Example output:

```
decision: brand=acme  sku=shampoo-blue  cosine=0.842  (rule=centroid, gate=0.73)
top-5:
  0.842  acme/shampoo-blue/3.jpg
  0.831  acme/shampoo-blue/1.jpg
  0.789  acme/shampoo-green/2.jpg
  ...
```

## The decision rules (and why there are three)

`src/recognizer.ts` implements three interchangeable rules — each exists because the previous one failed in production:

1. **argmax** — nearest single reference. Fragile: one outlier reference of the *wrong* SKU can edge out eight correct references by 0.005 cosine.
2. **k-NN, margin-gated** — only intervenes on a near-tie. If top-1 leads the nearest *different* SKU by ≥ margin, argmax is trusted as-is; otherwise the top-K vote (cosine-weighted). Fixes the outlier case without ever overriding a confident argmax. Remaining flaw: **count bias** — a SKU with many references can "pull in" a lookalike by sheer numbers.
3. **centroid** (default) — one prototype per SKU (mean of its references, renormalized); each SKU votes exactly once regardless of reference count. More references only sharpen the prototype. The reported match is the chosen SKU's best *real* reference, not the synthetic mean.

Two safety layers sit on top:

- **Abstention gate** — below a cosine threshold the recognizer returns `null` rather than a guess. In our production calibration, a 0.73 gate corresponded to ≈95% precision; below ~0.6 was mostly partial/edge-crop noise. Calibrate on your own data — an abstaining model composes cleanly with a fallback (in production, a VLM pass), a guessing one poisons everything downstream.
- **Brand-consensus rescue** — when top-1 falls below the gate but ≥5 of the top-6 neighbours agree on one *brand* (with a floor on top-1 and a margin over the runner-up brand), emit a brand-confident / variant-weak result instead of nothing. Guards matter: the floor stops off-catalog junk from becoming a brand; the margin stops a brand with many references winning by base rate.

## Repo layout

```
python/export_siglip_onnx.py   one-time export + torch↔onnx parity check
src/embed.ts                   ONNX session + sharp preprocessing → unit embedding
src/recognizer.ts              argmax / margin-gated kNN / centroid + consensus + gate
scripts/build-index.ts         embed reference/<brand>/<sku>/*.jpg → models/reference_index.json
scripts/recognize.ts           CLI demo
```

## License

MIT
