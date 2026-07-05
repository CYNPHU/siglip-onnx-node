// Visual SKU recognizer over a precomputed reference index:
//   crop → SigLIP embedding → nearest reference → {brand, sku, cosine} — or ABSTAIN.
//
// Three interchangeable decision rules (each exists because the previous one
// failed on real shelf photos):
//   argmax   — nearest single reference. Fragile: one outlier reference of the
//              WRONG SKU can edge out many correct references by a hair.
//   knn      — margin-gated soft vote. If top-1 leads the nearest reference of a
//              DIFFERENT SKU by ≥ margin, argmax is trusted as-is; only on a
//              near-tie do the top-K vote (cosine-weighted). Fixes the outlier
//              case without ever overriding a confident argmax. Remaining flaw:
//              count bias — a SKU with many references can win by sheer numbers.
//   centroid — one prototype per SKU (mean of its references, renormalized);
//              every SKU votes exactly ONCE regardless of reference count, so
//              more references only sharpen the prototype (default rule).
//
// Safety layers:
//   abstention gate — below MIN_COS return null instead of a guess. An
//              abstaining recognizer composes cleanly with a fallback; a
//              guessing one poisons everything downstream.
//   brand consensus — when top-1 is below the gate but the top-K neighbours
//              AGREE on one brand, emit a brand-confident/variant-weak result.
//              Guards: floor on top-1 (off-catalog junk must not become a
//              brand), ≥MIN of K share the brand, and that brand leads the
//              runner-up brand by MARGIN (a brand with many references must
//              not win by base rate alone).
import * as fs from 'fs';
import * as path from 'path';
import { cosine, embedImage } from './embed.js';

const INDEX_PATH = process.env.SIGLIP_INDEX || path.join('models', 'reference_index.json');

export const MIN_COS = Number(process.env.SIGLIP_MIN_COS ?? 0.73);
const KNN_K = Number(process.env.SIGLIP_KNN_K ?? 10);
const KNN_MARGIN = Number(process.env.SIGLIP_KNN_MARGIN ?? 0.03);
const CONSENSUS_K = Number(process.env.SIGLIP_CONSENSUS_K ?? 6);
const CONSENSUS_MIN = Number(process.env.SIGLIP_CONSENSUS_MIN ?? 5);
const CONSENSUS_FLOOR = Number(process.env.SIGLIP_CONSENSUS_FLOOR ?? 0.70);
const CONSENSUS_MARGIN = Number(process.env.SIGLIP_CONSENSUS_MARGIN ?? 0.05);

export type Rule = 'argmax' | 'knn' | 'centroid';
export type Recognition = { id: string; brand: string; sku: string; cosine: number; weak?: boolean };
type IndexFile = { items: Array<{ id: string; brand: string; sku: string }>; emb: number[][] };

let ids: string[] = [];
let metaById = new Map<string, { brand: string; sku: string }>();
let emb: Float32Array[] = [];
let cents: { key: string; vec: Float32Array; members: number[] }[] = [];

export function loadIndex(): void {
  if (emb.length) return;
  if (!fs.existsSync(INDEX_PATH)) {
    throw new Error(`missing ${INDEX_PATH} — run: npm run build-index`);
  }
  const idx = JSON.parse(fs.readFileSync(INDEX_PATH, 'utf8')) as IndexFile;
  ids = idx.items.map(it => it.id);
  metaById = new Map(idx.items.map(it => [it.id, { brand: it.brand, sku: it.sku }]));
  emb = idx.emb.map(r => Float32Array.from(r)); // pre-normalized at build time
  cents = buildCentroids();
}

const asRec = (i: number, cos: number): Recognition => {
  const m = metaById.get(ids[i])!;
  return { id: ids[i], brand: m.brand, sku: m.sku, cosine: cos };
};
const skuKey = (i: number) => { const m = metaById.get(ids[i])!; return `${m.brand}|${m.sku}`; };

// One prototype per (brand|sku): mean of its reference embeddings, renormalized.
// Member indices are kept so the decision can report a REAL reference (id + its
// actual cosine) from the chosen SKU rather than the synthetic mean.
function buildCentroids(): typeof cents {
  const dim = emb[0]?.length ?? 0;
  const groups = new Map<string, { sum: Float32Array; members: number[] }>();
  for (let i = 0; i < emb.length; i++) {
    const key = skuKey(i);
    let g = groups.get(key);
    if (!g) { g = { sum: new Float32Array(dim), members: [] }; groups.set(key, g); }
    const e = emb[i]; for (let k = 0; k < dim; k++) g.sum[k] += e[k];
    g.members.push(i);
  }
  const out: typeof cents = [];
  for (const [key, g] of groups) {
    let n = 0; for (const x of g.sum) n += x * x; n = Math.sqrt(n) || 1;
    const vec = new Float32Array(dim);
    for (let k = 0; k < dim; k++) vec[k] = g.sum[k] / n;
    out.push({ key, vec, members: g.members });
  }
  return out;
}

function scoredDesc(v: Float32Array): { i: number; d: number }[] {
  const s = emb.map((e, i) => ({ i, d: cosine(v, e) }));
  s.sort((a, b) => b.d - a.d);
  return s;
}

function nearestArgmax(v: Float32Array): Recognition | null {
  const s = scoredDesc(v);
  return s.length ? asRec(s[0].i, s[0].d) : null;
}

function nearestKNN(v: Float32Array): Recognition | null {
  const s = scoredDesc(v);
  if (!s.length) return null;
  const top1 = s[0];
  // runner-up of a DIFFERENT SKU; if argmax leads it clearly, trust argmax
  const runner = s.find(x => skuKey(x.i) !== skuKey(top1.i));
  if (!runner || top1.d - runner.d >= KNN_MARGIN) return asRec(top1.i, top1.d);
  // near-tie: cosine-weighted vote among top-K
  const groups = new Map<string, { sum: number; bestI: number; bestD: number }>();
  for (const x of s.slice(0, KNN_K)) {
    const key = skuKey(x.i);
    const g = groups.get(key);
    if (g) { g.sum += x.d; if (x.d > g.bestD) { g.bestD = x.d; g.bestI = x.i; } }
    else groups.set(key, { sum: x.d, bestI: x.i, bestD: x.d });
  }
  let win: { sum: number; bestI: number; bestD: number } | null = null;
  for (const g of groups.values()) if (!win || g.sum > win.sum) win = g;
  return win ? asRec(win.bestI, win.bestD) : asRec(top1.i, top1.d);
}

function nearestCentroid(v: Float32Array): Recognition | null {
  if (!cents.length) return nearestArgmax(v);
  let win = -1, wd = -Infinity;
  for (let i = 0; i < cents.length; i++) {
    const d = cosine(v, cents[i].vec);
    if (d > wd) { wd = d; win = i; }
  }
  if (win < 0) return null;
  let mb = -Infinity, mi = -1;
  for (const idx of cents[win].members) {
    const d = cosine(v, emb[idx]);
    if (d > mb) { mb = d; mi = idx; }
  }
  return asRec(mi, mb);
}

/** Top-K nearest references, sorted desc by cosine, NO gate. */
export function nearestTopK(v: Float32Array, k: number): Recognition[] {
  return scoredDesc(v).slice(0, k).map(x => asRec(x.i, x.d));
}

/** Brand-consensus over an ungated topK: WEAK recognition when the neighbours
 *  agree on a brand, else null. See guards in the header comment. */
export function brandConsensus(topK: Recognition[]): Recognition | null {
  const top = topK[0];
  if (!top || top.cosine < CONSENSUS_FLOOR) return null;               // (1) floor on top-1
  const bn = (b: string) => (b ?? '').toLowerCase().trim();
  const counts = new Map<string, number>();
  for (const r of topK) counts.set(bn(r.brand), (counts.get(bn(r.brand)) ?? 0) + 1);
  const [domBrand, domCount] = [...counts].sort((a, b) => b[1] - a[1])[0];
  if (!domBrand || domCount < CONSENSUS_MIN) return null;              // (2) consensus strength
  const domMatches = topK.filter(r => bn(r.brand) === domBrand);       // sorted desc cos
  const otherBest = topK.find(r => bn(r.brand) !== domBrand);
  if (otherBest && domMatches[0].cosine - otherBest.cosine < CONSENSUS_MARGIN) return null; // (3) margin
  // voted sku among the dominant brand's matches (ties → highest cosine)
  const vc = new Map<string, number>();
  for (const r of domMatches) vc.set(r.sku, (vc.get(r.sku) ?? 0) + 1);
  const topSku = [...vc].sort((a, b) => b[1] - a[1])[0][0];
  const pick = domMatches.find(r => r.sku === topSku) ?? domMatches[0];
  return { ...pick, weak: true };                                      // brand-confident, sku-weak
}

export type RecognizeOpts = { rule?: Rule; minCos?: number; consensus?: boolean };

/** Recognize one image. Returns null (ABSTAIN) when below the gate and no consensus. */
export async function recognize(input: string | Buffer, opts: RecognizeOpts = {}): Promise<Recognition | null> {
  loadIndex();
  const { rule = 'centroid', minCos = MIN_COS, consensus = true } = opts;
  const v = await embedImage(input);
  const r = rule === 'centroid' ? nearestCentroid(v) : rule === 'knn' ? nearestKNN(v) : nearestArgmax(v);
  if (r && r.cosine >= minCos) return r;
  if (consensus) {
    const rescue = brandConsensus(nearestTopK(v, CONSENSUS_K));
    if (rescue) return rescue;
  }
  return null;
}
