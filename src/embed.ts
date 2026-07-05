// SigLIP image embedding in pure Node: sharp preprocessing + onnxruntime-node.
// Replicates the HF SiglipImageProcessor exactly: resize 224² (bicubic, fill),
// rescale 1/255, normalize mean/std=0.5 (→ x/255*2−1), CHW, then L2-normalize
// the output so downstream similarity is a plain dot product.
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import * as fs from 'fs';
import * as path from 'path';

export const IMG = 224;
const PX = IMG * IMG;
const MODEL_PATH = process.env.SIGLIP_ONNX || path.join('models', 'siglip_vision.onnx');

let session: ort.InferenceSession | null = null;

export async function initSession(): Promise<ort.InferenceSession> {
  if (session) return session;
  if (!fs.existsSync(MODEL_PATH)) {
    throw new Error(`missing ${MODEL_PATH} — run: python python/export_siglip_onnx.py`);
  }
  session = await ort.InferenceSession.create(MODEL_PATH);
  return session;
}

/** Embed one image (path or buffer) → unit-norm Float32Array (768-d for siglip-base). */
export async function embedImage(input: string | Buffer): Promise<Float32Array> {
  const sess = await initSession();
  const { data } = await sharp(input).resize(IMG, IMG, { kernel: 'cubic', fit: 'fill' })
    .removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const f = new Float32Array(3 * PX);
  for (let i = 0; i < PX; i++) {
    for (let c = 0; c < 3; c++) f[c * PX + i] = (data[i * 3 + c] / 255) * 2 - 1;
  }
  const out = await sess.run({ pixel_values: new ort.Tensor('float32', f, [1, 3, IMG, IMG]) });
  const v = out.image_features.data as Float32Array;
  let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let d = 0; for (let i = 0; i < a.length; i++) d += a[i] * b[i]; return d;
}
