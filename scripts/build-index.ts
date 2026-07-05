// Embed every reference image under reference/<brand>/<sku>/*.jpg through the
// SAME embedding path queries use at inference (one embedding space, by
// construction), and write models/reference_index.json.
//   npm run build-index
import * as fs from 'fs';
import * as path from 'path';
import { embedImage } from '../src/embed.js';

const REF_DIR = process.env.SIGLIP_REF_DIR || 'reference';
const OUT = process.env.SIGLIP_INDEX || path.join('models', 'reference_index.json');
const EXT = new Set(['.jpg', '.jpeg', '.png']);

async function main() {
  const items: Array<{ id: string; brand: string; sku: string }> = [];
  const embs: number[][] = [];
  const brands = fs.existsSync(REF_DIR)
    ? fs.readdirSync(REF_DIR, { withFileTypes: true }).filter(d => d.isDirectory())
    : [];
  if (!brands.length) {
    console.error(`no reference images — expected reference/<brand>/<sku>/*.jpg`);
    process.exit(1);
  }
  for (const b of brands) {
    const bDir = path.join(REF_DIR, b.name);
    for (const s of fs.readdirSync(bDir, { withFileTypes: true }).filter(d => d.isDirectory())) {
      const sDir = path.join(bDir, s.name);
      for (const f of fs.readdirSync(sDir).filter(f => EXT.has(path.extname(f).toLowerCase()))) {
        const fp = path.join(sDir, f);
        const v = await embedImage(fp);
        items.push({ id: `${b.name}/${s.name}/${f}`, brand: b.name, sku: s.name });
        embs.push(Array.from(v));
        process.stdout.write(`\r  embedded ${items.length} images...`);
      }
    }
  }
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify({ items, emb: embs }));
  const skus = new Set(items.map(it => `${it.brand}|${it.sku}`));
  console.log(`\n== index built: ${items.length} references · ${skus.size} SKUs · ${new Set(items.map(i => i.brand)).size} brands → ${OUT}`);
}

main().catch(e => { console.error(e); process.exit(1); });
