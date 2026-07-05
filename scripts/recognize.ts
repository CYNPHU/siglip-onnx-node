// CLI demo: recognize an image against the reference index.
//   npm run recognize -- path/to/crop.jpg [--rule centroid|knn|argmax] [--min-cos 0.73] [--top 5]
import { embedImage } from '../src/embed.js';
import { loadIndex, nearestTopK, recognize, MIN_COS, type Rule } from '../src/recognizer.js';

function arg(name: string, dflt: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}

async function main() {
  const img = process.argv[2];
  if (!img || img.startsWith('--')) {
    console.error('usage: npm run recognize -- <image> [--rule centroid|knn|argmax] [--min-cos 0.73] [--top 5]');
    process.exit(1);
  }
  const rule = arg('rule', 'centroid') as Rule;
  const minCos = Number(arg('min-cos', String(MIN_COS)));
  const topN = Number(arg('top', '5'));

  const r = await recognize(img, { rule, minCos });
  if (r) {
    console.log(`decision: brand=${r.brand}  sku=${r.sku}  cosine=${r.cosine.toFixed(3)}`
      + `${r.weak ? '  [WEAK — brand consensus, sku uncertain]' : ''}  (rule=${rule}, gate=${minCos})`);
  } else {
    console.log(`decision: ABSTAIN — no reference above gate ${minCos} and no brand consensus (rule=${rule})`);
  }
  loadIndex();
  console.log(`top-${topN}:`);
  for (const t of nearestTopK(await embedImage(img), topN)) {
    console.log(`  ${t.cosine.toFixed(3)}  ${t.id}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
