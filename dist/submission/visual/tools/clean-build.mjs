// Remove previous production assets so hashed bundles never accumulate in dist/visual/.
// Runs before `vite build` (see app/package.json prebuild); only touches generated output.
import { readdir, rm } from 'node:fs/promises';
import { resolve, join } from 'node:path';
const visual = resolve(import.meta.dirname, '..');
const assets = join(visual, 'assets');
let removed = 0;
try {
  for (const name of await readdir(assets)) {
    if (name.endsWith('.js') || name.endsWith('.css') || name.endsWith('.map')) {
      await rm(join(assets, name));
      removed++;
    }
  }
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
}
console.log(`Cleaned ${removed} stale asset file(s).`);
