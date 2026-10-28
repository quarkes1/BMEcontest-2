// 清理上一次生成的生产资源，避免带哈希的旧文件持续累积。
// 在 vite build 前运行，仅处理构建产物。
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
