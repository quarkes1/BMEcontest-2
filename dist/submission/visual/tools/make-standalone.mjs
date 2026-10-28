// 将生产构建内嵌为单个自包含 index.html。
//
// 在 vite build 后读取输出页面及其引用的脚本和样式，
// 将模块脚本与样式链接替换为内嵌内容。
// 检查立即调用函数包中不含模块专属语法，
// 随后移除已不再使用的外部资源。生成页面可直接
// 通过本地文件方式打开，也可由 HTTP 服务提供。
import { readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';

const visual = resolve(import.meta.dirname, '..');
const htmlPath = join(visual, 'index.html');
const assetsDir = join(visual, 'assets');

let html = await readFile(htmlPath, 'utf8');
const scriptMatch = html.match(/<script type="module"[^>]*src="\.\/assets\/([^"]+)"[^>]*><\/script>/);
// 立即调用函数包自行注入样式，因此独立样式链接是可选项。
const styleMatch = html.match(/<link rel="stylesheet"[^>]*href="\.\/assets\/([^"]+)"[^>]*>/);
if (!scriptMatch) throw new Error('make-standalone: module script tag not found in index.html');

const script = await readFile(join(assetsDir, scriptMatch[1]), 'utf8');
for (const forbidden of [/^\s*import\s/m, /^\s*export\s/m, /import\.meta/]) {
  if (forbidden.test(script)) throw new Error(`make-standalone: bundle contains module-only syntax ${forbidden}`);
}
// 必须使用函数作为替换值，避免包内的美元符号序列
// 被字符串替换机制误解释为替换模式。
// 普通内嵌脚本不同于延迟执行的模块脚本，必须在页面解析后运行。
// 因此将脚本放在 body 末尾，避免其在界面根节点创建之前执行，
// 否则会触发 React 299 错误或空白页。
html = html.replace(scriptMatch[0], '');
if (!html.includes('</body>')) throw new Error('make-standalone: </body> not found in index.html');
html = html.replace('</body>', () => `<script>\n${script}\n</script>\n</body>`);
if (styleMatch) {
  const css = await readFile(join(assetsDir, styleMatch[1]), 'utf8');
  html = html.replace(styleMatch[0], () => `<style>\n${css}\n</style>`);
}
if (/(src|href)="\.\/assets\//.test(html)) throw new Error('make-standalone: external asset reference left in index.html');
// Vite 注入标签时偶尔产生单独的回车字符。
// 将完整文档统一为 LF，可保持交付文件与仓库约定
// 以及提交包构建器复制的文件逐字节一致。
// HTML、CSS 和 JavaScript 都将回车视为换行，统一为 LF 不改变语义。
html = html.replace(/\r\n?/g, '\n');
if (/\r/.test(html)) throw new Error('make-standalone: CR byte left in index.html');
await writeFile(htmlPath, html, 'utf8', { encoding: 'utf8' });

let removed = 0;
for (const name of await readdir(assetsDir).catch(() => [])) {
  await rm(join(assetsDir, name));
  removed++;
}
console.log(`standalone index.html written (${(html.length / 1024).toFixed(0)} kB, inlined ${removed} asset file(s))`);
