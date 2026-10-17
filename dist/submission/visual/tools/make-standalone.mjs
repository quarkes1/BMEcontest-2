// Inline the production build into one self-contained index.html.
//
// Runs after `vite build` (see app/package.json): reads the emitted index.html plus
// its referenced JS/CSS from disk, replaces the module script and stylesheet link
// with inline classic <script>/<style> blocks, verifies the IIFE bundle carries no
// module-only syntax, then removes the now-unused assets. The result opens directly
// from file:// on any machine and still serves correctly over http.
import { readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';

const visual = resolve(import.meta.dirname, '..');
const htmlPath = join(visual, 'index.html');
const assetsDir = join(visual, 'assets');

let html = await readFile(htmlPath, 'utf8');
const scriptMatch = html.match(/<script type="module"[^>]*src="\.\/assets\/([^"]+)"[^>]*><\/script>/);
// In IIFE mode the bundle injects the CSS itself, so a stylesheet link is optional.
const styleMatch = html.match(/<link rel="stylesheet"[^>]*href="\.\/assets\/([^"]+)"[^>]*>/);
if (!scriptMatch) throw new Error('make-standalone: module script tag not found in index.html');

const script = await readFile(join(assetsDir, scriptMatch[1]), 'utf8');
for (const forbidden of [/^\s*import\s/m, /^\s*export\s/m, /import\.meta/]) {
  if (forbidden.test(script)) throw new Error(`make-standalone: bundle contains module-only syntax ${forbidden}`);
}
// Function replacers are required: bundle code contains `$&`/`$'` sequences that a
// string replacement would expand as replacement patterns.
// The classic inline script must execute after parsing (unlike deferred module
// scripts), so it moves to the end of <body>; a head-placed classic script would run
// before #root exists (React error #299 / blank page).
html = html.replace(scriptMatch[0], '');
if (!html.includes('</body>')) throw new Error('make-standalone: </body> not found in index.html');
html = html.replace('</body>', () => `<script>\n${script}\n</script>\n</body>`);
if (styleMatch) {
  const css = await readFile(join(assetsDir, styleMatch[1]), 'utf8');
  html = html.replace(styleMatch[0], () => `<style>\n${css}\n</style>`);
}
if (/(src|href)="\.\/assets\//.test(html)) throw new Error('make-standalone: external asset reference left in index.html');
// Vite occasionally emits a bare CR around its injected tags; normalizing the whole
// document to LF keeps the shipped bundle byte-identical to the repo convention
// (`* text=auto eol=lf`) and to the copy the submission builder derives from it.
// CR is a line terminator in HTML/CSS/JS alike, so folding it to LF is semantics-safe.
html = html.replace(/\r\n?/g, '\n');
if (/\r/.test(html)) throw new Error('make-standalone: CR byte left in index.html');
await writeFile(htmlPath, html, 'utf8', { encoding: 'utf8' });

let removed = 0;
for (const name of await readdir(assetsDir).catch(() => [])) {
  await rm(join(assetsDir, name));
  removed++;
}
console.log(`standalone index.html written (${(html.length / 1024).toFixed(0)} kB, inlined ${removed} asset file(s))`);
