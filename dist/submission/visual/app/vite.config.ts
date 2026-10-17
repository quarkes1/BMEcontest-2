import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Production output is one self-contained index.html.
 *
 * The bundle is emitted as IIFE (classic-script safe), and
 * `tools/make-standalone.mjs` runs after the build to inline the emitted JS/CSS into
 * a single index.html with no module scripts and no external asset files — so the
 * page works when double-clicked from `file://` on any machine, and equally when
 * served over http.
 */
export default defineConfig({
  plugins: [react()],
  base: './',
  publicDir: false,
  build: {
    outDir: '..',
    emptyOutDir: false,
    rolldownOptions: {
      output: { format: 'iife', inlineDynamicImports: true },
    },
  },
  test: { environment: 'node' },
});
