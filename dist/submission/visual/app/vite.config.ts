import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/** 生产构建输出单个自包含 index.html。采用普通脚本可执行的立即调用函数格式，构建后由 tools/make-standalone.mjs 内嵌脚本和样式，不依赖外部资源；本地双击和 HTTP 服务均可使用。 */
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
