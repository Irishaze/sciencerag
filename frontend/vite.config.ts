import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  // Served at /app (see sciencerag/app.py's StaticFiles mount, not "/") —
  // without this, the built index.html references /assets/... absolute
  // from domain root, which 404s once the app itself lives under /app/.
  base: '/app/',
  plugins: [react()],
  server: {
    // Without an explicit host, Vite's default "localhost" binding can
    // resolve to the IPv6 loopback (::1) only, depending on Node/OS — the
    // page itself still loads (browser navigation tries both address
    // families), but it leaves the dev server on a different interface
    // than the backend (uvicorn binds IPv4 127.0.0.1, see the proxy target
    // below), which shows up as an intermittent "TypeError: Failed to
    // fetch" once something in the path prefers IPv4. Pinning both to the
    // same literal IPv4 address removes the ambiguity.
    host: '127.0.0.1',
    // Dev-only: forwards /sciencerag/* to the FastAPI backend
    // (uv run uvicorn sciencerag.app:app --port 8000) so `npm run dev` can
    // call the real API without a separate CORS setup. In production this
    // app is built and served BY that same FastAPI process (see
    // sciencerag/app.py), so requests are same-origin and this proxy is
    // unused — see api.ts's BASE = "" comment.
    proxy: {
      '/sciencerag': 'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: 'dist',
  },
})
