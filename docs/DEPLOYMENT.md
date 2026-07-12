# Deployment

The application is two halves with very different hosting needs. Deploy them
separately.

| Half | What it is | Right host |
|---|---|---|
| Frontend | One static HTML file (`app/webapp/index.html`, Three.js from a CDN) | Vercel, Netlify, GitHub Pages — any static host |
| Backend | Python compute server (JAX shell FEA, gmsh/OpenCASCADE meshing) | A container / VM host: Render, Railway, Fly.io, Hugging Face Spaces, or your own machine |

## Why the backend cannot be a Vercel serverless function

If you point Vercel at the whole repo you get:

```
Error: Total bundle size (1421.91 MB) exceeds the maximum function size (500 MB).
```

That is not a packaging problem to shrink away — it is a category mismatch:

1. **Size.** JAX + jaxlib + gmsh (bundles OpenCASCADE) + SciPy + PyVista ≈ 1.4 GB.
   Those dependencies *are* the solver; there is nothing meaningful to trim.
2. **Time.** A winding optimization runs 90–120 s; even the first fast screen pays
   20–30 s of meshing + JAX JIT compilation. Serverless timeouts are far shorter on
   most plans, and every cold start would pay the JIT cost again.
3. **State.** The engine keeps a mesh + compiled-solver cache in memory between
   requests. Serverless functions are stateless.

## Deploy the frontend to Vercel

The repo ships `vercel.json` (static deployment of `app/webapp/`, install/build
disabled) and `.vercelignore` (uploads only the frontend). So:

```bash
npm i -g vercel        # if needed
vercel --prod          # from the repo root
```

or import the GitHub repo in the Vercel dashboard — the committed config keeps it
static, so the 500 MB function error cannot recur.

## Deploy the backend

### Option A — your own machine (simplest, free)

```bash
pip install -e .
python -m app.server           # serves on http://localhost:8081
```

Then open the Vercel URL with `?api=http://localhost:8081` once (see below).
Browsers treat `localhost` as a secure context, so an HTTPS page may call it.

### Option B — a container host (public backend)

```bash
docker build -t copv-studio .
docker run -p 8081:8081 copv-studio
```

Push the image to Render / Railway / Fly.io / Hugging Face Spaces (all accept
multi-GB images and long-running processes; HF Spaces has a workable free tier).
Set nothing else — the image already binds `0.0.0.0:8081`, and the API sends
`Access-Control-Allow-Origin: *` so the Vercel-hosted frontend can call it
cross-origin. Give the service ≥ 2 GB RAM; first solve JIT-compiles (~30 s), after
which the in-process cache makes repeat solves fast. If the platform demands a
specific port, set `COPV_PORT`.

Note: the image is defined here but sizes ~2 GB and takes several minutes to build —
build it on the host or in CI, and treat the first deploy as a test.

## Connect the two

Open the deployed frontend once with the backend address in the URL:

```
https://<your-project>.vercel.app/?api=https://<your-backend-host>
```

The frontend stores the address (localStorage) and uses it from then on;
`?api=reset` forgets it. Served straight from the backend (`http://localhost:8081`)
no parameter is needed — same-origin requests are used automatically.

## Security note

The backend has no authentication — anyone with the URL can submit solve jobs
(each optimize run occupies the solver for ~2 minutes). Fine for a demo; for
anything more, put it behind an authenticating proxy or keep it private and run
the frontend against a local backend.
