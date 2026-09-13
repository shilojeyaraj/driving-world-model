# Deploying this dashboard to Vercel

This folder is a fully self-contained static site — one `index.html` plus a
`dashboard_data/` folder of small MP4 clips and JSON, no build step, no server,
no API keys. Any of the three options below works.

## Option A — drag and drop (fastest, no installs)
1. Go to https://vercel.com/new
2. Drag this `dashboard/` folder onto the page (or click to browse and select it).
3. Framework preset: "Other" (it should auto-detect as static).
4. Deploy. You'll get a URL like `driving-console.vercel.app` in ~30 seconds.

## Option B — connect the GitHub repo (best if you'll keep updating it)
1. Push this repo to GitHub (if it isn't already).
2. In Vercel: New Project → Import your repo.
3. Set **Root Directory** to `dashboard`.
4. Framework preset: "Other" / leave build command empty, output directory `.`.
5. Deploy. Every future push to `main` auto-redeploys.

## Option C — Vercel CLI
Requires Node.js (not currently installed on this machine — install via
https://nodejs.org or `brew install node` first).
```bash
cd dashboard
npx vercel --prod
```
Follow the prompts (log in, confirm the project name/scope). It deploys this
exact folder as-is.

## Custom domain
Once deployed, add a custom domain (e.g. `driving.shilojeyaraj.com`) from the
Vercel project's Settings → Domains tab — free on Vercel's hobby tier, just
needs a DNS record at your domain registrar.

## Regenerating the data
Everything in `dashboard_data/` is produced by `../scripts/export_dashboard_data.py`
(run from the repo root, in the project's `.venv`). Re-run it, then re-copy the
folder here (`cp -r ../dashboard_data .`) and redeploy to refresh the numbers.
