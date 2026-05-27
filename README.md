# The Interactive Handbook on Machine Learning

A web-based interactive book teaching the foundations of classical machine
learning. Inspired by [cartesian.app](https://cartesian.app) — the same
format applied to ML.

## Quick start

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. The landing page lists all 22 chapters.
Chapter 18 (k-means) is the only one with content so far — it doubles as
the format prototype and the proof that all the pieces compose.

## Project structure

```
ml-book/
├── app/
│   ├── layout.tsx              Root layout. Fonts and metadata.
│   ├── page.tsx                Landing page / table of contents.
│   ├── globals.css             Design tokens and typography.
│   ├── icon.svg                Favicon (the three-cluster mark).
│   ├── robots.ts               robots.txt route handler.
│   ├── sitemap.ts              sitemap.xml route handler.
│   └── chapters/
│       ├── layout.tsx          Chapter shell with back-link.
│       └── 18-k-means/
│           └── page.mdx        Chapter content. Prose + math + widgets.
├── components/
│   ├── KMeansPlayback.tsx      §2 — main playback widget.
│   ├── UpdateStepDemo.tsx      §4 — draggable centroid + WCSS surface.
│   ├── ConvergencePlot.tsx     §5 — clustering + WCSS-over-iteration plot.
│   ├── KSweepWidget.tsx        §6 — elbow / silhouette across K.
│   ├── InitComparisonDemo.tsx  §7 — random vs k-means++ side by side.
│   └── PyodideEditor.tsx       §10 + §11 — Python editor in the browser.
├── lib/
│   ├── rng.ts                  Seeded RNG + Gaussian sampling.
│   ├── datasets.ts             Synthetic 2D datasets (blobs, aniso, moons).
│   ├── kmeans.ts               k-means algorithm with frame history.
│   ├── silhouette.ts           Silhouette coefficient.
│   └── pyodide.ts              Singleton Pyodide loader, stdout capture.
├── mdx-components.tsx          MDX customisation hook.
├── next.config.mjs             MDX, remark-math, rehype-katex wiring.
├── tailwind.config.ts          Fonts and colour tokens.
└── postcss.config.mjs
```

## Design system

The visual identity lives in `app/globals.css` as CSS custom properties:

- `--paper` warm off-white background, `--ink` near-black text
- `--accent` a single restrained teal, used sparingly
- `--rule` subtle separator colour
- Fonts: Source Serif 4 (body), Geist (UI chrome), JetBrains Mono (code)

Dark mode is automatic via `prefers-color-scheme`. Every chapter component
uses these tokens, so changing them in one place re-skins the whole book.

## Writing a new chapter

1. Create `app/chapters/<num>-<slug>/page.mdx`.
2. Import any widgets you need from `@/components/...`.
3. Export `metadata` from the top of the file (title, description, OG).
4. Write prose. Use `$...$` for inline math and `$$...$$` for display math.
5. Add the chapter to the `parts` array in `app/page.tsx` and set its `href`.
6. Add an entry in `app/sitemap.ts`.

## The playback pattern

`KMeansPlayback`, `ConvergencePlot`, and `InitComparisonDemo` all implement
the same pattern: run an algorithm once and capture a frame per step in a
history array; render the frame at the current index; expose play, scrub,
step, and reset controls. Every future iterative-algorithm chapter
(gradient descent, decision tree growing, MLP training, EM, …) will reuse
this shape. A planned refactor will extract a `usePlayback(frames)` hook
so future widgets only have to specify how to render one frame.

---

# Deploying to Vercel

The fastest path from this repo to a live URL. Estimated time: about ten
minutes.

## 1. Push the project to GitHub

If the project is not already a git repository:

```bash
cd ml-book
git init
git add -A
git commit -m "Initial commit: chapter 18 reference build"
```

Create a new repository on GitHub (private or public, your call), then:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/ml-book.git
git push -u origin main
```

## 2. Sign up for Vercel and connect GitHub

If you do not already have an account, sign up at
[vercel.com/signup](https://vercel.com/signup) — the *Hobby* plan is free
and sufficient for this project. During signup, authorise the Vercel GitHub
app on your GitHub account; this lets Vercel see your repositories.

## 3. Import the repository

From your Vercel dashboard, click *Add New* → *Project*, then select the
`ml-book` repository. Vercel will auto-detect that it is a Next.js project
and pre-fill all build settings:

- Framework Preset: Next.js
- Build Command: `next build`
- Output Directory: `.next`
- Install Command: `npm install`

You should not need to change any of them. Leave environment variables
empty — the project boots correctly with defaults.

Click *Deploy*. The first build takes 2–4 minutes.

## 4. Smoke test the deployment

When the build completes, Vercel gives you a URL like
`https://ml-book-<hash>.vercel.app`. Open it and run through this checklist:

- [ ] Landing page loads, table of contents renders.
- [ ] Click *18 — k-means clustering*. Chapter loads, prose renders.
- [ ] Math typesets cleanly (no raw `$...$` visible anywhere).
- [ ] §2 widget: press *Play*. Centroids glide, points get coloured.
- [ ] §4 widget: drag the centroid. WCSS values update live, level rings
      visible.
- [ ] §5 widget: press *Play*. Scatter and WCSS chart move in lockstep.
- [ ] §6 widget: slide K. Clustering updates, peak markers move.
- [ ] §7 widget: press *Play both*. Two panels animate together.
- [ ] §10 editor: press *Run*. First time takes 5–10 seconds (Pyodide
      download), then prints final centroids.
- [ ] §11 Problem 1: click *Show solution*. Solution expands.
- [ ] §11 Problem 2: press *Run* before changing anything. Should error
      cleanly (the `assignments = ...` placeholder is not valid). Then
      paste the solution and re-run.
- [ ] Toggle system dark mode in your OS. Site re-skins live.
- [ ] Resize window narrow (mobile width). Layout stays usable.

If anything fails, the *Logs* tab on the Vercel deployment page shows the
build and runtime output.

## 5. (Optional) Set a custom domain

In the Vercel dashboard, open the project → *Settings* → *Domains*. Add
your domain and follow the DNS instructions Vercel provides. Once the
domain is live, set the `NEXT_PUBLIC_SITE_URL` environment variable to
`https://yourdomain.com` so OG tags and the sitemap point at the canonical
URL. Without this variable the site falls back to Vercel's preview
hostname, which still works but is less polished for social sharing.

To set the env var: project → *Settings* → *Environment Variables* → add
`NEXT_PUBLIC_SITE_URL` with the value `https://yourdomain.com`, scope to
*Production* (and optionally *Preview*). Redeploy from the *Deployments*
tab.

## 6. (Optional) Vercel Analytics

A single line of code unlocks privacy-friendly traffic analytics:

```bash
npm install @vercel/analytics
```

Then in `app/layout.tsx`, import and render the component inside `<body>`:

```tsx
import { Analytics } from '@vercel/analytics/next'
// inside <body>:
<Analytics />
```

Free for the Hobby plan.

## Continuous deployment

Every push to `main` triggers a fresh production deploy. Every push to any
other branch creates a preview deployment with its own URL, which Vercel
will link from the GitHub pull request. This means:

- Writing Chapter 6 happens on a branch.
- The preview URL is shareable with readers for feedback before merge.
- Merging to `main` is the "ship it" action.

---

## Stack

Next.js 15 (App Router) · React 18 · TypeScript · Tailwind · MDX with
remark-math and rehype-katex · Source Serif 4 / Geist / JetBrains Mono ·
Pyodide for in-browser Python · CodeMirror 6 for editing.
