# Design plan: Chapter 19 — Hierarchical and density-based clustering

> Detailed plan for Chapter 19, written before any code (per the recipe). The
> middle chapter of **Part VI — Unsupervised learning**, and the direct sequel
> to Chapter 18 (k-means). Where k-means demanded you guess the number of
> clusters, found only round blobs, and forced every point into a group, this
> chapter builds the two classical families that fix each of those failings: a
> *hierarchy* of nested clusters that needs no k, and a *density* rule that
> finds clusters of any shape and calls the rest noise.

---

## Pedagogical role

Chapter 18 introduced clustering through k-means: pick k, drop k centroids,
alternate assign-and-update until it settles. It is fast and intuitive, and it
has three sharp limitations that this chapter is organised around — k must be
chosen in advance; the clusters it finds are always convex blobs; and every
point, however far-flung, is dragged into some cluster. Each is a real obstacle
on real data, and each has a classical cure:

- **Hierarchical (agglomerative) clustering** never asks for k. It builds a
  whole tree of nested clusterings — every point alone at the bottom, one giant
  cluster at the top — and you read off whatever number of clusters you want by
  cutting the tree at a height. The tree itself, the *dendrogram*, is an object
  of real insight.
- **DBSCAN** redefines what a cluster *is*: not "near a centre" but "in a dense
  region". That single change lets it trace clusters of any shape (the moons
  and rings k-means mangles), discover the number of clusters on its own, and
  label sparse outliers as *noise* rather than forcing them in.

It compounds on the book:
- Chapter 18 (k-means) is the foil throughout — the three failures motivate
  both methods, and the closing showdown puts all three on the same data.
- Chapter 4 (kNN) and Chapter 17 (scaling): both methods are built on
  *distances*, so feature scaling matters exactly as much here, and the
  neighbourhood idea returns in DBSCAN's density.
- Chapter 11's spirit (choosing a model honestly) recurs as the practical
  question of choosing a clustering algorithm and its parameters when there is
  no label to score against.

It sets up Chapter 20 (PCA): having clustered points in their raw space, the
last unsupervised chapter learns to *transform* the space itself — the
dimensionality-reduction counterpart to this chapter's grouping.

---

## The conceptual arc (12 sections)

1. **Where k-means runs out.** Sensory hook: hand k-means the two interleaving
   moons of Chapter 7 and it slices straight through both, because it can only
   carve the plane into convex tiles around its centres. Name its three
   limitations — you must guess k, it finds only round blobs, and it has no
   notion of an outlier — and promise two methods that each dissolve them.

2. **Clustering as a hierarchy.** The first new idea. Rather than commit to one
   partition, build a *nested family* of them: start with every point its own
   cluster, repeatedly merge the two closest clusters, and keep going until a
   single cluster contains everything. The record of those merges is a tree,
   and every horizontal slice through it is a clustering. No k required up
   front.

3. **Agglomerative clustering, step by step.** The bottom-up algorithm in full:
   find the two closest clusters, merge them, repeat. Watch it run — points
   fusing into small groups, groups into larger ones — with the **dendrogram**
   growing alongside, each merge a new horizontal bar at the height of the
   distance it bridged. → **Widget 1: DendrogramBuilder**.

4. **Linkage: what does "closest" mean?** The one real choice in hierarchical
   clustering. Cluster-to-point distance is obvious; *cluster-to-cluster* is
   not, and the definition you pick changes everything. **Single** linkage
   (nearest pair) chains clusters along thin bridges; **complete** linkage
   (farthest pair) insists on tight, compact balls; **average** linkage
   compromises; **Ward** merges the pair that least increases within-cluster
   variance, the k-means-flavoured default. → **Widget 2: LinkageComparison**.

5. **Reading and cutting the dendrogram.** The tree is the output; a clustering
   is a cut. Slice the dendrogram horizontally at a chosen height and the
   branches you sever become your clusters — cut low for many small clusters,
   high for a few large ones. The big *vertical gaps* between merges are the
   natural places to cut, the visual signature of a "real" number of clusters,
   and the closest hierarchical clustering comes to telling you k.

6. **A different question: density.** The second new idea, and a genuine change
   of definition. Forget centres entirely. A cluster is a region where points
   are packed closely together, bounded by regions where they thin out — exactly
   how your eye finds the moons. Dense neighbourhoods grown outward trace
   whatever shape the data takes, and the empty spaces between become the
   borders.

7. **DBSCAN: core, border, and noise.** The algorithm that makes density
   precise, with two parameters: a radius $\varepsilon$ and a count `minPts`. A
   **core** point has at least `minPts` neighbours within $\varepsilon$; a
   **border** point is within $\varepsilon$ of a core but not itself dense; a
   **noise** point is neither, and is left unclustered. Clusters grow by
   chaining core points through one another's neighbourhoods. → **Widget 3:
   DBSCANPlayback**.

8. **Choosing $\varepsilon$ and minPts.** The practical handles. `minPts` sets
   how many points make a neighbourhood "dense" (a common default is $2 \times$
   the dimension); $\varepsilon$ sets how close is "close", and it is the
   sensitive one — too small and everything is noise, too large and separate
   clusters merge into one. The standard heuristic is the **k-distance plot**:
   sort every point's distance to its k-th neighbour and look for the elbow.
   And, as in Chapter 17, because it is all distances, **scale your features
   first**.

9. **Arbitrary shapes, and the showdown.** The payoff. Put k-means,
   hierarchical (single linkage), and DBSCAN side by side on blobs, moons, and
   concentric rings. On tidy round blobs all three agree; on the moons and rings
   k-means fails completely while DBSCAN traces them exactly — and DBSCAN quietly
   flags the stray points as noise. → **Widget 4: ShapeShowdown**.

10. **Choosing a clustering algorithm.** The decision guide and the costs.
    *k-means*: fast ($O(nki)$), needs k, convex blobs, all points assigned —
    reach for it first on large, blobby data. *Hierarchical*: no k up front and
    a full dendrogram to inspect, but $O(n^2 \log n)$ time and $O(n^2)$ memory
    confine it to smaller datasets. *DBSCAN*: any shape, finds its own cluster
    count, handles noise, no k — but struggles when clusters have very different
    densities, and lives or dies by $\varepsilon$. None is "best"; each answers
    a different question.

11. **Implementing it yourself.** DBSCAN from scratch in ~30 lines — a region
    query, a queue that grows each cluster through its core points, and a label
    for the noise — run on the moons it was built for.

12. **Problems.** Five, increasing difficulty (below).

### Problems (sketch)

1. **Conceptual — three cures for three failings.** (No code.) State k-means'
   three limitations and explain precisely how hierarchical clustering and
   DBSCAN each address them. Why can DBSCAN find the moons when k-means cannot?
   What does it mean for a point to be "noise", and which methods have that
   notion?
2. **Agglomerative merging by hand.** Given a handful of points and single
   linkage, carry out the merges in order and record the height of each — the
   dendrogram as a list. Confirm cutting at the largest gap gives the visually
   obvious clusters.
3. **DBSCAN from scratch.** Implement the region query and the cluster-growing
   loop; label core, border, and noise; run it on the moons and recover two
   clusters plus a handful of noise points.
4. **The $\varepsilon$ knob.** Sweep $\varepsilon$ on a fixed dataset and watch
   the cluster count swing from "all noise" through "the right answer" to "one
   blob"; draw the k-distance plot and find the elbow that picks a good value.
5. **Linkage and the chaining trap.** On data with a thin bridge between two
   groups, show that single linkage chains them into one cluster while complete
   or Ward keeps them apart — and connect this to DBSCAN's same vulnerability
   when $\varepsilon$ is too large. The capstone on what "closest" really means.

---

## The four visualisations (detailed)

All reuse Chapter 18's screen-coordinate datasets (`lib/datasets.ts`,
`Point = [number, number]`, ~680×360 viewBox) for visual continuity with
k-means, extended with concentric **circles** and a **variable-density** set.
New algorithms live in `lib/clustering.ts`.

### Widget 1 — `DendrogramBuilder` (hierarchical; build first)

- Two panels: **left**, the points in the plane, fusing into coloured groups as
  merges happen; **right**, the **dendrogram** drawing itself, one new bracket
  per merge at the height of the merged distance. **Playback** (Play / Step /
  Reset / scrub) walks the agglomeration from n singletons down to one cluster.
  A horizontal **cut line** (a slider on height) colours the points by the
  clusters that cut produces.
- Pedagogical job: hierarchical clustering is a *process* and a *tree*, and the
  tree contains every clustering at once. Seeing the points merge and the
  dendrogram grow in lock-step is the whole idea.

### Widget 2 — `LinkageComparison` (what "closest" means)

- The same dataset clustered four ways — **single**, **complete**, **average**,
  **Ward** linkage — shown as a 2×2 grid (or a linkage selector) cut to the same
  number of clusters, the points coloured by cluster in each. On a dataset with
  a thin bridge, single linkage visibly *chains* the two halves into one cluster
  while complete and Ward keep them compact and separate.
- Pedagogical job: the linkage rule is the substance of hierarchical
  clustering, and its failure mode (chaining) is famous and worth seeing.

### Widget 3 — `DBSCANPlayback` (density; core / border / noise)

- A 2-D dataset (moons by default) with two sliders, **$\varepsilon$** and
  **minPts**. Each point is drawn by its role — **core** (filled, cluster
  colour, with a faint $\varepsilon$-disc), **border** (ringed, cluster colour),
  **noise** (small, grey × ) — and clusters take distinct colours. Optional
  **playback** grows the clusters region-query by region-query, the frontier
  sweeping outward through the dense regions. The readout shows clusters found
  and noise count.
- Pedagogical job: density clustering made mechanical — what core/border/noise
  mean, how $\varepsilon$ and minPts move the boundary between signal and noise,
  and how a cluster is just core points chained through overlapping discs.

### Widget 4 — `ShapeShowdown` (k-means vs hierarchical vs DBSCAN)

- A dataset selector (**blobs / moons / circles / aniso**) and three panels side
  by side: **k-means**, **single-linkage hierarchical**, and **DBSCAN**, each
  clustering the same points. On blobs all three agree; switch to moons or
  circles and k-means' straight partition is exposed while DBSCAN (and
  single-linkage) trace the true shapes, DBSCAN also greying out the noise.
- Pedagogical job: the direct, side-by-side answer to "why not just use
  k-means?" — the chapter's thesis in one picture, and the callback that closes
  the clustering arc.

---

## Files to create

```
lib/
  datasets.ts                   ← EXTEND: add `circles` (two concentric rings)
                                  and `varied` (blobs of differing density) to
                                  DatasetShape, for DBSCAN / the showdown.
  clustering.ts                 ← NEW: agglomerative clustering with linkage +
                                  merge history (dendrogram), cut-at-k / cut-at-
                                  height, and DBSCAN (labels + core/border/noise).
components/
  DendrogramBuilder.tsx         ← Widget 1
  LinkageComparison.tsx         ← Widget 2
  DBSCANPlayback.tsx            ← Widget 3
  ShapeShowdown.tsx             ← Widget 4
app/chapters/
  19-hierarchical-density-clustering/
    page.mdx                    ← Full chapter
```

Wire-up: `app/page.tsx` (`href: '/chapters/19-hierarchical-density-clustering'`)
and `app/sitemap.ts`. **Proposed slug:
`19-hierarchical-density-clustering`** (the landing title is "Hierarchical and
density-based clustering").

Reused: `generateDataset` and the `Point = [number, number]` convention
(`lib/datasets.ts`), `runKMeans` (`lib/kmeans.ts`) for the showdown, the
playback pattern from Chapter 18's widgets, the cluster colour palette. No new
dependencies.

---

## Library design sketch

```ts
// ── lib/clustering.ts — new ───────────────────────────────────────────
import type { Point } from './datasets'

export type Linkage = 'single' | 'complete' | 'average' | 'ward'

/** One merge in the agglomeration: which two clusters joined, at what height. */
export type Merge = { a: number; b: number; height: number; size: number }

/** Full agglomerative run: the merge history (a dendrogram) over n leaves. */
export function agglomerative(points: Point[], linkage: Linkage): Merge[]

/** Cut the dendrogram into exactly k clusters → a label per point. */
export function cutTreeK(merges: Merge[], n: number, k: number): number[]
/** Cut at a height threshold → a label per point. */
export function cutTreeHeight(merges: Merge[], n: number, height: number): number[]

// ── DBSCAN ───────────────────────────────────────────────────────────
export type PointRole = 'core' | 'border' | 'noise'
export type DBSCANResult = {
  labels: number[]        // cluster id per point, -1 = noise
  roles: PointRole[]
  numClusters: number
}
export function dbscan(points: Point[], eps: number, minPts: number): DBSCANResult

/** Sorted k-th-nearest-neighbour distances, for the elbow heuristic. */
export function kDistances(points: Point[], k: number): number[]
```

Notes & numerical-verification plan (run in Node *before* building widgets):
- **Agglomerative:** verify on three clean blobs that cutting at k = 3 recovers
  the blobs for every linkage; verify single-linkage *chains* a bridged dataset
  into one cluster while complete/Ward split it; verify merge heights are
  monotonically non-decreasing (a valid dendrogram).
- **DBSCAN:** verify on the moons that sensible $(\varepsilon, \text{minPts})$
  recovers exactly two clusters with a small noise set; that tiny $\varepsilon$
  → all noise and huge $\varepsilon$ → one cluster; and that core/border/noise
  counts are self-consistent (every border is within $\varepsilon$ of a core).
- **Showdown:** confirm `runKMeans(k=2)` mis-clusters the moons (centroids land
  mid-arc, the split cuts both) while DBSCAN traces them — the contrast the
  widget depends on.
- **Performance:** agglomerative is $O(n^2 \log n)$ and DBSCAN $O(n^2)$ naive;
  keep widget datasets to ~120–200 points so a full run is single-digit
  milliseconds. Verify before shipping.
- **Determinism:** seeded datasets, deterministic tie-breaking in merges and
  region queries — no `Math.random`, no `any`, British spelling throughout.

---

## Visualisation colour palette

Reuse the established cluster palette (Chapter 18) — no new colours.

| Role | Colour | Use |
|---|---|---|
| Cluster A / B / C … | the Ch 18 cluster palette (teal, blue, orange, olive…) | all four widgets |
| Noise points | `var(--ink-faint)` small × | DBSCAN, showdown |
| Core vs border | filled vs ringed in cluster colour | DBSCAN |
| ε-disc | `var(--accent)` very faint fill | DBSCAN |
| Dendrogram brackets | `var(--ink)` / `var(--ink-muted)` | DendrogramBuilder |
| Cut line | `var(--accent)` dashed | DendrogramBuilder |
| Merge highlight / current | `var(--accent)` | playback widgets |

Fonts: `font-sans` UI/labels, `font-mono` numeric readouts. Figure numbering
19.1–19.4.

---

## Opening paragraph (draft, in voice)

> Hand k-means the two interleaving crescents from Chapter 7 — the moons every
> classifier in this book has cut its teeth on — and ask it for two clusters.
> It will think for a moment and then draw a straight line clean through the
> middle of both, splitting each crescent in half and pairing the wrong halves
> together. It is not a bug. k-means can only carve the plane into convex tiles
> around its centres, and no two such tiles trace a pair of crescents. The
> shape was always beyond it.
>
> That failure is one of three. k-means makes you name the number of clusters
> before it has looked at a single point; it can only find clusters that are
> round and roughly equal in size; and it forces every point, including the lone
> outlier halfway across the plane, into one cluster or another, because it has
> no concept of a point that simply does not belong. For tidy, blobby data none
> of this matters, and k-means is the right first tool. For everything else, we
> need clustering that asks a different question.
>
> This chapter builds two such methods, each dissolving k-means' limits from a
> different direction. The first stops trying to produce one answer and instead
> builds a whole tree of them — every point alone at the bottom, one cluster at
> the top, and every clustering you might want in between — so you choose the
> number of groups by where you cut, not before you start. The second throws out
> the idea of a centre altogether and defines a cluster as a *dense region* of
> the plane, which lets it trace the moons exactly, count its own clusters, and
> shrug off the outliers as noise. By the end you will know when to reach past
> k-means, and for which of the two.

## Closing paragraph (draft, in voice)

> The three clustering algorithms of these two chapters are not rivals so much
> as different questions asked of the same scatter of points. k-means asks
> "which of k centres is each point nearest?" — fast, simple, and blind to
> shape. Hierarchical clustering asks "in what order do points and groups fuse
> as we relax our standard for togetherness?" — and hands back a whole tree to
> read. DBSCAN asks "where is the plane crowded, and where is it empty?" — and
> traces clusters of any shape while naming the stragglers as noise. Knowing
> which question fits your data is most of the skill of clustering.
>
> All three, though, work in the space the data arrived in, taking its
> dimensions as given — and that is the last assumption Part VI has left to
> question. The features themselves might be redundant, correlated, or simply
> too many to see; the structure you are hunting might live in a handful of
> directions hidden among hundreds.
>
> ---
>
> *Next: Chapter 20 — PCA and dimensionality reduction.* The final unsupervised
> idea: not grouping the points, but reshaping the space they live in — finding
> the few directions that carry the data's real variation, and throwing the rest
> away.

---

## Expected scope

- Extend `lib/datasets.ts` (two new shapes) + one new lib (`lib/clustering.ts`,
  ~150 lines: agglomerative + linkages + dendrogram cutting, DBSCAN,
  k-distances) + four components.
- One MDX file (~850 lines, 12 sections, 5 problems).
- Wire-up in `app/page.tsx` and `app/sitemap.ts`.
- Build verification + smoke test (HTTP 200, KaTeX present, 5 problem
  headings, 5 `<details>`, 5 "Show solution", sitemap entry, no American
  spellings, no `Math.random`, no `any`).
- Bundle size: ~12–15 kB / ~270 kB First Load JS, in line with Ch 18's widgets.
- **Numerical verification first:** a Node port confirming the agglomerative
  merges and dendrogram cuts, single-linkage chaining, DBSCAN on the moons
  (two clusters + noise) and its $\varepsilon$ extremes, and the k-means-fails-
  on-moons contrast — before any component is built.

Total estimated work: a touch heavier than Chapter 18's later widgets — the
dendrogram rendering and the DBSCAN role-colouring are the two new pieces of
real work; the datasets, playback, and palette all carry over.

---

## Open questions for sign-off

1. **Slug** — `19-hierarchical-density-clustering` (the landing title is
   "Hierarchical and density-based clustering"). Good, or prefer something
   shorter like `19-clustering-shapes`?
2. **The four widgets** — DendrogramBuilder (hierarchical core) →
   LinkageComparison (single/complete/average/Ward) → DBSCANPlayback
   (core/border/noise) → ShapeShowdown (k-means vs hierarchical vs DBSCAN).
   Two hierarchical, one DBSCAN, one comparison. Happy with that balance, or
   would you swap LinkageComparison for a second DBSCAN widget (e.g. an
   interactive **k-distance / ε-elbow** picker)?
3. **Showdown contents** — three methods side by side (k-means, single-linkage
   hierarchical, DBSCAN). I lean on all three for the full contrast; happy to
   drop hierarchical from the showdown and show just k-means vs DBSCAN if you'd
   rather keep it to two clean panels. Assume three unless you object.
```
