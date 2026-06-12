import { createRng, gauss } from './rng'

export type Point = [number, number]
export type DatasetShape = 'blobs' | 'aniso' | 'moons' | 'circles' | 'varied'

export function generateDataset(
  shape: DatasetShape,
  n: number,
  seed: number,
): Point[] {
  const rng = createRng(seed)
  switch (shape) {
    case 'blobs':
      return blobs(rng, n)
    case 'aniso':
      return aniso(rng, n)
    case 'moons':
      return moons(rng, n)
    case 'circles':
      return circles(rng, n)
    case 'varied':
      return varied(rng, n)
  }
}

/** Two concentric rings — a dense inner disc encircled by an outer ring. */
function circles(rng: () => number, n: number): Point[] {
  const cx = 340
  const cy = 200
  const half = Math.floor(n / 2)
  const out: Point[] = []
  for (let i = 0; i < n; i++) {
    const inner = i < half
    const r = (inner ? 52 : 132) + gauss(rng, 0, 7)
    const t = 2 * Math.PI * rng()
    out.push([cx + r * Math.cos(t), cy + r * Math.sin(t)])
  }
  return out
}

/** Three blobs of deliberately different density — DBSCAN's hard case. */
function varied(rng: () => number, n: number): Point[] {
  const specs: { c: Point; sd: number }[] = [
    { c: [180, 170], sd: 16 },
    { c: [380, 150], sd: 34 },
    { c: [500, 260], sd: 24 },
  ]
  const out: Point[] = []
  for (let i = 0; i < n; i++) {
    const s = specs[i % 3]
    out.push([gauss(rng, s.c[0], s.sd), gauss(rng, s.c[1], s.sd)])
  }
  return out
}

function blobs(rng: () => number, n: number): Point[] {
  const centres: Point[] = [
    [180, 110],
    [500, 120],
    [340, 270],
  ]
  const out: Point[] = []
  for (let i = 0; i < n; i++) {
    const c = centres[i % 3]
    out.push([gauss(rng, c[0], 28), gauss(rng, c[1], 28)])
  }
  return out
}

function aniso(rng: () => number, n: number): Point[] {
  const centres: Point[] = [
    [210, 140],
    [470, 140],
    [340, 260],
  ]
  const cosT = Math.cos(0.7)
  const sinT = Math.sin(0.7)
  const out: Point[] = []
  for (let i = 0; i < n; i++) {
    const c = centres[i % 3]
    const x0 = gauss(rng, 0, 75)
    const y0 = gauss(rng, 0, 12)
    out.push([c[0] + x0 * cosT - y0 * sinT, c[1] + x0 * sinT + y0 * cosT])
  }
  return out
}

function moons(rng: () => number, n: number): Point[] {
  const out: Point[] = []
  const cx = 340
  const cy = 200
  const r = 95
  const half = Math.floor(n / 2)
  for (let i = 0; i < half; i++) {
    const t = (i / (half - 1)) * Math.PI
    out.push([
      cx - 55 + r * Math.cos(t) + gauss(rng, 0, 6),
      cy + 25 - r * Math.sin(t) + gauss(rng, 0, 6),
    ])
  }
  for (let i = 0; i < n - half; i++) {
    const t = (i / (half - 1)) * Math.PI
    out.push([
      cx + 55 - r * Math.cos(t) + gauss(rng, 0, 6),
      cy - 25 + r * Math.sin(t) + gauss(rng, 0, 6),
    ])
  }
  return out
}
