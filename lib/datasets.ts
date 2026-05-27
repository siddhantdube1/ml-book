import { createRng, gauss } from './rng'

export type Point = [number, number]
export type DatasetShape = 'blobs' | 'aniso' | 'moons'

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
  }
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
