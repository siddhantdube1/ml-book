import { createRng, gauss } from './rng'

export type Vec = number[]

export type LossFn = {
  name: string
  /** Evaluate the loss at a point. */
  eval: (x: Vec) => number
  /** Analytic gradient at a point. */
  grad: (x: Vec) => Vec
  /** Bounding box [min, max] for plotting and divergence checks. */
  domain: [Vec, Vec]
  /** Optional: known global minimum location, for annotation. */
  globalMin?: Vec
  /** Optional: a list of local-minimum locations, for annotation. */
  localMinima?: Vec[]
  /** Suggested contour levels for plotting. */
  contourLevels?: number[]
}

export type GDFrame = {
  position: Vec
  gradient: Vec
  loss: number
  step: number
  /** True if this frame diverged out of the plotting domain. */
  diverged?: boolean
}

// ─── 1D loss functions ────────────────────────────────────────────────

export const PARABOLA_1D: LossFn = {
  name: 'parabola',
  eval: ([x]) => x * x,
  grad: ([x]) => [2 * x],
  domain: [[-3], [3]],
  globalMin: [0],
}

export const QUARTIC_1D: LossFn = {
  name: 'double-well',
  eval: ([x]) => x ** 4 - 2 * x * x + 1,
  grad: ([x]) => [4 * x ** 3 - 4 * x],
  domain: [[-2.2], [2.2]],
  localMinima: [[-1], [1]],
}

export const ABS_1D: LossFn = {
  name: 'absolute',
  eval: ([x]) => Math.abs(x),
  grad: ([x]) => [x > 0 ? 1 : x < 0 ? -1 : 0],
  domain: [[-3], [3]],
  globalMin: [0],
}

export const LOSSES_1D: Record<string, LossFn> = {
  parabola: PARABOLA_1D,
  'double-well': QUARTIC_1D,
  absolute: ABS_1D,
}

// ─── 2D loss functions ────────────────────────────────────────────────

export const BOWL_2D: LossFn = {
  name: 'bowl',
  eval: ([x, y]) => x * x + y * y,
  grad: ([x, y]) => [2 * x, 2 * y],
  domain: [[-3, -3], [3, 3]],
  globalMin: [0, 0],
  contourLevels: [0.25, 1, 2.25, 4, 6.25, 9],
}

export const VALLEY_2D: LossFn = {
  name: 'valley',
  eval: ([x, y]) => 0.25 * x * x + 4 * y * y,
  grad: ([x, y]) => [0.5 * x, 8 * y],
  domain: [[-4, -1.5], [4, 1.5]],
  globalMin: [0, 0],
  contourLevels: [0.25, 1, 2.25, 4, 6.25, 9],
}

export const SADDLE_2D: LossFn = {
  name: 'saddle',
  eval: ([x, y]) => x * x - y * y,
  grad: ([x, y]) => [2 * x, -2 * y],
  domain: [[-3, -3], [3, 3]],
  contourLevels: [-8, -4, -2, -1, 0, 1, 2, 4, 8],
}

export const HIMMELBLAU: LossFn = {
  name: 'himmelblau',
  eval: ([x, y]) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
  grad: ([x, y]) => [
    4 * x * (x * x + y - 11) + 2 * (x + y * y - 7),
    2 * (x * x + y - 11) + 4 * y * (x + y * y - 7),
  ],
  domain: [[-5, -5], [5, 5]],
  localMinima: [
    [3, 2],
    [-2.805, 3.131],
    [-3.779, -3.283],
    [3.584, -1.848],
  ],
  contourLevels: [2, 10, 30, 80, 180, 350, 600, 1000],
}

export const LOSSES_2D: Record<string, LossFn> = {
  bowl: BOWL_2D,
  valley: VALLEY_2D,
  saddle: SADDLE_2D,
  himmelblau: HIMMELBLAU,
}

// ─── Optimizer ────────────────────────────────────────────────────────

export type OptimizerOptions = {
  lr: number
  /** Heavy-ball momentum coefficient. 0 = vanilla gradient descent. */
  momentum?: number
  /** Standard deviation of Gaussian noise added to each gradient component. */
  noiseScale?: number
  /** RNG seed used when noiseScale > 0. */
  seed?: number
  maxSteps?: number
  /** Stop when ||gradient|| falls below this. */
  tol?: number
}

/**
 * Runs an iterative optimizer and returns the full frame history.
 * One function covers vanilla GD, GD-with-momentum, and SGD (and any
 * combination) via options. Frames include the position, gradient,
 * and loss at every step, ready for playback.
 */
export function runOptimizer(
  loss: LossFn,
  start: Vec,
  opts: OptimizerOptions,
): GDFrame[] {
  const {
    lr,
    momentum = 0,
    noiseScale = 0,
    seed = 0,
    maxSteps = 80,
    tol = 1e-4,
  } = opts

  const rng = createRng(seed)
  const [domainMin, domainMax] = loss.domain
  const divergeBound =
    Math.max(...domainMax.map((d, i) => Math.abs(d - domainMin[i]))) * 3

  const frames: GDFrame[] = []
  let x = start.slice()
  let v = x.map(() => 0)

  for (let s = 0; s <= maxSteps; s++) {
    const g = loss.grad(x)
    const noisyG = noiseScale > 0 ? g.map((gi) => gi + noiseScale * gauss(rng, 0, 1)) : g
    const l = loss.eval(x)

    const diverged =
      !Number.isFinite(l) ||
      x.some((xi, i) => Math.abs(xi) > Math.abs(domainMin[i]) + divergeBound)

    frames.push({
      position: x.slice(),
      gradient: g,
      loss: diverged ? Infinity : l,
      step: s,
      diverged,
    })

    if (diverged) break

    const gnorm = Math.hypot(...g)
    if (gnorm < tol && momentum === 0) break

    v = v.map((vi, i) => momentum * vi + noisyG[i])
    x = x.map((xi, i) => xi - lr * v[i])
  }

  return frames
}
