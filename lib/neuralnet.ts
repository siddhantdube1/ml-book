import { createRng, gauss } from './rng'

// ─── Activations ─────────────────────────────────────────────────────

export type Activation = 'step' | 'sigmoid' | 'tanh' | 'relu' | 'leaky' | 'linear'

export function sigmoid(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z))
}

export function activate(kind: Activation, z: number): number {
  switch (kind) {
    case 'step':
      return z >= 0 ? 1 : 0
    case 'sigmoid':
      return sigmoid(z)
    case 'tanh':
      return Math.tanh(z)
    case 'relu':
      return Math.max(0, z)
    case 'leaky':
      return z > 0 ? z : 0.01 * z
    case 'linear':
      return z
  }
}

export function activateDeriv(kind: Activation, z: number): number {
  switch (kind) {
    case 'step':
      return 0
    case 'sigmoid': {
      const s = sigmoid(z)
      return s * (1 - s)
    }
    case 'tanh': {
      const t = Math.tanh(z)
      return 1 - t * t
    }
    case 'relu':
      return z > 0 ? 1 : 0
    case 'leaky':
      return z > 0 ? 1 : 0.01
    case 'linear':
      return 1
  }
}

// ─── Data ────────────────────────────────────────────────────────────

export type Point = { x: [number, number]; y: 0 | 1 }

export function makeXORData(n: number, seed: number): Point[] {
  const rng = createRng(seed)
  const centres: [number, number][] = [[1.4, 1.4], [-1.4, 1.4], [-1.4, -1.4], [1.4, -1.4]]
  const out: Point[] = []
  for (let i = 0; i < n; i++) {
    const q = i % 4
    const [cx, cy] = centres[q]
    const y: 0 | 1 = cx > 0 !== cy > 0 ? 1 : 0
    out.push({ x: [gauss(rng, cx, 0.5), gauss(rng, cy, 0.5)], y })
  }
  return shuffle(out, rng)
}

export function makeMoonsData(n: number, seed: number, noise = 0.2): Point[] {
  const rng = createRng(seed)
  const out: Point[] = []
  const half = Math.floor(n / 2)
  const S = 1.7
  for (let i = 0; i < half; i++) {
    const t = (i / Math.max(half - 1, 1)) * Math.PI
    out.push({ x: [S * (Math.cos(t) - 0.5) + gauss(rng, 0, noise), S * (Math.sin(t) - 0.25) + gauss(rng, 0, noise)], y: 0 })
  }
  for (let i = 0; i < n - half; i++) {
    const t = (i / Math.max(half - 1, 1)) * Math.PI
    out.push({ x: [S * (0.5 - Math.cos(t)) + gauss(rng, 0, noise), S * (0.25 - Math.sin(t)) + gauss(rng, 0, noise)], y: 1 })
  }
  return shuffle(out, rng)
}

export function makeCirclesData(n: number, seed: number, noise = 0.16): Point[] {
  const rng = createRng(seed)
  const out: Point[] = []
  const half = Math.floor(n / 2)
  for (let i = 0; i < n; i++) {
    const inner = i < half
    const r = (inner ? 0.9 : 2.2) + gauss(rng, 0, noise)
    const t = 2 * Math.PI * rng()
    out.push({ x: [r * Math.cos(t), r * Math.sin(t)], y: inner ? 0 : 1 })
  }
  return shuffle(out, rng)
}

function shuffle<T>(arr: T[], rng: () => number): T[] {
  const a = arr.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

// ─── Single neuron / perceptron ──────────────────────────────────────

export type Neuron = { w: [number, number]; b: number }

export function neuronOutput(neuron: Neuron, x: [number, number], act: Activation): number {
  return activate(act, neuron.w[0] * x[0] + neuron.w[1] * x[1] + neuron.b)
}

export type PerceptronFrame = { w: [number, number]; b: number; errors: number; picked: number | null }

function perceptronErrors(data: Point[], w: [number, number], b: number): number {
  let e = 0
  for (const p of data) if ((w[0] * p.x[0] + w[1] * p.x[1] + b >= 0 ? 1 : 0) !== p.y) e++
  return e
}

/**
 * The perceptron learning rule. Records a frame after every weight update, so
 * the playback shows the boundary nudge on each mistake. Converges on linearly
 * separable data; thrashes forever (capped) on XOR.
 */
export function trainPerceptron(data: Point[], epochs: number, lr: number): PerceptronFrame[] {
  let w: [number, number] = [0.4, -0.6]
  let b = 0
  const frames: PerceptronFrame[] = [{ w: [...w] as [number, number], b, errors: perceptronErrors(data, w, b), picked: null }]
  for (let e = 0; e < epochs; e++) {
    let updated = false
    for (let i = 0; i < data.length; i++) {
      const p = data[i]
      const pred = w[0] * p.x[0] + w[1] * p.x[1] + b >= 0 ? 1 : 0
      if (pred !== p.y) {
        const g = p.y - pred
        w = [w[0] + lr * g * p.x[0], w[1] + lr * g * p.x[1]]
        b += lr * g
        frames.push({ w: [...w] as [number, number], b, errors: perceptronErrors(data, w, b), picked: i })
        updated = true
        if (frames.length > 600) return frames
      }
    }
    if (!updated) break // converged: a full pass with no mistakes
  }
  return frames
}

// ─── Multilayer perceptron (forward + backprop training) ─────────────

export type MLP = { W: number[][][]; b: number[][]; act: Activation }

function forwardFull(net: MLP, x: number[]): { z: number[][]; a: number[][] } {
  const a: number[][] = [x]
  const z: number[][] = []
  const L = net.W.length
  for (let l = 0; l < L; l++) {
    const prev = a[l]
    const zl = net.W[l].map((row, i) => {
      let s = net.b[l][i]
      for (let j = 0; j < row.length; j++) s += row[j] * prev[j]
      return s
    })
    z.push(zl)
    const isOut = l === L - 1
    a.push(zl.map((v) => (isOut ? sigmoid(v) : activate(net.act, v))))
  }
  return { z, a }
}

export function mlpForward(net: MLP, x: number[]): { activations: number[][]; output: number } {
  const { a } = forwardFull(net, x)
  return { activations: a, output: a[a.length - 1][0] }
}

/** Hidden-layer activations for a given input (layer 0 = first hidden layer). */
export function hiddenActivations(net: MLP, x: number[], layer = 0): number[] {
  const { a } = forwardFull(net, x)
  return a[layer + 1]
}

export type MLPFrame = { net: MLP; loss: number; accuracy: number; epoch: number; valLoss?: number }

// ─── Backprop trace (one example, for the visualisation) ─────────────

export type BackpropTrace = {
  z: number[][] // pre-activations per layer
  a: number[][] // activations per layer (a[0] = input)
  delta: number[][] // dL/dz per layer — the backward error
  gradW: number[][][] // dL/dW per layer
  gradB: number[][] // dL/db per layer
  output: number
  loss: number
}

/** Forward then backward through the net for a single labelled example. */
export function backpropTrace(net: MLP, x: number[], y: 0 | 1): BackpropTrace {
  const { z, a } = forwardFull(net, x)
  const L = net.W.length
  const output = a[L][0]
  const loss = -(y * Math.log(output + 1e-9) + (1 - y) * Math.log(1 - output + 1e-9))
  const delta: number[][] = new Array(L)
  const gradW: number[][][] = new Array(L)
  const gradB: number[][] = new Array(L)
  let d = [output - y]
  for (let l = L - 1; l >= 0; l--) {
    delta[l] = d
    gradB[l] = d.slice()
    gradW[l] = d.map((di) => a[l].map((aj) => di * aj))
    if (l > 0) {
      const nIn = net.W[l][0].length
      const nd = new Array(nIn).fill(0)
      for (let i = 0; i < d.length; i++) for (let j = 0; j < nIn; j++) nd[j] += net.W[l][i][j] * d[i]
      for (let j = 0; j < nIn; j++) nd[j] *= activateDeriv(net.act, z[l - 1][j])
      d = nd
    }
  }
  return { z, a, delta, gradW, gradB, output, loss }
}

function cloneNet(net: MLP): MLP {
  return { W: net.W.map((m) => m.map((r) => r.slice())), b: net.b.map((r) => r.slice()), act: net.act }
}

function initNet(layers: number[], act: Activation, rng: () => number): MLP {
  const W: number[][][] = []
  const b: number[][] = []
  for (let l = 1; l < layers.length; l++) {
    const nIn = layers[l - 1]
    const nOut = layers[l]
    const scale = Math.sqrt(2 / nIn)
    W.push(Array.from({ length: nOut }, () => Array.from({ length: nIn }, () => gauss(rng, 0, scale))))
    b.push(new Array(nOut).fill(0))
  }
  return { W, b, act }
}

function lossAndAcc(net: MLP, data: Point[]): { loss: number; acc: number } {
  let loss = 0
  let ok = 0
  for (const p of data) {
    const out = mlpForward(net, p.x).output
    const yy = p.y
    loss += -(yy * Math.log(out + 1e-9) + (1 - yy) * Math.log(1 - out + 1e-9))
    if ((out >= 0.5 ? 1 : 0) === yy) ok++
  }
  return { loss: loss / data.length, acc: ok / data.length }
}

export type Optimizer = 'sgd' | 'momentum' | 'adam'

export type MLPTrainOptions = {
  epochs: number
  lr: number
  act: Activation
  seed: number
  optimizer?: Optimizer
  batchSize?: number
  weightDecay?: number
  valData?: Point[]
  recordEvery?: number
}

type Grads = { gW: number[][][]; gB: number[][] }

function zerosLike(net: MLP): Grads {
  return { gW: net.W.map((m) => m.map((r) => r.map(() => 0))), gB: net.b.map((r) => r.map(() => 0)) }
}

/** Accumulate backprop gradients for one example into (gW, gB). */
function accumulateGrad(net: MLP, p: Point, g: Grads): void {
  const L = net.W.length
  const { z, a } = forwardFull(net, p.x)
  let delta = [a[L][0] - p.y]
  for (let l = L - 1; l >= 0; l--) {
    for (let i = 0; i < delta.length; i++) {
      g.gB[l][i] += delta[i]
      for (let j = 0; j < a[l].length; j++) g.gW[l][i][j] += delta[i] * a[l][j]
    }
    if (l > 0) {
      const nIn = net.W[l][0].length
      const nd = new Array(nIn).fill(0)
      for (let i = 0; i < delta.length; i++) for (let j = 0; j < nIn; j++) nd[j] += net.W[l][i][j] * delta[i]
      for (let j = 0; j < nIn; j++) nd[j] *= activateDeriv(net.act, z[l - 1][j])
      delta = nd
    }
  }
}

/**
 * Train a binary MLP by backprop on the binary cross-entropy loss (sigmoid
 * output). Supports SGD / momentum / Adam, mini-batches, L2 weight decay, and
 * optional validation-loss recording. Returns a snapshot frame history for
 * scrubbable playback.
 */
export function trainMLP(data: Point[], hidden: number[], opts: MLPTrainOptions): MLPFrame[] {
  const rng = createRng(opts.seed)
  const net = initNet([2, ...hidden, 1], opts.act, rng)
  const L = net.W.length
  const opt = opts.optimizer ?? 'sgd'
  const wd = opts.weightDecay ?? 0
  const batchSize = opts.batchSize ?? data.length
  const recordEvery = opts.recordEvery ?? Math.max(1, Math.floor(opts.epochs / 60))
  const frames: MLPFrame[] = []

  // optimiser state
  const vel = zerosLike(net) // momentum velocity / Adam first moment
  const sq = zerosLike(net) // Adam second moment
  let tStep = 0
  const beta1 = 0.9, beta2 = 0.999, adamEps = 1e-8, mu = 0.9

  const snap = (epoch: number) => {
    const { loss, acc } = lossAndAcc(net, data)
    const f: MLPFrame = { net: cloneNet(net), loss, accuracy: acc, epoch }
    if (opts.valData) f.valLoss = lossAndAcc(net, opts.valData).loss
    frames.push(f)
  }
  snap(0)

  const idx = data.map((_, i) => i)
  for (let e = 1; e <= opts.epochs; e++) {
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      ;[idx[i], idx[j]] = [idx[j], idx[i]]
    }
    for (let start = 0; start < idx.length; start += batchSize) {
      const batch = idx.slice(start, start + batchSize)
      const g = zerosLike(net)
      for (const bi of batch) accumulateGrad(net, data[bi], g)
      tStep++
      const n = batch.length
      for (let l = 0; l < L; l++) {
        for (let i = 0; i < net.W[l].length; i++) {
          // bias
          {
            const grad = g.gB[l][i] / n
            const upd = optimStep(grad, vel.gB[l], sq.gB[l], i, opt, opts.lr, beta1, beta2, adamEps, mu, tStep)
            net.b[l][i] -= upd
          }
          for (let j = 0; j < net.W[l][i].length; j++) {
            const grad = g.gW[l][i][j] / n + wd * net.W[l][i][j]
            const upd = optimStep(grad, vel.gW[l][i], sq.gW[l][i], j, opt, opts.lr, beta1, beta2, adamEps, mu, tStep)
            net.W[l][i][j] -= upd
          }
        }
      }
    }
    if (e % recordEvery === 0 || e === opts.epochs) snap(e)
  }
  return frames
}

function optimStep(
  grad: number, m: number[], v: number[], k: number, opt: Optimizer,
  lr: number, beta1: number, beta2: number, eps: number, mu: number, t: number,
): number {
  if (opt === 'momentum') {
    m[k] = mu * m[k] + grad
    return lr * m[k]
  }
  if (opt === 'adam') {
    m[k] = beta1 * m[k] + (1 - beta1) * grad
    v[k] = beta2 * v[k] + (1 - beta2) * grad * grad
    const mh = m[k] / (1 - Math.pow(beta1, t))
    const vh = v[k] / (1 - Math.pow(beta2, t))
    return (lr * mh) / (Math.sqrt(vh) + eps)
  }
  return lr * grad // sgd
}

// ─── Loss surface over two chosen weights (for the landscape widget) ─

export type WeightRef = { layer: number; i: number; j: number } // j < 0 → bias i

function getW(net: MLP, r: WeightRef): number {
  return r.j < 0 ? net.b[r.layer][r.i] : net.W[r.layer][r.i][r.j]
}
function setW(net: MLP, r: WeightRef, v: number): void {
  if (r.j < 0) net.b[r.layer][r.i] = v
  else net.W[r.layer][r.i][r.j] = v
}

/** grid×grid loss values varying two weights around the net's current values. */
export function lossSurface(net: MLP, data: Point[], a1: WeightRef, a2: WeightRef, range: number, grid: number): { surface: number[][]; c1: number; c2: number } {
  const base = cloneNet(net)
  const c1 = getW(base, a1)
  const c2 = getW(base, a2)
  const surface: number[][] = []
  for (let i = 0; i < grid; i++) {
    const row: number[] = []
    setW(base, a1, c1 + ((i / (grid - 1)) * 2 - 1) * range)
    for (let j = 0; j < grid; j++) {
      setW(base, a2, c2 + ((j / (grid - 1)) * 2 - 1) * range)
      row.push(lossAndAcc(base, data).loss)
    }
    surface.push(row)
  }
  setW(base, a1, c1)
  setW(base, a2, c2)
  return { surface, c1, c2 }
}

/** Gradient descent restricted to two weights (finite-difference), for the
 *  landscape trajectory. Returns the path in (w1, w2) weight-value space. */
export function descendSurface(
  net: MLP, data: Point[], a1: WeightRef, a2: WeightRef,
  start: [number, number], lr: number, steps: number,
): [number, number][] {
  const base = cloneNet(net)
  let w1 = start[0]
  let w2 = start[1]
  const eps = 1e-3
  const L = (v1: number, v2: number): number => {
    setW(base, a1, v1)
    setW(base, a2, v2)
    return lossAndAcc(base, data).loss
  }
  const path: [number, number][] = [[w1, w2]]
  for (let s = 0; s < steps; s++) {
    const g1 = (L(w1 + eps, w2) - L(w1 - eps, w2)) / (2 * eps)
    const g2 = (L(w1, w2 + eps) - L(w1, w2 - eps)) / (2 * eps)
    w1 -= lr * g1
    w2 -= lr * g2
    path.push([w1, w2])
  }
  return path
}
