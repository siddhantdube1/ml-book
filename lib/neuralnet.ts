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

export type MLPFrame = { net: MLP; loss: number; accuracy: number; epoch: number }

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

export type MLPTrainOptions = { epochs: number; lr: number; act: Activation; seed: number; recordEvery?: number }

/**
 * Train a binary MLP by full-batch gradient descent (backprop) on the binary
 * cross-entropy loss with a sigmoid output. Returns a frame history (snapshots
 * every `recordEvery` epochs) for scrubbable playback of the boundary forming.
 */
export function trainMLP(data: Point[], hidden: number[], opts: MLPTrainOptions): MLPFrame[] {
  const rng = createRng(opts.seed)
  const layers = [2, ...hidden, 1]
  const net = initNet(layers, opts.act, rng)
  const L = net.W.length
  const recordEvery = opts.recordEvery ?? Math.max(1, Math.floor(opts.epochs / 60))
  const frames: MLPFrame[] = []

  const snap = (epoch: number) => {
    const { loss, acc } = lossAndAcc(net, data)
    frames.push({ net: cloneNet(net), loss, accuracy: acc, epoch })
  }
  snap(0)

  for (let e = 1; e <= opts.epochs; e++) {
    // accumulate gradients over the batch
    const gW = net.W.map((m) => m.map((r) => r.map(() => 0)))
    const gB = net.b.map((r) => r.map(() => 0))
    for (const p of data) {
      const { z, a } = forwardFull(net, p.x)
      let delta = [a[L][0] - p.y] // BCE + sigmoid output
      for (let l = L - 1; l >= 0; l--) {
        for (let i = 0; i < delta.length; i++) {
          gB[l][i] += delta[i]
          for (let j = 0; j < a[l].length; j++) gW[l][i][j] += delta[i] * a[l][j]
        }
        if (l > 0) {
          const nInPrev = net.W[l][0].length
          const newDelta = new Array(nInPrev).fill(0)
          for (let i = 0; i < delta.length; i++) for (let j = 0; j < nInPrev; j++) newDelta[j] += net.W[l][i][j] * delta[i]
          for (let j = 0; j < nInPrev; j++) newDelta[j] *= activateDeriv(net.act, z[l - 1][j])
          delta = newDelta
        }
      }
    }
    const scale = opts.lr / data.length
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < net.W[l].length; i++) {
        net.b[l][i] -= scale * gB[l][i]
        for (let j = 0; j < net.W[l][i].length; j++) net.W[l][i][j] -= scale * gW[l][i][j]
      }
    }
    if (e % recordEvery === 0 || e === opts.epochs) snap(e)
  }
  return frames
}
