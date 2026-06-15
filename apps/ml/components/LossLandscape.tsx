'use client'

import { useMemo, useRef, useState } from 'react'
import { trainMLP, lossSurface, descendSurface, makeXORData, type MLP, type WeightRef } from '@/lib/neuralnet'

const W = 420
const H = 420
const PAD = 16
const GRID = 50
const RANGE = 5
// the two input weights of the two hidden units, for the same input feature:
// swapping them is a network symmetry, so this slice is genuinely multi-valleyed.
const A1: WeightRef = { layer: 0, i: 0, j: 0 }
const A2: WeightRef = { layer: 0, i: 1, j: 0 }

export default function LossLandscape() {
  const svgRef = useRef<SVGSVGElement>(null)
  const data = useMemo(() => makeXORData(160, 3), [])
  // a fully-trained tiny net; zero the two sliced weights so the slice is
  // centred on the origin and both symmetric valleys fall inside the view.
  const net = useMemo<MLP>(() => {
    const trained = trainMLP(makeXORData(120, 3), [3], { epochs: 1200, lr: 0.6, act: 'tanh', seed: 7 })
    const n = trained[trained.length - 1].net
    const clone: MLP = { W: n.W.map((m) => m.map((r) => r.slice())), b: n.b.map((r) => r.slice()), act: n.act }
    clone.W[A1.layer][A1.i][A1.j] = 0
    clone.W[A2.layer][A2.i][A2.j] = 0
    return clone
  }, [])
  const { surface, c1, c2 } = useMemo(() => lossSurface(net, data, A1, A2, RANGE, GRID), [net, data])

  const [userStarts, setUserStarts] = useState<[number, number][]>([])

  // contrast-enhanced shading: clamp the high (ridge) end and take a sqrt so the
  // valley structure spreads across the colour range instead of washing out.
  const { lo, clampHi } = useMemo(() => {
    const sorted = [...surface.flat()].sort((a, b) => a - b)
    return { lo: sorted[0], clampHi: sorted[Math.floor(sorted.length * 0.7)] }
  }, [surface])

  // weight-value <-> screen
  const wx = (w1: number) => PAD + ((w1 - (c1 - RANGE)) / (2 * RANGE)) * (W - 2 * PAD)
  const wy = (w2: number) => PAD + ((w2 - (c2 - RANGE)) / (2 * RANGE)) * (H - 2 * PAD)

  const presets: [number, number][] = [
    [c1 - RANGE * 0.7, c2 - RANGE * 0.7],
    [c1 + RANGE * 0.7, c2 + RANGE * 0.7],
    [c1 - RANGE * 0.7, c2 + RANGE * 0.7],
  ]
  const trajectories = useMemo(
    () => [...presets, ...userStarts].map((s) => descendSurface(net, data, A1, A2, s, 0.7, 60)),
    [net, data, userStarts], // eslint-disable-line react-hooks/exhaustive-deps
  )

  const shade = (v: number) => {
    let t = (v - lo) / (clampHi - lo)
    t = Math.sqrt(Math.max(0, Math.min(1, t))) // 0 low loss .. 1 high
    const r = 29 + (244 - 29) * t, g = 109 + (238 - 109) * t, b = 94 + (226 - 94) * t
    return `rgb(${r.toFixed(0)},${g.toFixed(0)},${b.toFixed(0)})`
  }

  const cs = (W - 2 * PAD) / GRID
  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current
    if (!svg) return
    const pt = svg.createSVGPoint()
    pt.x = e.clientX; pt.y = e.clientY
    const { x, y } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
    const w1 = (c1 - RANGE) + ((x - PAD) / (W - 2 * PAD)) * 2 * RANGE
    const w2 = (c2 - RANGE) + ((y - PAD) / (H - 2 * PAD)) * 2 * RANGE
    if (Math.abs(w1 - c1) <= RANGE && Math.abs(w2 - c2) <= RANGE) setUserStarts((u) => [...u, [w1, w2]])
  }

  return (
    <figure className="my-10">
      <div className="flex items-center justify-between mb-3 font-sans text-sm">
        <span className="text-ink-muted">Click the surface to drop a starting point. Dark valleys = low loss; pale ridges = high. Axes are two of the network&rsquo;s weights.</span>
        {userStarts.length > 0 && <button onClick={() => setUserStarts([])} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Clear</button>}
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block cursor-crosshair" onClick={handleClick}>
          {/* loss surface heatmap (i varies w1 -> x, j varies w2 -> y) */}
          {surface.map((row, i) => row.map((v, j) => (
            <rect key={`${i}-${j}`} x={PAD + i * cs} y={PAD + j * cs} width={cs + 0.5} height={cs + 0.5} fill={shade(v)} shapeRendering="crispEdges" />
          )))}
          {/* trajectories */}
          {trajectories.map((path, k) => (
            <g key={k}>
              <polyline points={path.map(([a, b]) => `${wx(a).toFixed(1)},${wy(b).toFixed(1)}`).join(' ')} fill="none" stroke="var(--accent)" strokeWidth={1.75} />
              <circle cx={wx(path[0][0])} cy={wy(path[0][1])} r={4} fill="var(--paper)" stroke="var(--ink)" strokeWidth={1.5} />
              <circle cx={wx(path[path.length - 1][0])} cy={wy(path[path.length - 1][1])} r={4} fill="var(--accent)" stroke="var(--paper)" strokeWidth={1.25} />
            </g>
          ))}
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 22.3 — The landscape is not a bowl. This is a real slice of a tiny
        network&rsquo;s loss surface over two of its weights — and unlike the clean
        convex bowls of Parts II–III, it is bumpy, with more than one valley
        separated by ridges. Gradient descent (hollow start, filled end) just
        rolls downhill from wherever it begins, so different starts settle in
        different minima — there is no guarantee of the global best. The deep
        surprise of neural networks is that this barely matters: in high
        dimensions, good-enough valleys are everywhere, and descent reliably finds
        one. Click to try your own start.
      </figcaption>
    </figure>
  )
}
