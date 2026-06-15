'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { LOSSES_2D, runOptimizer } from '@/lib/gradient'
import { computeContours } from '@/lib/contours'

const W = 720
const H = 420
const PAD = 30

const FN_LABELS: Record<string, string> = {
  bowl: 'Bowl — convex, isotropic',
  valley: 'Valley — convex, ill-conditioned',
  saddle: 'Saddle — passes through a non-minimum',
  himmelblau: 'Himmelblau — four local minima',
}

const DEFAULTS: Record<
  string,
  { start: [number, number]; lr: number; lrMax: number }
> = {
  bowl: { start: [2.5, 2.2], lr: 0.12, lrMax: 0.5 },
  valley: { start: [3.5, 1.1], lr: 0.05, lrMax: 0.25 },
  saddle: { start: [2.2, 0.25], lr: 0.1, lrMax: 0.5 },
  himmelblau: { start: [-3.7, -2.6], lr: 0.008, lrMax: 0.025 },
}

export default function GradientDescent2D() {
  const [lossName, setLossName] = useState('valley')
  const [start, setStart] = useState<[number, number]>(DEFAULTS.valley.start)
  const [lr, setLr] = useState(DEFAULTS.valley.lr)
  const [frameIdx, setFrameIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const loss = LOSSES_2D[lossName]
  const [[xMin, yMin], [xMax, yMax]] = loss.domain
  const lrMax = DEFAULTS[lossName].lrMax

  const frames = useMemo(
    () => runOptimizer(loss, start, { lr, maxSteps: 80, tol: 1e-4 }),
    [loss, start, lr],
  )

  const contours = useMemo(
    () => computeContours(loss, loss.contourLevels ?? [], 80),
    [loss],
  )

  useEffect(() => {
    setFrameIdx(0)
    setPlaying(false)
  }, [frames])

  useEffect(() => {
    if (!playing) return
    if (frameIdx >= frames.length - 1) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setFrameIdx((i) => i + 1), 220)
    return () => clearTimeout(t)
  }, [playing, frameIdx, frames.length])

  const sx = (x: number) =>
    PAD + ((x - xMin) / (xMax - xMin)) * (W - 2 * PAD)
  const sy = (y: number) =>
    H - PAD - ((y - yMin) / (yMax - yMin)) * (H - 2 * PAD)

  const svgRef = useRef<SVGSVGElement>(null)

  function handleClick(e: React.MouseEvent<SVGSVGElement>) {
    const svg = svgRef.current
    if (!svg) return
    const pt = svg.createSVGPoint()
    pt.x = e.clientX
    pt.y = e.clientY
    const { x, y } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
    if (x < PAD || x > W - PAD || y < PAD || y > H - PAD) return
    const domX = xMin + ((x - PAD) / (W - 2 * PAD)) * (xMax - xMin)
    const domY = yMin + ((H - PAD - y) / (H - 2 * PAD)) * (yMax - yMin)
    setStart([domX, domY])
  }

  function handleFnChange(name: string) {
    const d = DEFAULTS[name]
    setLossName(name)
    setStart(d.start)
    setLr(d.lr)
  }

  // Opacity calibration so lower-loss contours are more visible
  const levels = contours.map((c) => c.level)
  const lvlMin = Math.min(...levels)
  const lvlMax = Math.max(...levels)
  const opacityForLevel = (lvl: number) => {
    if (levels.length <= 1) return 0.4
    const t = (lvl - lvlMin) / (lvlMax - lvlMin)
    return 0.18 + (1 - t) * 0.42
  }

  // Validate trajectory for display (filter diverged frames)
  const visibleFrames = frames.filter((f) => !f.diverged)
  const diverged = frames[frames.length - 1]?.diverged ?? false
  const safeIdx = Math.min(frameIdx, visibleFrames.length - 1)
  const cur = visibleFrames[safeIdx] ?? visibleFrames[0]

  const trajPath = visibleFrames
    .slice(0, safeIdx + 1)
    .map((f) => `${sx(f.position[0]).toFixed(2)},${sy(f.position[1]).toFixed(2)}`)
    .join(' ')

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Loss surface</span>
          <select
            value={lossName}
            onChange={(e) => handleFnChange(e.target.value)}
            className="bg-paper border border-rule rounded px-2 py-1 text-ink"
          >
            {Object.keys(LOSSES_2D).map((n) => (
              <option key={n} value={n}>
                {FN_LABELS[n] ?? n}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Learning rate ={' '}
            <span className="font-mono">{lr.toFixed(3)}</span>
          </span>
          <input
            type="range"
            min={0.001}
            max={lrMax}
            step={0.001}
            value={lr}
            onChange={(e) => setLr(parseFloat(e.target.value))}
            className="w-40"
          />
        </label>
        <span className="text-ink-muted text-xs italic">
          Click anywhere on the surface to choose a starting position.
        </span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-auto block cursor-crosshair"
          onClick={handleClick}
        >
          {/* contour lines */}
          {contours.map((c, i) => (
            <g
              key={i}
              stroke="var(--ink)"
              strokeWidth={1}
              fill="none"
              opacity={opacityForLevel(c.level)}
            >
              {c.segments.map(([x1, y1, x2, y2], si) => (
                <line
                  key={si}
                  x1={sx(x1)}
                  y1={sy(y1)}
                  x2={sx(x2)}
                  y2={sy(y2)}
                />
              ))}
            </g>
          ))}

          {/* global minimum marker */}
          {loss.globalMin && (
            <g>
              <circle
                cx={sx(loss.globalMin[0])}
                cy={sy(loss.globalMin[1])}
                r={6}
                fill="none"
                stroke="var(--accent)"
                strokeWidth={1.5}
              />
              <circle
                cx={sx(loss.globalMin[0])}
                cy={sy(loss.globalMin[1])}
                r={1.5}
                fill="var(--accent)"
              />
            </g>
          )}

          {/* local minima markers */}
          {loss.localMinima?.map((p, i) => (
            <g key={i}>
              <circle
                cx={sx(p[0])}
                cy={sy(p[1])}
                r={5}
                fill="none"
                stroke="var(--accent)"
                strokeWidth={1.2}
                opacity={0.8}
              />
              <circle
                cx={sx(p[0])}
                cy={sy(p[1])}
                r={1.2}
                fill="var(--accent)"
                opacity={0.8}
              />
            </g>
          ))}

          {/* trajectory polyline */}
          {trajPath && safeIdx > 0 && (
            <polyline
              points={trajPath}
              fill="none"
              stroke="var(--accent)"
              strokeWidth={1.75}
              opacity={0.85}
            />
          )}

          {/* trajectory dots at each step (subtle) */}
          {visibleFrames.slice(0, safeIdx + 1).map((f, i) => (
            <circle
              key={i}
              cx={sx(f.position[0])}
              cy={sy(f.position[1])}
              r={1.5}
              fill="var(--accent)"
              opacity={0.7}
            />
          ))}

          {/* starting position */}
          <circle
            cx={sx(start[0])}
            cy={sy(start[1])}
            r={4.5}
            fill="var(--paper)"
            stroke="var(--accent)"
            strokeWidth={2}
          />

          {/* current position */}
          {cur && safeIdx > 0 && (
            <circle
              cx={sx(cur.position[0])}
              cy={sy(cur.position[1])}
              r={5.5}
              fill="var(--accent)"
              stroke="var(--paper)"
              strokeWidth={1.5}
            />
          )}
        </svg>
      </div>

      {/* Playback */}
      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button
          onClick={() => {
            if (frameIdx >= frames.length - 1) setFrameIdx(0)
            setPlaying((p) => !p)
          }}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[60px]"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={() => {
            setFrameIdx(0)
            setPlaying(false)
          }}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          Reset
        </button>
        <button
          onClick={() =>
            setFrameIdx((i) => Math.min(i + 1, frames.length - 1))
          }
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          Step
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(frames.length - 1, 1)}
          value={frameIdx}
          onChange={(e) => {
            setPlaying(false)
            setFrameIdx(parseInt(e.target.value))
          }}
          className="flex-1"
        />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">
          step {frameIdx} / {frames.length - 1}
        </span>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>
          θ = ({cur?.position[0].toFixed(3) ?? '—'},{' '}
          {cur?.position[1].toFixed(3) ?? '—'})
        </div>
        <div>
          loss ={' '}
          {cur && Number.isFinite(cur.loss)
            ? cur.loss.toFixed(4)
            : diverged
              ? '∞ (diverged)'
              : '—'}
        </div>
        <div>
          ‖∇L‖ ={' '}
          {cur ? Math.hypot(...cur.gradient).toFixed(4) : '—'}
        </div>
      </div>

      {diverged && (
        <p className="mt-2 font-sans text-xs text-ink-muted italic">
          ⚠ The trajectory diverged — the learning rate is too large for this
          surface. Try lowering it.
        </p>
      )}

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 6.2 — Gradient descent on a 2D loss surface. Contours are level
        sets of the loss; the ring marks a minimum. Each dot along the path
        is one step. Click anywhere to relocate the start.
      </figcaption>
    </figure>
  )
}
