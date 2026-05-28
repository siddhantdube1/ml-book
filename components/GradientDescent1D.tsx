'use client'

import { useEffect, useMemo, useState } from 'react'
import { LOSSES_1D, runOptimizer } from '@/lib/gradient'

const W = 640
const H = 300
const PAD_X = 36
const PAD_Y = 28

const FN_LABELS: Record<string, string> = {
  parabola: 'Parabola — f(x) = x²',
  'double-well': 'Double-well — f(x) = x⁴ − 2x² + 1',
  absolute: 'Absolute value — f(x) = |x|',
}

export default function GradientDescent1D() {
  const [lossName, setLossName] = useState<string>('parabola')
  const [startX, setStartX] = useState(2.4)
  const [lr, setLr] = useState(0.15)
  const [frameIdx, setFrameIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const loss = LOSSES_1D[lossName]
  const xMin = loss.domain[0][0]
  const xMax = loss.domain[1][0]

  const frames = useMemo(
    () => runOptimizer(loss, [startX], { lr, maxSteps: 60, tol: 1e-5 }),
    [loss, startX, lr],
  )

  const { yMin, yMax } = useMemo(() => {
    let lo = Infinity
    let hi = -Infinity
    const N = 200
    for (let i = 0; i <= N; i++) {
      const x = xMin + (i / N) * (xMax - xMin)
      const y = loss.eval([x])
      if (y < lo) lo = y
      if (y > hi) hi = y
    }
    const pad = (hi - lo) * 0.08 || 0.5
    return { yMin: lo - pad, yMax: hi + pad }
  }, [loss, xMin, xMax])

  const sx = (x: number) =>
    PAD_X + ((x - xMin) / (xMax - xMin)) * (W - 2 * PAD_X)
  const sy = (y: number) =>
    H - PAD_Y - ((y - yMin) / (yMax - yMin)) * (H - 2 * PAD_Y)

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
    const t = setTimeout(() => setFrameIdx((i) => i + 1), 380)
    return () => clearTimeout(t)
  }, [playing, frameIdx, frames.length])

  const fnPath = useMemo(() => {
    const N = 240
    const pts: string[] = []
    for (let i = 0; i <= N; i++) {
      const x = xMin + (i / N) * (xMax - xMin)
      const y = loss.eval([x])
      pts.push(`${i === 0 ? 'M' : 'L'} ${sx(x).toFixed(2)} ${sy(y).toFixed(2)}`)
    }
    return pts.join(' ')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loss, xMin, xMax, yMin, yMax])

  const cur = frames[Math.min(frameIdx, frames.length - 1)]
  const curX = cur.position[0]
  const curY = cur.loss
  const curG = cur.gradient[0]

  // Tangent line spanning a fixed fraction of the domain on each side
  const halfSpan = (xMax - xMin) * 0.09
  const tx1 = curX - halfSpan
  const tx2 = curX + halfSpan
  const ty1 = curY - halfSpan * curG
  const ty2 = curY + halfSpan * curG

  // Next step (where we land after one update)
  const nextX = curX - lr * curG

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Function</span>
          <select
            value={lossName}
            onChange={(e) => setLossName(e.target.value)}
            className="bg-paper border border-rule rounded px-2 py-1 text-ink"
          >
            {Object.keys(LOSSES_1D).map((n) => (
              <option key={n} value={n}>
                {FN_LABELS[n] ?? n}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Start x = <span className="font-mono">{startX.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={xMin}
            max={xMax}
            step={0.05}
            value={startX}
            onChange={(e) => setStartX(parseFloat(e.target.value))}
            className="w-32"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Learning rate = <span className="font-mono">{lr.toFixed(3)}</span>
          </span>
          <input
            type="range"
            min={0.005}
            max={0.5}
            step={0.005}
            value={lr}
            onChange={(e) => setLr(parseFloat(e.target.value))}
            className="w-32"
          />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* x-axis */}
          <line
            x1={sx(xMin)}
            y1={sy(0)}
            x2={sx(xMax)}
            y2={sy(0)}
            stroke="var(--rule)"
            strokeWidth={1}
          />
          {/* axis labels */}
          <text
            x={sx(xMax) - 4}
            y={sy(0) - 6}
            fontSize={11}
            fill="var(--ink-muted)"
            textAnchor="end"
            fontFamily="var(--font-mono, monospace)"
          >
            x
          </text>

          {/* function curve */}
          <path d={fnPath} fill="none" stroke="var(--ink)" strokeWidth={2} />

          {/* trail: small ticks on x-axis for past positions */}
          {frames.slice(0, frameIdx).map((f, i) => (
            <line
              key={i}
              x1={sx(f.position[0])}
              y1={sy(0) - 3}
              x2={sx(f.position[0])}
              y2={sy(0) + 3}
              stroke="var(--accent)"
              strokeWidth={1}
              opacity={0.25 + (0.6 * (i + 1)) / Math.max(frameIdx, 1)}
            />
          ))}

          {/* drop line from current x to f(x) */}
          <line
            x1={sx(curX)}
            y1={sy(0)}
            x2={sx(curX)}
            y2={sy(curY)}
            stroke="var(--accent)"
            strokeWidth={1}
            strokeDasharray="2,2"
            opacity={0.5}
          />

          {/* tangent line at current point */}
          <line
            x1={sx(tx1)}
            y1={sy(ty1)}
            x2={sx(tx2)}
            y2={sy(ty2)}
            stroke="var(--accent)"
            strokeWidth={1.5}
            opacity={0.7}
          />

          {/* next-step arrow on x-axis */}
          {Math.abs(nextX - curX) > 0.01 && (
            <g>
              <line
                x1={sx(curX)}
                y1={sy(0)}
                x2={sx(nextX)}
                y2={sy(0)}
                stroke="var(--accent)"
                strokeWidth={2.5}
              />
              <polygon
                points={`${sx(nextX)},${sy(0)} ${sx(nextX) - (nextX > curX ? 6 : -6)},${sy(0) - 3} ${sx(nextX) - (nextX > curX ? 6 : -6)},${sy(0) + 3}`}
                fill="var(--accent)"
              />
            </g>
          )}

          {/* current point */}
          <circle
            cx={sx(curX)}
            cy={sy(curY)}
            r={5}
            fill="var(--accent)"
            stroke="var(--paper)"
            strokeWidth={1.5}
          />
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
        <div>x = {curX.toFixed(4)}</div>
        <div>f(x) = {Number.isFinite(curY) ? curY.toFixed(4) : '∞ (diverged)'}</div>
        <div>f′(x) = {curG.toFixed(4)}</div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 6.1 — Gradient descent on a 1D function. The dot is the current
        x; the line through it is the tangent (its slope is the gradient);
        the horizontal arrow shows the next step, which is −α times the slope.
      </figcaption>
    </figure>
  )
}
