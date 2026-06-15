'use client'

import { useEffect, useMemo, useState } from 'react'
import { LOSSES_2D, runOptimizer } from '@/lib/gradient'
import { computeContours } from '@/lib/contours'

const W = 720
const H = 360
const PAD = 30

const COLORS = [
  '#3c5a8c', // small LR — blue
  '#1d6d5e', // medium LR — teal (accent)
  '#c7522a', // large LR — orange
  '#8b3a62', // huge LR — magenta
]

const LR_FACTORS = [0.3, 1, 2.5, 5]
const LR_FACTOR_LABELS = ['×0.3', '×1', '×2.5', '×5']

const FN_LABELS: Record<string, string> = {
  bowl: 'Bowl',
  valley: 'Valley',
}

const DEFAULTS: Record<
  string,
  { start: [number, number]; baseLr: number }
> = {
  bowl: { start: [2.5, 2.2], baseLr: 0.06 },
  valley: { start: [3.5, 1.1], baseLr: 0.03 },
}

export default function LRComparison() {
  const [lossName, setLossName] = useState('valley')
  const [baseLr, setBaseLr] = useState(DEFAULTS.valley.baseLr)
  const [frameIdx, setFrameIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const loss = LOSSES_2D[lossName]
  const [[xMin, yMin], [xMax, yMax]] = loss.domain
  const start = DEFAULTS[lossName].start

  const trajectories = useMemo(
    () =>
      LR_FACTORS.map((f) =>
        runOptimizer(loss, start, {
          lr: baseLr * f,
          maxSteps: 80,
          tol: 1e-5,
        }),
      ),
    [loss, start, baseLr],
  )

  const maxFrames = Math.max(...trajectories.map((t) => t.length))

  const contours = useMemo(
    () => computeContours(loss, loss.contourLevels ?? [], 80),
    [loss],
  )

  useEffect(() => {
    setFrameIdx(0)
    setPlaying(false)
  }, [trajectories])

  useEffect(() => {
    if (!playing) return
    if (frameIdx >= maxFrames - 1) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setFrameIdx((i) => i + 1), 220)
    return () => clearTimeout(t)
  }, [playing, frameIdx, maxFrames])

  const sx = (x: number) =>
    PAD + ((x - xMin) / (xMax - xMin)) * (W - 2 * PAD)
  const sy = (y: number) =>
    H - PAD - ((y - yMin) / (yMax - yMin)) * (H - 2 * PAD)

  function handleFnChange(name: string) {
    setLossName(name)
    setBaseLr(DEFAULTS[name].baseLr)
  }

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
            {Object.keys(DEFAULTS).map((n) => (
              <option key={n} value={n}>
                {FN_LABELS[n] ?? n}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Base learning rate ={' '}
            <span className="font-mono">{baseLr.toFixed(3)}</span>
          </span>
          <input
            type="range"
            min={0.005}
            max={0.1}
            step={0.001}
            value={baseLr}
            onChange={(e) => setBaseLr(parseFloat(e.target.value))}
            className="w-40"
          />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* contours */}
          {contours.map((c, i) => (
            <g
              key={i}
              stroke="var(--ink)"
              strokeWidth={1}
              fill="none"
              opacity={0.22}
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

          {/* global minimum */}
          {loss.globalMin && (
            <circle
              cx={sx(loss.globalMin[0])}
              cy={sy(loss.globalMin[1])}
              r={5}
              fill="none"
              stroke="var(--ink-muted)"
              strokeWidth={1.2}
            />
          )}

          {/* trajectories */}
          {trajectories.map((traj, ti) => {
            const visible = traj
              .slice(0, frameIdx + 1)
              .filter((f) => !f.diverged)
            if (visible.length === 0) return null
            const path = visible
              .map(
                (f) =>
                  `${sx(f.position[0]).toFixed(2)},${sy(f.position[1]).toFixed(2)}`,
              )
              .join(' ')
            const last = visible[visible.length - 1]
            const isDiverged =
              traj[Math.min(frameIdx, traj.length - 1)]?.diverged ?? false
            return (
              <g key={ti}>
                <polyline
                  points={path}
                  fill="none"
                  stroke={COLORS[ti]}
                  strokeWidth={1.75}
                  opacity={0.85}
                />
                {!isDiverged && (
                  <circle
                    cx={sx(last.position[0])}
                    cy={sy(last.position[1])}
                    r={4}
                    fill={COLORS[ti]}
                    stroke="var(--paper)"
                    strokeWidth={1}
                  />
                )}
              </g>
            )
          })}

          {/* starting position */}
          <circle
            cx={sx(start[0])}
            cy={sy(start[1])}
            r={4.5}
            fill="var(--paper)"
            stroke="var(--ink)"
            strokeWidth={1.75}
          />
        </svg>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-x-5 gap-y-2 mt-3 font-mono text-xs">
        {LR_FACTORS.map((f, i) => {
          const lr = baseLr * f
          const traj = trajectories[i]
          const diverged = traj[traj.length - 1]?.diverged
          return (
            <span key={i} className="flex items-center gap-1.5">
              <span
                className="inline-block w-3 h-0.5"
                style={{ background: COLORS[i] }}
              />
              <span className="text-ink-muted">
                lr = {lr.toFixed(3)} ({LR_FACTOR_LABELS[i]})
                {diverged && (
                  <span className="ml-1 italic">— diverged</span>
                )}
              </span>
            </span>
          )
        })}
      </div>

      {/* Playback */}
      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button
          onClick={() => {
            if (frameIdx >= maxFrames - 1) setFrameIdx(0)
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
        <input
          type="range"
          min={0}
          max={Math.max(maxFrames - 1, 1)}
          value={frameIdx}
          onChange={(e) => {
            setPlaying(false)
            setFrameIdx(parseInt(e.target.value))
          }}
          className="flex-1"
        />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">
          step {frameIdx} / {maxFrames - 1}
        </span>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 6.3 — Four learning rates on the same problem, same starting
        position. The smallest crawls; the middle ones converge cleanly; the
        largest either oscillates or diverges entirely.
      </figcaption>
    </figure>
  )
}
