'use client'

import { useEffect, useMemo, useState } from 'react'
import { generateDataset, type Point } from '@/lib/datasets'
import { runKMeans } from '@/lib/kmeans'

const CLUSTER_COLORS = [
  '#1d6d5e',
  '#c7522a',
  '#3c5a8c',
  '#a06614',
  '#6b4a8a',
]

export default function ConvergencePlot() {
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [seed, setSeed] = useState(11)

  const points: Point[] = useMemo(() => generateDataset('blobs', 150, seed), [seed])
  const history = useMemo(() => runKMeans(points, 3, 'random', seed + 7), [points, seed])

  const wcssSeries = useMemo(() => {
    const series: { it: number; wcss: number }[] = []
    let it = 0
    for (let i = 1; i < history.length; i += 2) {
      const w = history[i].wcss
      if (w != null) {
        it++
        series.push({ it, wcss: w })
      }
    }
    return series
  }, [history])

  useEffect(() => {
    setFrame(0)
  }, [history])

  useEffect(() => {
    if (!playing) return
    if (frame >= history.length - 1) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setFrame((f) => f + 1), 700)
    return () => clearTimeout(t)
  }, [playing, frame, history.length])

  const current = history[frame]
  const currentIterIndex = Math.max(0, Math.floor(frame / 2))

  const maxWcss = wcssSeries.length ? wcssSeries[0].wcss : 1
  const minWcss = wcssSeries.length ? wcssSeries[wcssSeries.length - 1].wcss : 0
  const wcssRange = Math.max(1, maxWcss - minWcss)
  const yPad = wcssRange * 0.15

  const chartW = 320
  const chartH = 220
  const padL = 50
  const padR = 16
  const padT = 16
  const padB = 32
  const innerW = chartW - padL - padR
  const innerH = chartH - padT - padB

  const xOf = (i: number) =>
    padL +
    (wcssSeries.length <= 1
      ? 0
      : (i / (wcssSeries.length - 1)) * innerW)
  const yOf = (w: number) =>
    padT + innerH - ((w - (minWcss - yPad)) / (wcssRange + 2 * yPad)) * innerH

  const linePath = wcssSeries
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xOf(i).toFixed(1)} ${yOf(p.wcss).toFixed(1)}`)
    .join(' ')

  return (
    <figure className="not-prose my-10">
      <div className="rounded-lg border border-rule bg-paper p-4 sm:p-5">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-stretch">
          <svg
            viewBox="0 0 680 360"
            xmlns="http://www.w3.org/2000/svg"
            className="w-full block rounded-md"
            style={{
              background: 'color-mix(in srgb, var(--ink) 3%, var(--paper))',
              aspectRatio: '680 / 360',
            }}
          >
            {points.map((p, i) => {
              const a = current.assignments[i]
              const color = a >= 0 ? CLUSTER_COLORS[a] : 'var(--ink-faint)'
              const op = a >= 0 ? 0.85 : 0.5
              return (
                <circle
                  key={i}
                  cx={p[0]}
                  cy={p[1]}
                  r={4.5}
                  fill={color}
                  opacity={op}
                  style={{ transition: 'fill 0.25s ease, opacity 0.25s ease' }}
                />
              )
            })}
            {current.centroids.map((c, k) => (
              <g
                key={k}
                transform={`translate(${c[0].toFixed(2)}, ${c[1].toFixed(2)})`}
                style={{
                  transition:
                    'transform 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
                }}
              >
                <circle
                  r={11}
                  fill={CLUSTER_COLORS[k]}
                  stroke="var(--paper)"
                  strokeWidth={3}
                />
                <path
                  d="M-5,0 L5,0 M0,-5 L0,5"
                  stroke="var(--paper)"
                  strokeWidth={2.5}
                  strokeLinecap="round"
                />
              </g>
            ))}
          </svg>

          <svg
            viewBox={`0 0 ${chartW} ${chartH}`}
            xmlns="http://www.w3.org/2000/svg"
            className="w-full block rounded-md"
            style={{
              background: 'color-mix(in srgb, var(--ink) 3%, var(--paper))',
              aspectRatio: `${chartW} / ${chartH}`,
            }}
          >
            <line
              x1={padL}
              y1={padT + innerH}
              x2={padL + innerW}
              y2={padT + innerH}
              stroke="var(--ink-faint)"
              strokeWidth={0.5}
            />
            <line
              x1={padL}
              y1={padT}
              x2={padL}
              y2={padT + innerH}
              stroke="var(--ink-faint)"
              strokeWidth={0.5}
            />
            <text
              x={padL}
              y={padT - 4}
              fontFamily="var(--font-sans)"
              fontSize={10}
              fill="var(--ink-muted)"
            >
              WCSS
            </text>
            <text
              x={padL + innerW}
              y={padT + innerH + 18}
              fontFamily="var(--font-sans)"
              fontSize={10}
              fill="var(--ink-muted)"
              textAnchor="end"
            >
              iteration
            </text>

            <path
              d={linePath}
              fill="none"
              stroke="var(--accent)"
              strokeWidth={1.5}
              opacity={0.6}
            />

            {wcssSeries.map((p, i) => {
              const active = i + 1 === currentIterIndex
              const passed = i + 1 <= currentIterIndex
              return (
                <circle
                  key={i}
                  cx={xOf(i)}
                  cy={yOf(p.wcss)}
                  r={active ? 5 : 3}
                  fill={passed ? 'var(--accent)' : 'var(--ink-faint)'}
                  opacity={passed ? 1 : 0.5}
                  style={{ transition: 'r 0.2s, opacity 0.2s' }}
                />
              )
            })}

            {currentIterIndex >= 1 &&
              currentIterIndex <= wcssSeries.length && (
                <text
                  x={xOf(currentIterIndex - 1)}
                  y={yOf(wcssSeries[currentIterIndex - 1].wcss) - 12}
                  fontFamily="var(--font-sans)"
                  fontSize={11}
                  fill="var(--accent)"
                  textAnchor="middle"
                  style={{ fontWeight: 500 }}
                >
                  {Math.round(wcssSeries[currentIterIndex - 1].wcss).toLocaleString()}
                </text>
              )}
          </svg>
        </div>

        <div className="grid grid-cols-3 gap-3 my-3">
          <Stat
            label="Iteration"
            value={`${Math.max(0, currentIterIndex)} / ${wcssSeries.length}`}
          />
          <Stat
            label="WCSS now"
            value={
              current.wcss == null
                ? '—'
                : Math.round(current.wcss).toLocaleString()
            }
          />
          <Stat
            label="Status"
            value={current.label}
            small
          />
        </div>

        <div className="flex items-center gap-2 mt-2">
          <button
            onClick={() => {
              setPlaying(false)
              setFrame(0)
            }}
            className="font-sans px-3 py-1.5 rounded-md border border-rule text-sm hover:border-ink-faint transition-colors"
            style={{ color: 'var(--ink)' }}
          >
            Reset
          </button>
          <button
            onClick={() => {
              setPlaying(false)
              setFrame((f) => Math.max(0, f - 1))
            }}
            className="font-sans px-3 py-1.5 rounded-md border border-rule text-sm hover:border-ink-faint transition-colors"
            style={{ color: 'var(--ink)' }}
          >
            ‹ Step
          </button>
          <button
            onClick={() => {
              if (frame >= history.length - 1) setFrame(0)
              setPlaying((p) => !p)
            }}
            className="font-sans px-4 py-1.5 rounded-md text-sm"
            style={{ background: 'var(--accent)', color: 'var(--paper)' }}
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={() => {
              setPlaying(false)
              setFrame((f) => Math.min(history.length - 1, f + 1))
            }}
            className="font-sans px-3 py-1.5 rounded-md border border-rule text-sm hover:border-ink-faint transition-colors"
            style={{ color: 'var(--ink)' }}
          >
            Step ›
          </button>
          <input
            type="range"
            min={0}
            max={history.length - 1}
            value={frame}
            onChange={(e) => {
              setPlaying(false)
              setFrame(+e.target.value)
            }}
            className="flex-1"
            style={{ accentColor: 'var(--accent)' }}
          />
          <button
            onClick={() => setSeed(Math.floor(Math.random() * 100000))}
            className="font-sans text-xs uppercase tracking-wider px-3 py-1.5 rounded-md border border-rule text-ink-muted hover:text-ink hover:border-ink-faint transition-colors"
          >
            New data
          </button>
        </div>
      </div>
      <figcaption className="font-sans text-sm text-ink-muted mt-3 text-center">
        Figure 18.3 — Clustering state alongside its WCSS. Each iteration the
        WCSS goes down, never up.
      </figcaption>
    </figure>
  )
}

function Stat({
  label,
  value,
  small,
}: {
  label: string
  value: string
  small?: boolean
}) {
  return (
    <div
      className="rounded-md px-3 py-2"
      style={{ background: 'color-mix(in srgb, var(--ink) 4%, var(--paper))' }}
    >
      <div className="font-sans text-xs uppercase tracking-wider text-ink-muted">
        {label}
      </div>
      <div
        className={`font-sans tabular-nums ${small ? 'text-sm' : 'text-lg'} text-ink`}
        style={{ fontWeight: 500 }}
      >
        {value}
      </div>
    </div>
  )
}
