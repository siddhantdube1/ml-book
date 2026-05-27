'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  generateDataset,
  type DatasetShape,
  type Point,
} from '@/lib/datasets'
import { runKMeans, type InitMethod } from '@/lib/kmeans'

const CLUSTER_COLORS = [
  '#1d6d5e',
  '#c7522a',
  '#3c5a8c',
  '#a06614',
  '#6b4a8a',
  '#a83263',
]

export default function InitComparisonDemo() {
  const [shape, setShape] = useState<DatasetShape>('aniso')
  const [k, setK] = useState(3)
  const [seed, setSeed] = useState(91)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)

  const points: Point[] = useMemo(
    () => generateDataset(shape, 150, seed),
    [shape, seed],
  )

  const randomHistory = useMemo(
    () => runKMeans(points, k, 'random', seed + 13),
    [points, k, seed],
  )
  const kppHistory = useMemo(
    () => runKMeans(points, k, 'kpp', seed + 13),
    [points, k, seed],
  )

  const maxFrame = Math.max(randomHistory.length, kppHistory.length) - 1

  useEffect(() => {
    setFrame(0)
    setPlaying(false)
  }, [maxFrame])

  useEffect(() => {
    if (!playing) return
    if (frame >= maxFrame) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setFrame((f) => f + 1), 700)
    return () => clearTimeout(t)
  }, [playing, frame, maxFrame])

  const randomFrame =
    randomHistory[Math.min(frame, randomHistory.length - 1)]
  const kppFrame = kppHistory[Math.min(frame, kppHistory.length - 1)]

  const randomFinal = randomHistory[randomHistory.length - 1].wcss ?? 0
  const kppFinal = kppHistory[kppHistory.length - 1].wcss ?? 0
  const ratio = randomFinal > 0 ? kppFinal / randomFinal : 1
  const kppBetter = kppFinal < randomFinal - 0.5

  return (
    <figure className="not-prose my-10">
      <div className="rounded-lg border border-rule bg-paper p-4 sm:p-5">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
          <label className="flex flex-col gap-1.5">
            <span className="font-sans text-xs uppercase tracking-wider text-ink-muted">
              Dataset
            </span>
            <select
              value={shape}
              onChange={(e) => setShape(e.target.value as DatasetShape)}
              className="h-9 rounded-md px-2 border border-rule font-sans text-sm"
              style={{ background: 'var(--paper)', color: 'var(--ink)' }}
            >
              <option value="blobs">Three blobs</option>
              <option value="aniso">Anisotropic</option>
              <option value="moons">Two moons</option>
            </select>
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="font-sans text-xs uppercase tracking-wider text-ink-muted">
              K = {k}
            </span>
            <input
              type="range"
              min={2}
              max={6}
              step={1}
              value={k}
              onChange={(e) => setK(+e.target.value)}
              style={{ accentColor: 'var(--accent)' }}
            />
          </label>
          <div className="flex items-end justify-end">
            <button
              onClick={() => setSeed(Math.floor(Math.random() * 100000))}
              className="font-sans text-xs uppercase tracking-wider px-3 py-1.5 rounded-md border border-rule text-ink-muted hover:text-ink hover:border-ink-faint transition-colors"
            >
              New seed
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Panel
            title="Random initialisation"
            method="random"
            points={points}
            frame={randomFrame}
            finalWcss={randomFinal}
            converged={frame >= randomHistory.length - 1}
          />
          <Panel
            title="k-means++"
            method="kpp"
            points={points}
            frame={kppFrame}
            finalWcss={kppFinal}
            converged={frame >= kppHistory.length - 1}
          />
        </div>

        <div
          className="mt-4 rounded-md px-4 py-3 text-sm"
          style={{
            background: kppBetter
              ? 'color-mix(in srgb, var(--accent) 8%, var(--paper))'
              : 'color-mix(in srgb, var(--ink) 4%, var(--paper))',
            color: 'var(--ink)',
          }}
        >
          <span className="font-sans" style={{ fontWeight: 500 }}>
            Final WCSS:
          </span>{' '}
          <span className="font-sans tabular-nums">
            random {Math.round(randomFinal).toLocaleString()} · k-means++{' '}
            {Math.round(kppFinal).toLocaleString()}
          </span>
          {kppBetter && (
            <span
              className="font-sans ml-2"
              style={{ color: 'var(--accent)' }}
            >
              k-means++ found a better minimum ({Math.round((1 - ratio) * 100)}% lower).
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 mt-3">
          <button
            onClick={() => {
              setPlaying(false)
              setFrame(0)
            }}
            className="font-sans px-3 py-1.5 rounded-md border border-rule text-sm"
            style={{ color: 'var(--ink)' }}
          >
            Reset
          </button>
          <button
            onClick={() => {
              if (frame >= maxFrame) setFrame(0)
              setPlaying((p) => !p)
            }}
            className="font-sans px-4 py-1.5 rounded-md text-sm"
            style={{ background: 'var(--accent)', color: 'var(--paper)' }}
          >
            {playing ? 'Pause' : 'Play both'}
          </button>
          <input
            type="range"
            min={0}
            max={maxFrame}
            value={frame}
            onChange={(e) => {
              setPlaying(false)
              setFrame(+e.target.value)
            }}
            className="flex-1"
            style={{ accentColor: 'var(--accent)' }}
          />
          <span
            className="font-sans tabular-nums text-sm text-ink-muted"
            style={{ minWidth: 60, textAlign: 'right' }}
          >
            {frame} / {maxFrame}
          </span>
        </div>
      </div>
      <figcaption className="font-sans text-sm text-ink-muted mt-3 text-center">
        Figure 18.5 — Same data, same K, two different initialisations. Press
        new seed a few times — random sometimes finds the right answer, sometimes
        does not. k-means++ is far more reliable.
      </figcaption>
    </figure>
  )
}

function Panel({
  title,
  method,
  points,
  frame,
  finalWcss,
  converged,
}: {
  title: string
  method: InitMethod
  points: Point[]
  frame: { centroids: Point[]; assignments: number[]; wcss: number | null }
  finalWcss: number
  converged: boolean
}) {
  return (
    <div
      className="rounded-md p-3"
      style={{ background: 'color-mix(in srgb, var(--ink) 3%, var(--paper))' }}
    >
      <div className="flex items-baseline justify-between mb-2">
        <span
          className="font-sans"
          style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink)' }}
        >
          {title}
        </span>
        <span
          className="font-sans tabular-nums"
          style={{ fontSize: 11, color: 'var(--ink-muted)' }}
        >
          {converged
            ? `final WCSS ${Math.round(finalWcss).toLocaleString()}`
            : frame.wcss == null
              ? '—'
              : `WCSS ${Math.round(frame.wcss).toLocaleString()}`}
        </span>
      </div>
      <svg
        viewBox="0 0 680 360"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full block rounded-md"
        style={{
          background: 'var(--paper)',
          aspectRatio: '680 / 360',
        }}
      >
        {points.map((p, i) => {
          const a = frame.assignments[i]
          return (
            <circle
              key={i}
              cx={p[0]}
              cy={p[1]}
              r={4}
              fill={a >= 0 ? CLUSTER_COLORS[a] : 'var(--ink-faint)'}
              opacity={a >= 0 ? 0.85 : 0.5}
              style={{ transition: 'fill 0.25s ease, opacity 0.25s ease' }}
            />
          )
        })}
        {frame.centroids.map((c, k) => (
          <g
            key={k}
            transform={`translate(${c[0].toFixed(2)}, ${c[1].toFixed(2)})`}
            style={{ transition: 'transform 0.4s cubic-bezier(0.4, 0, 0.2, 1)' }}
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
    </div>
  )
}
