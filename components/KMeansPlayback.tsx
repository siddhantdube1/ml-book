'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  generateDataset,
  type DatasetShape,
  type Point,
} from '@/lib/datasets'
import { runKMeans, type InitMethod, type KMeansFrame } from '@/lib/kmeans'

const CLUSTER_COLORS = [
  '#1d6d5e',
  '#c7522a',
  '#3c5a8c',
  '#a06614',
  '#6b4a8a',
  '#a83263',
  '#5d7a25',
  '#a3322b',
]

export default function KMeansPlayback() {
  const [shape, setShape] = useState<DatasetShape>('blobs')
  const [k, setK] = useState(3)
  const [init, setInit] = useState<InitMethod>('random')
  const [seed, setSeed] = useState(42)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(700)

  const points: Point[] = useMemo(
    () => generateDataset(shape, 150, seed),
    [shape, seed],
  )

  const history: KMeansFrame[] = useMemo(
    () => runKMeans(points, k, init, seed + 7),
    [points, k, init, seed],
  )

  useEffect(() => {
    setFrame(0)
  }, [history])

  useEffect(() => {
    if (!playing) return
    if (frame >= history.length - 1) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setFrame((f) => f + 1), speed)
    return () => clearTimeout(t)
  }, [playing, frame, history.length, speed])

  const current = history[frame] ?? history[0]
  const isLast = frame >= history.length - 1

  return (
    <figure className="not-prose my-10 -mx-2 sm:mx-0">
      <div className="rounded-lg border border-rule bg-paper p-4 sm:p-5">
        <Controls
          shape={shape}
          onShape={setShape}
          k={k}
          onK={setK}
          init={init}
          onInit={setInit}
        />

        <Canvas points={points} frame={current} />

        <Stats frame={current} index={frame} max={history.length - 1} />

        <Playback
          playing={playing}
          isLast={isLast}
          onPlay={() => {
            if (isLast) setFrame(0)
            setPlaying((p) => !p)
          }}
          onReset={() => {
            setPlaying(false)
            setFrame(0)
          }}
          onBack={() => {
            setPlaying(false)
            setFrame((f) => Math.max(0, f - 1))
          }}
          onForward={() => {
            setPlaying(false)
            setFrame((f) => Math.min(history.length - 1, f + 1))
          }}
          frame={frame}
          maxFrame={history.length - 1}
          onScrub={(v) => {
            setPlaying(false)
            setFrame(v)
          }}
          speed={speed}
          onSpeed={setSpeed}
          onReseed={() => setSeed(Math.floor(Math.random() * 100000))}
        />
      </div>
      <figcaption className="font-sans text-sm text-ink-muted mt-3 text-center">
        Figure 18.1 — k-means convergence on synthetic data. Modify any control
        and the algorithm re-runs from scratch.
      </figcaption>
    </figure>
  )
}

function Controls({
  shape,
  onShape,
  k,
  onK,
  init,
  onInit,
}: {
  shape: DatasetShape
  onShape: (s: DatasetShape) => void
  k: number
  onK: (k: number) => void
  init: InitMethod
  onInit: (i: InitMethod) => void
}) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
      <Field label="Dataset">
        <select
          value={shape}
          onChange={(e) => onShape(e.target.value as DatasetShape)}
          className="select"
        >
          <option value="blobs">Three blobs</option>
          <option value="aniso">Anisotropic</option>
          <option value="moons">Two moons</option>
        </select>
      </Field>
      <Field label={`K = ${k}`}>
        <input
          type="range"
          min={2}
          max={8}
          step={1}
          value={k}
          onChange={(e) => onK(+e.target.value)}
          className="range"
        />
      </Field>
      <Field label="Initialisation">
        <select
          value={init}
          onChange={(e) => onInit(e.target.value as InitMethod)}
          className="select"
        >
          <option value="random">Random</option>
          <option value="kpp">k-means++</option>
        </select>
      </Field>

      <style jsx>{`
        .select {
          width: 100%;
          height: 36px;
          background: var(--paper);
          color: var(--ink);
          border: 1px solid var(--rule);
          border-radius: 6px;
          padding: 0 10px;
          font-family: var(--font-sans);
          font-size: 14px;
        }
        .range {
          width: 100%;
          accent-color: var(--accent);
        }
      `}</style>
    </div>
  )
}

function Field({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="font-sans text-xs uppercase tracking-wider text-ink-muted">
        {label}
      </span>
      {children}
    </label>
  )
}

function Canvas({ points, frame }: { points: Point[]; frame: KMeansFrame }) {
  return (
    <svg
      viewBox="0 0 680 360"
      xmlns="http://www.w3.org/2000/svg"
      className="w-full block rounded-md"
      style={{ background: 'color-mix(in srgb, var(--ink) 3%, var(--paper))', height: 'auto' }}
    >
      <g>
        {points.map((p, i) => {
          const a = frame.assignments[i]
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
      </g>
      <g>
        {frame.centroids.map((c, k) => (
          <g
            key={k}
            transform={`translate(${c[0].toFixed(2)}, ${c[1].toFixed(2)})`}
            style={{
              transition:
                'transform 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
            }}
          >
            <circle
              r={12}
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
      </g>
    </svg>
  )
}

function Stats({
  frame,
  index,
  max,
}: {
  frame: KMeansFrame
  index: number
  max: number
}) {
  return (
    <div className="grid grid-cols-3 gap-3 my-3">
      <Stat label="Iteration" value={`${index} / ${max}`} />
      <Stat
        label="WCSS (inertia)"
        value={frame.wcss == null ? '—' : Math.round(frame.wcss).toLocaleString()}
      />
      <Stat label="Status" value={frame.label} small />
    </div>
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

function Playback({
  playing,
  isLast,
  onPlay,
  onReset,
  onBack,
  onForward,
  frame,
  maxFrame,
  onScrub,
  speed,
  onSpeed,
  onReseed,
}: {
  playing: boolean
  isLast: boolean
  onPlay: () => void
  onReset: () => void
  onBack: () => void
  onForward: () => void
  frame: number
  maxFrame: number
  onScrub: (v: number) => void
  speed: number
  onSpeed: (v: number) => void
  onReseed: () => void
}) {
  return (
    <div className="mt-2">
      <div className="flex items-center gap-2">
        <IconButton label="Reset" onClick={onReset}>
          ↺
        </IconButton>
        <IconButton label="Step back" onClick={onBack}>
          ‹
        </IconButton>
        <IconButton
          label={playing ? 'Pause' : isLast ? 'Replay' : 'Play'}
          onClick={onPlay}
          primary
        >
          {playing ? '❚❚' : '▶'}
        </IconButton>
        <IconButton label="Step forward" onClick={onForward}>
          ›
        </IconButton>
        <input
          type="range"
          min={0}
          max={maxFrame}
          value={frame}
          onChange={(e) => onScrub(+e.target.value)}
          className="flex-1"
          style={{ accentColor: 'var(--accent)' }}
        />
      </div>
      <div className="flex items-center gap-3 mt-3">
        <span className="font-sans text-xs uppercase tracking-wider text-ink-muted">
          Speed
        </span>
        <input
          type="range"
          min={200}
          max={1400}
          step={100}
          value={speed}
          onChange={(e) => onSpeed(+e.target.value)}
          className="flex-1 max-w-[220px]"
          style={{ accentColor: 'var(--accent)' }}
        />
        <button
          onClick={onReseed}
          className="ml-auto font-sans text-xs uppercase tracking-wider px-3 py-1.5 rounded-md border border-rule text-ink-muted hover:text-ink hover:border-ink-faint transition-colors"
        >
          New points
        </button>
      </div>
    </div>
  )
}

function IconButton({
  children,
  onClick,
  label,
  primary,
}: {
  children: React.ReactNode
  onClick: () => void
  label: string
  primary?: boolean
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      className="flex items-center justify-center transition-colors"
      style={{
        width: 38,
        height: 38,
        borderRadius: 6,
        border: '1px solid var(--rule)',
        background: primary ? 'var(--accent)' : 'transparent',
        color: primary ? 'var(--paper)' : 'var(--ink)',
        fontFamily: 'var(--font-sans)',
        fontSize: 14,
      }}
    >
      {children}
    </button>
  )
}
