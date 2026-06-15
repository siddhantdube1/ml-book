'use client'

import { useState } from 'react'

const STAGES = [
  {
    name: 'The residual stream',
    text: 'A token enters as a vector and travels straight up the residual stream — the block’s backbone. Each sublayer reads from the stream and adds its result back, so information is never overwritten, only refined.',
  },
  {
    name: 'Attention sublayer',
    text: 'First sublayer: normalise, then self-attention — the step where tokens look at each other (Chapter 4). Its output is added back to the stream. This is the only place tokens communicate.',
  },
  {
    name: 'MLP sublayer',
    text: 'Second sublayer: normalise, then a small per-token feed-forward network. It processes each token on its own — no looking around — and adds its result back. This is where much of the model’s stored knowledge lives.',
  },
]

const W = 520
const H = 380
const SX = 120 // residual stream x

type Props = { caption?: string }

export default function TransformerBlock({ caption }: Props) {
  const [stage, setStage] = useState(0)

  const attnActive = stage === 0 || stage === 1
  const mlpActive = stage === 0 || stage === 2

  const box = (
    x: number,
    y: number,
    w: number,
    h: number,
    lines: string[],
    active: boolean,
  ) => (
    <g opacity={active ? 1 : 0.28}>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={6}
        fill="var(--paper)"
        stroke={active ? 'var(--accent)' : 'var(--rule)'}
        strokeWidth={1.5}
      />
      {lines.map((l, i) => (
        <text
          key={i}
          x={x + w / 2}
          y={y + 20 + i * 16}
          textAnchor="middle"
          className="font-sans"
          fontSize={12}
          fill="var(--ink)"
        >
          {l}
        </text>
      ))}
    </g>
  )

  const plus = (y: number, active: boolean) => (
    <g opacity={active ? 1 : 0.28}>
      <circle
        cx={SX}
        cy={y}
        r={11}
        fill="var(--paper)"
        stroke={active ? 'var(--accent)' : 'var(--rule)'}
        strokeWidth={1.5}
      />
      <text x={SX} y={y + 4} textAnchor="middle" fontSize={14} fill="var(--ink)">
        +
      </text>
    </g>
  )

  return (
    <figure className="my-10 not-prose">
      <div className="flex flex-wrap gap-2 mb-4 font-sans text-sm">
        {STAGES.map((s, i) => (
          <button
            key={i}
            onClick={() => setStage(i)}
            className={`px-3 py-1 border rounded transition-colors ${
              stage === i
                ? 'border-accent text-accent'
                : 'border-rule text-ink-muted hover:border-accent'
            }`}
          >
            {s.name}
          </button>
        ))}
      </div>

      <div className="border border-rule rounded-lg bg-paper-soft p-2">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* Residual stream spine */}
          <line x1={SX} y1={20} x2={SX} y2={H - 20} stroke="var(--ink-muted)" strokeWidth={2} />
          <text x={SX - 16} y={H - 24} textAnchor="end" className="font-sans" fontSize={11} fill="var(--ink-muted)">
            in
          </text>
          <text x={SX - 16} y={32} textAnchor="end" className="font-sans" fontSize={11} fill="var(--ink-muted)">
            out
          </text>

          {/* MLP sublayer (upper) */}
          <path
            d={`M ${SX} 110 H 300`}
            fill="none"
            stroke={mlpActive ? 'var(--accent)' : 'var(--rule)'}
            strokeWidth={1.5}
          />
          <path
            d={`M 300 175 V 130 H ${SX}`}
            fill="none"
            stroke={mlpActive ? 'var(--accent)' : 'var(--rule)'}
            strokeWidth={1.5}
            markerEnd=""
          />
          {box(300, 110, 180, 56, ['LayerNorm →', 'MLP (feed-forward)'], mlpActive)}
          {plus(130, mlpActive)}

          {/* Attention sublayer (lower) */}
          <path
            d={`M ${SX} 250 H 300`}
            fill="none"
            stroke={attnActive ? 'var(--accent)' : 'var(--rule)'}
            strokeWidth={1.5}
          />
          <path
            d={`M 300 285 V 270 H ${SX}`}
            fill="none"
            stroke={attnActive ? 'var(--accent)' : 'var(--rule)'}
            strokeWidth={1.5}
          />
          {box(300, 250, 180, 56, ['LayerNorm →', 'Self-Attention'], attnActive)}
          {plus(270, attnActive)}
        </svg>
      </div>

      <div className="mt-3 rounded-lg border border-rule bg-paper-soft px-4 py-3 font-sans text-sm text-ink-muted min-h-[4.5rem]">
        <span className="font-medium text-ink">{STAGES[stage].name}. </span>
        {STAGES[stage].text}
      </div>

      {caption && (
        <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}
