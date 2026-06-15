'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  decodeDistribution,
  type NextTokenFixture,
} from '@trilogy/book-kit/model'

function tokenLabel(text: string): string {
  if (text === '') return '·'
  if (/^\s+$/.test(text)) return '␣'
  return text.replace(/^ /, '␣')
}

const SHOWN = 14 // bars to display

type Props = {
  /** Fixture ids under /public/fixtures/nexttoken; the first is the default. */
  fixtureIds: string[]
  caption?: string
}

export default function SamplingExplorer({ fixtureIds, caption }: Props) {
  const [which, setWhich] = useState(0)
  const [fix, setFix] = useState<NextTokenFixture | null>(null)
  const [temperature, setTemperature] = useState(1)
  const [topK, setTopK] = useState(0)
  const [topP, setTopP] = useState(1)
  const [samples, setSamples] = useState<string[]>([])

  const fixtureId = fixtureIds[which]

  useEffect(() => {
    let alive = true
    setFix(null)
    setSamples([])
    fetch(`/fixtures/nexttoken/${fixtureId}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d: NextTokenFixture) => alive && setFix(d))
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [fixtureId])

  const probs = useMemo(() => {
    if (!fix) return []
    return decodeDistribution(
      fix.topk.map((t) => t.logit),
      { temperature, topK, topP },
    )
  }, [fix, temperature, topK, topP])

  if (!fix) {
    return (
      <figure className="my-10 not-prose">
        <div className="border border-rule rounded-lg bg-paper-soft p-6 text-sm text-ink-faint font-sans">
          Loading distribution…
        </div>
      </figure>
    )
  }

  const rows = fix.topk
    .map((t, i) => ({ ...t, p: probs[i] }))
    .slice(0, SHOWN)
  const maxP = Math.max(...rows.map((r) => r.p), 0.001)

  function sample() {
    const r = Math.random()
    let cum = 0
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i]
      if (r <= cum) {
        setSamples((s) => [fix!.topk[i].text, ...s].slice(0, 12))
        return
      }
    }
  }

  return (
    <figure className="my-10 not-prose">
      {/* Context selector + controls */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Context</span>
          <select
            value={which}
            onChange={(e) => setWhich(parseInt(e.target.value))}
            className="bg-paper border border-rule rounded px-2 py-1 font-mono text-xs text-ink"
          >
            {fixtureIds.map((id, i) => (
              <option key={id} value={i}>
                {id}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="border border-rule rounded-lg bg-paper-soft px-5 py-4">
        <div className="font-serif text-lg mb-1">
          {fix.context}
          <span className="text-ink-faint"> ▮</span>
        </div>
        <div className="font-sans text-xs text-ink-faint mb-4">
          GPT-2&rsquo;s next-token distribution — reshaped live by your settings.
        </div>

        {/* Bars */}
        <div className="space-y-1">
          {rows.map((r, i) => {
            const excluded = r.p === 0
            return (
              <div key={i} className="flex items-center gap-2">
                <span
                  className={`font-mono text-xs w-20 shrink-0 text-right ${
                    excluded ? 'text-ink-faint line-through' : 'text-ink'
                  }`}
                >
                  {tokenLabel(r.text)}
                </span>
                <div className="flex-1 h-4 relative">
                  <div
                    className="h-4 rounded-sm"
                    style={{
                      width: `${(r.p / maxP) * 100}%`,
                      background: excluded
                        ? 'var(--rule)'
                        : 'var(--accent)',
                      minWidth: r.p > 0 ? 2 : 0,
                      transition: 'width 0.18s ease',
                    }}
                  />
                </div>
                <span
                  className={`font-mono text-xs w-12 shrink-0 ${
                    excluded ? 'text-ink-faint' : 'text-ink-muted'
                  }`}
                >
                  {(r.p * 100).toFixed(1)}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Sliders */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 mt-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Temperature <span className="font-mono">{temperature.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.05}
            max={2}
            step={0.05}
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            className="w-36"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            top-k{' '}
            <span className="font-mono">{topK === 0 ? 'off' : topK}</span>
          </span>
          <input
            type="range"
            min={0}
            max={40}
            step={1}
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            className="w-32"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            top-p <span className="font-mono">{topP.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.1}
            max={1}
            step={0.05}
            value={topP}
            onChange={(e) => setTopP(parseFloat(e.target.value))}
            className="w-32"
          />
        </label>
        <button
          onClick={sample}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          Sample
        </button>
      </div>

      {samples.length > 0 && (
        <div className="mt-3 font-mono text-sm text-ink-muted">
          drawn:{' '}
          {samples.map((s, i) => (
            <span
              key={i}
              className="text-ink"
              style={{ opacity: 1 - i * 0.07 }}
            >
              {s.trim() || tokenLabel(s)}
              {i < samples.length - 1 ? ' · ' : ''}
            </span>
          ))}
        </div>
      )}

      {caption && (
        <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}
