'use client'

import { useEffect, useState } from 'react'
import type { GenerationFixture } from '@trilogy/book-kit/model'

type Props = {
  fixtureIds: string[]
  caption?: string
}

/**
 * Token-by-token autoregressive generation: scrub the steps, watch the text
 * grow one greedy token at a time, with the top alternatives at each step.
 */
export default function GenerationPlayback({ fixtureIds, caption }: Props) {
  const [which, setWhich] = useState(0)
  const [fix, setFix] = useState<GenerationFixture | null>(null)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const fixtureId = fixtureIds[which]

  useEffect(() => {
    let alive = true
    setFix(null)
    setStep(0)
    setPlaying(false)
    fetch(`/fixtures/generation/${fixtureId}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d: GenerationFixture) => alive && setFix(d))
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [fixtureId])

  useEffect(() => {
    if (!playing || !fix) return
    if (step >= fix.steps.length) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setStep((s) => s + 1), 400)
    return () => clearTimeout(t)
  }, [playing, step, fix])

  if (!fix) {
    return (
      <figure className="my-10 not-prose">
        <div className="border border-rule rounded-lg bg-paper-soft p-6 text-sm text-ink-faint font-sans">
          Loading…
        </div>
      </figure>
    )
  }

  const generated = fix.steps.slice(0, step).map((s) => s.chosen).join('')
  const current = step < fix.steps.length ? fix.steps[step] : null

  return (
    <figure className="my-10 not-prose">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-4 font-sans text-sm">
        <span className="text-ink-muted">Prompt</span>
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
      </div>

      <div className="border border-rule rounded-lg bg-paper-soft px-5 py-4">
        <div className="font-serif text-lg leading-relaxed min-h-[3.5rem]">
          <span className="text-ink-muted">{fix.prompt}</span>
          <span className="text-ink">{generated}</span>
          {current && <span className="text-accent">▮</span>}
        </div>

        {current && (
          <div className="mt-4 pt-3 border-t border-rule">
            <div className="font-sans text-xs uppercase tracking-wider text-ink-faint mb-2">
              Next-token candidates — greedy takes the top one
            </div>
            <div className="space-y-1">
              {current.top.map((t, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span
                    className={`font-mono text-xs w-24 shrink-0 text-right ${
                      i === 0 ? 'text-accent' : 'text-ink'
                    }`}
                  >
                    {t.text.trim() || '·'}
                  </span>
                  <div className="flex-1 h-3.5">
                    <div
                      className="h-3.5 rounded-sm"
                      style={{
                        width: `${(t.prob / current.top[0].prob) * 100}%`,
                        background: i === 0 ? 'var(--accent)' : 'var(--rule)',
                        minWidth: 2,
                      }}
                    />
                  </div>
                  <span className="font-mono text-xs w-12 shrink-0 text-ink-muted">
                    {(t.prob * 100).toFixed(1)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button
          onClick={() => {
            if (step >= fix.steps.length) setStep(0)
            setPlaying((p) => !p)
          }}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[64px]"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={() => {
            setStep(0)
            setPlaying(false)
          }}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          Reset
        </button>
        <button
          onClick={() => setStep((s) => Math.min(s + 1, fix.steps.length))}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          Step
        </button>
        <input
          type="range"
          min={0}
          max={fix.steps.length}
          value={step}
          onChange={(e) => {
            setPlaying(false)
            setStep(parseInt(e.target.value))
          }}
          className="flex-1"
        />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">
          {step} / {fix.steps.length}
        </span>
      </div>

      {caption && (
        <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}
