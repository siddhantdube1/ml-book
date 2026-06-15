'use client'

import { useMemo, useState } from 'react'

// Illustrative reward-model training: you compare two responses, and a reward
// model updates toward your preference via the Bradley-Terry gradient. The
// content is author-written (not a live model) — it teaches the mechanism.
type Example = {
  prompt: string
  a: string
  b: string
}

const EXAMPLES: Example[] = [
  {
    prompt: 'How can I make my essay more persuasive?',
    a: 'Open with a clear thesis, back each point with concrete evidence, and address the strongest counterargument head-on.',
    b: "Just write whatever comes to mind — persuasion is mostly luck anyway.",
  },
  {
    prompt: 'My houseplant’s leaves are turning yellow. What should I do?',
    a: 'Yellow leaves often mean overwatering — check that the soil drains and let it dry between waterings before adding light.',
    b: 'Plants die sometimes. You should probably just buy a new one.',
  },
]

const LR = 0.6

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x))
}

type Props = { caption?: string }

export default function PreferenceTuning({ caption }: Props) {
  const [ex, setEx] = useState(0)
  // reward[example][response]
  const [reward, setReward] = useState<number[][]>(
    EXAMPLES.map(() => [0, 0]),
  )
  const [count, setCount] = useState(0)

  const rA = reward[ex][0]
  const rB = reward[ex][1]
  const pA = useMemo(() => sigmoid(rA - rB), [rA, rB])

  function prefer(choice: 0 | 1) {
    setReward((prev) => {
      const next = prev.map((r) => [...r])
      const [ra, rb] = next[ex]
      const pPref = sigmoid((choice === 0 ? ra - rb : rb - ra))
      const g = LR * (1 - pPref) // Bradley-Terry gradient step
      next[ex][choice] += g
      next[ex][1 - choice] -= g
      return next
    })
    setCount((c) => c + 1)
  }

  const e = EXAMPLES[ex]
  const bar = (v: number) => `${Math.round(sigmoid(v) * 100)}%`

  return (
    <figure className="my-10 not-prose">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-4 font-sans text-sm">
        <span className="text-ink-muted">Prompt</span>
        <select
          value={ex}
          onChange={(e) => setEx(parseInt(e.target.value))}
          className="bg-paper border border-rule rounded px-2 py-1 text-xs text-ink max-w-xs"
        >
          {EXAMPLES.map((x, i) => (
            <option key={i} value={i}>
              {x.prompt}
            </option>
          ))}
        </select>
      </div>

      <div className="border border-rule rounded-lg bg-paper-soft px-5 py-4">
        <div className="font-serif text-lg mb-3">{e.prompt}</div>
        <div className="grid gap-3 sm:grid-cols-2">
          {([0, 1] as const).map((c) => (
            <button
              key={c}
              onClick={() => prefer(c)}
              className="text-left rounded-lg border border-rule bg-paper px-4 py-3 hover:border-accent transition-colors"
            >
              <div className="font-sans text-xs uppercase tracking-wider text-ink-faint mb-1">
                Response {c === 0 ? 'A' : 'B'} — click if better
              </div>
              <div className="font-serif text-ink leading-snug">
                {c === 0 ? e.a : e.b}
              </div>
              <div className="mt-3 flex items-center gap-2">
                <div className="flex-1 h-2 rounded-full bg-rule overflow-hidden">
                  <div
                    className="h-2"
                    style={{
                      width: bar(reward[ex][c]),
                      background: 'var(--accent)',
                      transition: 'width 0.25s ease',
                    }}
                  />
                </div>
                <span className="font-mono text-xs text-ink-muted w-10 text-right">
                  {reward[ex][c].toFixed(2)}
                </span>
              </div>
            </button>
          ))}
        </div>

        <div className="mt-4 font-sans text-sm text-ink-muted">
          Reward model predicts you prefer{' '}
          <span className="font-mono text-ink">A</span> with probability{' '}
          <span className="font-mono text-accent">{(pA * 100).toFixed(0)}%</span>
          {count > 0 && (
            <span className="text-ink-faint">
              {' '}
              · {count} comparison{count === 1 ? '' : 's'} learned
            </span>
          )}
          .
        </div>
      </div>

      <div className="mt-3 font-sans text-xs text-ink-faint">
        Click the better response a few times: the reward model learns your
        preference, and the policy is then trained to produce higher-reward
        answers. (Illustrative — the responses are written, not generated.)
      </div>

      {caption && (
        <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}
