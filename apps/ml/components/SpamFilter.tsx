'use client'

import { useMemo, useState } from 'react'
import { SPAM_VOCAB, spamLogOdds } from '@/lib/naivebayes'

const SPAM = '#c7522a'
const HAM = '#3c5a8c'

const W = 520
const ROW_H = 26
const MIDX = 250 // centre of the evidence panel
const MAX_BAR = 150 // px at |logOdds| = 3
const SCALE = MAX_BAR / 3

export default function SpamFilter() {
  const [present, setPresent] = useState<boolean[]>(() =>
    SPAM_VOCAB.map((w) => w === 'meeting' || w === 'report'),
  )
  const [smoothing, setSmoothing] = useState(true)

  const alpha = smoothing ? 1 : 0
  const result = useMemo(() => spamLogOdds(present, alpha), [present, alpha])

  const active = SPAM_VOCAB.map((w, i) => ({ w, i })).filter((o) => present[o.i])
  const H = 60 + (active.length + 1) * ROW_H + 20

  const toggle = (i: number) =>
    setPresent((p) => p.map((v, k) => (k === i ? !v : v)))

  const clampBar = (lo: number) => {
    if (!isFinite(lo)) return lo > 0 ? MAX_BAR : -MAX_BAR
    return Math.max(-MAX_BAR, Math.min(MAX_BAR, lo * SCALE))
  }

  const pSpam = result.pSpam
  const pct = (pSpam * 100).toFixed(pSpam > 0.999 || pSpam < 0.001 ? 1 : 0)
  const needleX = 40 + pSpam * (W - 80)

  return (
    <figure className="my-10">
      {/* word toggles */}
      <div className="flex flex-wrap gap-2 mb-4 font-sans text-sm">
        {SPAM_VOCAB.map((w, i) => (
          <button
            key={w}
            onClick={() => toggle(i)}
            className="px-2.5 py-1 border rounded transition-colors"
            style={{
              borderColor: present[i] ? 'var(--ink)' : 'var(--rule)',
              background: present[i] ? 'var(--paper-soft)' : 'transparent',
              fontWeight: present[i] ? 600 : 400,
              color: present[i] ? 'var(--ink)' : 'var(--ink-muted)',
            }}
          >
            {w}
          </button>
        ))}
      </div>

      <label className="flex items-center gap-2 mb-4 font-sans text-sm cursor-pointer w-fit">
        <input type="checkbox" checked={smoothing} onChange={(e) => setSmoothing(e.target.checked)} />
        <span className="text-ink-muted">Laplace smoothing (add-one)</span>
      </label>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* needle gauge */}
          <text x={40} y={24} fontSize={11} fill={HAM} fontFamily="var(--font-sans, sans-serif)" fontWeight={600}>HAM</text>
          <text x={W - 40} y={24} fontSize={11} fill={SPAM} fontFamily="var(--font-sans, sans-serif)" fontWeight={600} textAnchor="end">SPAM</text>
          <defs>
            <linearGradient id="sf-grad" x1="0" x2="1">
              <stop offset="0" stopColor={HAM} stopOpacity={0.35} />
              <stop offset="0.5" stopColor="var(--rule)" stopOpacity={0.5} />
              <stop offset="1" stopColor={SPAM} stopOpacity={0.35} />
            </linearGradient>
          </defs>
          <rect x={40} y={30} width={W - 80} height={8} rx={4} fill="url(#sf-grad)" />
          <line x1={(W) / 2} y1={28} x2={W / 2} y2={40} stroke="var(--ink-faint)" strokeWidth={1} />
          <polygon points={`${needleX},28 ${needleX - 5},18 ${needleX + 5},18`} fill={pSpam >= 0.5 ? SPAM : HAM} />
          <text x={Math.max(78, Math.min(W - 78, needleX))} y={52} fontSize={11} fill={pSpam >= 0.5 ? SPAM : HAM} textAnchor="middle" fontFamily="var(--font-mono, monospace)">
            P(spam) = {pct}%
          </text>

          {/* evidence bars */}
          <line x1={MIDX} y1={62} x2={MIDX} y2={H - 16} stroke="var(--rule)" strokeWidth={1} />
          {/* prior */}
          {(() => {
            const bx = clampBar(result.prior)
            const y = 62 + ROW_H / 2 + 4
            return (
              <g>
                <rect x={Math.min(MIDX, MIDX + bx)} y={y - 7} width={Math.abs(bx)} height={11} fill="var(--ink-muted)" opacity={0.5} />
                <text x={MIDX + (bx >= 0 ? -6 : 6)} y={y + 3} fontSize={10} fill="var(--ink-muted)" textAnchor={bx >= 0 ? 'end' : 'start'} fontFamily="var(--font-mono, monospace)">prior</text>
              </g>
            )
          })()}
          {active.map((o, row) => {
            const lo = result.perWord[o.i]
            const bx = clampBar(lo)
            const y = 62 + (row + 1) * ROW_H + ROW_H / 2 + 4
            const col = lo >= 0 ? SPAM : HAM
            return (
              <g key={o.w}>
                <rect x={Math.min(MIDX, MIDX + bx)} y={y - 7} width={Math.abs(bx)} height={11} fill={col} opacity={0.8} />
                <text x={MIDX + (bx >= 0 ? -6 : 6)} y={y + 3} fontSize={10} fill="var(--ink)" textAnchor={bx >= 0 ? 'end' : 'start'} fontFamily="var(--font-mono, monospace)">{o.w}</text>
                <text x={(bx >= 0 ? MIDX + bx + 5 : MIDX + bx - 5)} y={y + 3} fontSize={9.5} fill={col} textAnchor={bx >= 0 ? 'start' : 'end'} fontFamily="var(--font-mono, monospace)">
                  {isFinite(lo) ? `${lo >= 0 ? '+' : ''}${lo.toFixed(2)}` : (lo > 0 ? '+∞' : '−∞')}
                </text>
              </g>
            )
          })}
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 16.3 — A naive Bayes spam filter. Each word in the message adds a
        piece of evidence — its log-likelihood ratio, pulling{' '}
        <span style={{ color: SPAM }}>towards spam</span> or{' '}
        <span style={{ color: HAM }}>towards ham</span> — and the running total
        (prior + every word) sets the needle. Toggle words and watch the scale
        tip. Now switch off Laplace smoothing and add{' '}
        <span className="font-mono">winner</span> or{' '}
        <span className="font-mono">prize</span>: never seen in a ham email,
        either one alone sends P(spam) to a certain 100% — one unseen word
        vetoing everything. Add-one smoothing is the fix.
      </figcaption>
    </figure>
  )
}
