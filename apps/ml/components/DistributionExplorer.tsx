'use client'

import { useMemo, useState } from 'react'
import { createRng, gauss } from '@/lib/rng'

const W = 520
const H = 320
const PAD_L = 20
const PAD_R = 20
const PAD_T = 20
const PAD_B = 36
const XMIN = -5
const XMAX = 9
const BINS = 34
const N = 400

export default function DistributionExplorer() {
  const [mean, setMean] = useState(2)
  const [std, setStd] = useState(1.4)

  // fixed standard-normal draws, transformed by (mean, std) — stable histogram
  const z = useMemo(() => Array.from({ length: N }, (_, i) => gauss(createRng(i + 1), 0, 1)), [])
  const samples = useMemo(() => z.map((zi) => mean + std * zi), [z, mean, std])

  const hist = useMemo(() => {
    const counts = new Array(BINS).fill(0)
    const bw = (XMAX - XMIN) / BINS
    for (const s of samples) {
      const b = Math.floor((s - XMIN) / bw)
      if (b >= 0 && b < BINS) counts[b]++
    }
    return { counts, bw, max: Math.max(...counts, 1) }
  }, [samples])

  const sampleMean = samples.reduce((a, b) => a + b, 0) / N
  const sampleVar = samples.reduce((a, b) => a + (b - sampleMean) ** 2, 0) / N

  const sx = (x: number) => PAD_L + ((x - XMIN) / (XMAX - XMIN)) * (W - PAD_L - PAD_R)
  const sy = (frac: number) => H - PAD_B - frac * (H - PAD_T - PAD_B)

  // gaussian pdf scaled so its peak matches the histogram's tallest bar
  const pdf = (x: number) => Math.exp(-((x - mean) ** 2) / (2 * std * std)) / (std * Math.sqrt(2 * Math.PI))
  const pdfPeak = pdf(mean)
  const expectedPeakFrac = (N * hist.bw * pdfPeak) / hist.max // peak count vs histogram max
  const curve = Array.from({ length: 160 }, (_, k) => { const x = XMIN + (k / 159) * (XMAX - XMIN); return `${k === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${sy((pdf(x) / pdfPeak) * expectedPeakFrac).toFixed(1)}` }).join(' ')

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-2 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2"><span className="text-ink-muted">mean μ = <span className="font-mono">{mean.toFixed(1)}</span></span><input type="range" min={-2} max={6} step={0.1} value={mean} onChange={(e) => setMean(parseFloat(e.target.value))} className="w-40" /></label>
        <label className="flex items-center gap-2"><span className="text-ink-muted">std σ = <span className="font-mono">{std.toFixed(1)}</span></span><input type="range" min={0.5} max={3} step={0.1} value={std} onChange={(e) => setStd(parseFloat(e.target.value))} className="w-36" /></label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <line x1={PAD_L} y1={H - PAD_B} x2={W - PAD_R} y2={H - PAD_B} stroke="var(--rule)" strokeWidth={1} />
          {/* histogram */}
          {hist.counts.map((c, i) => {
            const x = sx(XMIN + i * hist.bw)
            const wpx = (W - PAD_L - PAD_R) / BINS
            return <rect key={i} x={x + 0.5} y={sy(c / hist.max)} width={wpx - 1} height={H - PAD_B - sy(c / hist.max)} fill="#3c5a8c" opacity={0.4} />
          })}
          {/* pdf */}
          <path d={curve} fill="none" stroke="var(--accent)" strokeWidth={2.25} />
          {/* mean / std markers */}
          <line x1={sx(mean)} y1={PAD_T} x2={sx(mean)} y2={H - PAD_B} stroke="var(--accent)" strokeWidth={1} strokeDasharray="3,3" />
          <line x1={sx(mean - std)} y1={sy(0.5)} x2={sx(mean + std)} y2={sy(0.5)} stroke="var(--ink-muted)" strokeWidth={1.25} />
          <text x={sx(mean + std)} y={sy(0.5) - 5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-mono, monospace)">±σ</text>
          {[XMIN, 0, mean, XMAX].map((v, i) => <text key={i} x={sx(v)} y={H - PAD_B + 14} fontSize={9} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">{v.toFixed(0)}</text>)}
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>sample mean = <span className="text-ink">{sampleMean.toFixed(2)}</span> <span className="text-ink-faint">(≈ μ)</span></div>
        <div>sample variance = <span className="text-ink">{sampleVar.toFixed(2)}</span> <span className="text-ink-faint">(≈ σ² = {(std * std).toFixed(2)})</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 3.3 — The shape of randomness. The bars are 400 random samples; the
        teal curve is the Gaussian (normal) distribution they were drawn from. Two
        numbers fix it entirely: the <strong>mean</strong> μ, where it is centred,
        and the <strong>standard deviation</strong> σ, how spread out it is. The
        sample mean and variance of the draws track μ and σ² closely. This bell is
        the default model of noise and uncertainty across the book — the "± noise"
        in every dataset, and the likelihood behind logistic regression and naive
        Bayes.
      </figcaption>
    </figure>
  )
}
