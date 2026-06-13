'use client'

import { useMemo } from 'react'
import { makeCurveData, curveTrue } from '@/lib/foundations'
import { polyFeatures, ridgeRegression, evalPoly } from '@/lib/regularisation'

const W = 480
const H = 340
const PAD = 30
const XR = 1.08
const YR = 1.5

export default function TrainTestSplit() {
  const data = useMemo(() => makeCurveData(40, 5, 0.22), [])
  const train = useMemo(() => data.filter((_, i) => i % 4 !== 0), [data])
  const test = useMemo(() => data.filter((_, i) => i % 4 === 0), [data])
  const fit = useMemo(() => ridgeRegression(polyFeatures(train.map((p) => p.x), 3), train.map((p) => p.y), 1e-8), [train])

  const sx = (x: number) => PAD + ((x + XR) / (2 * XR)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y + YR) / (2 * YR)) * (H - 2 * PAD)

  const mse = (pts: typeof data) => pts.reduce((s, p) => s + (p.y - evalPoly(fit, p.x)) ** 2, 0) / pts.length
  const curve = Array.from({ length: 121 }, (_, k) => { const x = -XR + (k / 120) * 2 * XR; return `${k === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${sy(evalPoly(fit, x)).toFixed(1)}` }).join(' ')

  return (
    <figure className="my-10">
      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <line x1={PAD} y1={sy(0)} x2={W - PAD} y2={sy(0)} stroke="var(--rule)" strokeWidth={0.75} />
          <path d={Array.from({ length: 121 }, (_, k) => { const x = -XR + (k / 120) * 2 * XR; return `${k === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${sy(curveTrue(x)).toFixed(1)}` }).join(' ')} fill="none" stroke="var(--ink-muted)" strokeWidth={1} strokeDasharray="4,4" opacity={0.4} />
          <path d={curve} fill="none" stroke="var(--accent)" strokeWidth={2.25} />
          {/* residual stubs to test points */}
          {test.map((p, i) => <line key={`r${i}`} x1={sx(p.x)} y1={sy(p.y)} x2={sx(p.x)} y2={sy(evalPoly(fit, p.x))} stroke="#c7522a" strokeWidth={0.8} opacity={0.5} />)}
          {train.map((p, i) => <circle key={`tr${i}`} cx={sx(p.x)} cy={sy(p.y)} r={3.4} fill="#3c5a8c" stroke="var(--paper)" strokeWidth={0.6} />)}
          {test.map((p, i) => <circle key={`te${i}`} cx={sx(p.x)} cy={sy(p.y)} r={4} fill="none" stroke="#c7522a" strokeWidth={1.8} />)}
          <g transform="translate(40, 20)" fontFamily="var(--font-sans, sans-serif)" fontSize={10}>
            <circle cx={5} cy={0} r={3.4} fill="#3c5a8c" /><text x={13} y={3.5} fill="var(--ink-muted)">training (fit on these)</text>
            <circle cx={5} cy={15} r={4} fill="none" stroke="#c7522a" strokeWidth={1.8} /><text x={13} y={18.5} fill="var(--ink-muted)">test (judged on these)</text>
          </g>
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>training error = <span className="text-ink">{mse(train).toFixed(3)}</span></div>
        <div>test error = <span className="text-accent">{mse(test).toFixed(3)}</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 2.1 — Hold some data back. The model is fit using only the filled
        training points; the hollow test points are locked away and never shown to
        it. We then judge the model on those — its error on data it has never seen,
        a fair estimate of how it will do in the wild. Here the test error is close
        to the training error: the model genuinely learned the trend rather than
        memorising. The test set is the future, simulated; protecting it is the
        single most important habit in machine learning.
      </figcaption>
    </figure>
  )
}
