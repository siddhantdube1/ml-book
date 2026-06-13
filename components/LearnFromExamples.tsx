'use client'

import { useMemo, useState } from 'react'
import { makeNoisyScatter, polyFeatures, ridgeRegression, evalPoly, trueCurve } from '@/lib/regularisation'

const W = 480
const H = 340
const PAD = 30
const XR = 1.08
const YR = 1.35

export default function LearnFromExamples() {
  const [flex, setFlex] = useState(3) // polynomial degree = model flexibility

  const data = useMemo(() => makeNoisyScatter(22, 7, 0.16), [])
  const fit = useMemo(() => {
    const xs = data.map((p) => p.x)
    const ys = data.map((p) => p.y)
    return ridgeRegression(polyFeatures(xs, flex), ys, 1e-7)
  }, [data, flex])

  const sx = (x: number) => PAD + ((x + XR) / (2 * XR)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y + YR) / (2 * YR)) * (H - 2 * PAD)

  const curve = useMemo(() => {
    const pts: string[] = []
    for (let k = 0; k <= 120; k++) {
      const x = -XR + (k / 120) * 2 * XR
      pts.push(`${k === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${sy(evalPoly(fit, x)).toFixed(1)}`)
    }
    return pts.join(' ')
  }, [fit])

  const regime = flex <= 1 ? 'too rigid — underfits' : flex >= 9 ? 'too flexible — memorises the noise' : 'about right'

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2"><span className="text-ink-muted">model flexibility = <span className="font-mono">{flex}</span></span><input type="range" min={1} max={14} step={1} value={flex} onChange={(e) => setFlex(parseInt(e.target.value))} className="w-52" /></label>
        <span className="font-mono text-xs" style={{ color: regime === 'about right' ? 'var(--accent)' : '#c7522a' }}>{regime}</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <line x1={PAD} y1={sy(0)} x2={W - PAD} y2={sy(0)} stroke="var(--rule)" strokeWidth={0.75} />
          {/* the true underlying rule (faint dashed) */}
          <path d={Array.from({ length: 121 }, (_, k) => { const x = -XR + (k / 120) * 2 * XR; return `${k === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${sy(trueCurve(x)).toFixed(1)}` }).join(' ')} fill="none" stroke="var(--ink-muted)" strokeWidth={1.25} strokeDasharray="4,4" opacity={0.5} />
          {/* the learned model */}
          <path d={curve} fill="none" stroke="var(--accent)" strokeWidth={2.25} />
          {/* the examples */}
          {data.map((p, i) => (
            <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={3.5} fill="#3c5a8c" stroke="var(--paper)" strokeWidth={0.7} />
          ))}
          <text x={PAD} y={20} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">examples (dots) · true rule (dashed) · learned model (teal)</text>
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 1.1 — Learning from examples. No one wrote the rule that produced
        these points; the model has to infer it from the examples alone. Turn the
        flexibility down and it is too stiff to follow the trend (underfitting);
        turn it up and it threads through every single point, chasing the random
        noise instead of the underlying shape (overfitting). Somewhere in between
        it recovers the true rule. That tension — fit the examples, but learn the
        pattern, not the noise — is the whole of machine learning in one slider.
      </figcaption>
    </figure>
  )
}
