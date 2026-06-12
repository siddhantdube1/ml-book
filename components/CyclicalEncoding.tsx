'use client'

import { useState } from 'react'
import { cyclical } from '@/lib/features'

const W = 560
const H = 340
const BUSY = (h: number) => h >= 22 || h <= 3
const COL = (h: number) => (BUSY(h) ? '#c7522a' : '#3c5a8c')

// left number line
const LX0 = 46
const LX1 = 250
const LY = 150
// right clock
const CX = 430
const CY = 150
const R = 96

export default function CyclicalEncoding() {
  const [hour, setHour] = useState(23)
  const hours = Array.from({ length: 24 }, (_, h) => h)
  const lx = (h: number) => LX0 + (h / 23) * (LX1 - LX0)
  const clockPos = (h: number): [number, number] => {
    const [s, c] = cyclical(h, 24)
    return [CX + s * R, CY - c * R]
  }
  const [sinv, cosv] = cyclical(hour, 24)

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Hour = <span className="font-mono">{String(hour).padStart(2, '0')}:00</span></span>
          <input type="range" min={0} max={23} step={1} value={hour} onChange={(e) => setHour(parseInt(e.target.value))} className="w-52" />
        </label>
        <span className="font-mono text-xs" style={{ color: COL(hour) }}>{BUSY(hour) ? 'busy' : 'quiet'}</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* ── Left: hour as a raw number on a line ── */}
          <text x={LX0} y={30} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">raw: hour as a number</text>
          <line x1={LX0} y1={LY} x2={LX1} y2={LY} stroke="var(--rule)" strokeWidth={1.5} />
          {/* "actually adjacent" arc between 23 and 0 */}
          <path d={`M ${lx(0)} ${LY - 8} Q ${(lx(0) + lx(23)) / 2} ${LY - 62} ${lx(23)} ${LY - 8}`} fill="none" stroke="var(--ink-faint)" strokeWidth={1} strokeDasharray="3,3" />
          <text x={(lx(0) + lx(23)) / 2} y={LY - 50} fontSize={9.5} fill="var(--ink-faint)" textAnchor="middle" fontStyle="italic">1 hour apart, really</text>
          {hours.map((h) => (
            <g key={h}>
              <circle cx={lx(h)} cy={LY} r={h === hour ? 6 : 3.5} fill={COL(h)} stroke={h === hour ? 'var(--ink)' : 'var(--paper)'} strokeWidth={h === hour ? 1.5 : 0.6} />
              {h % 6 === 0 && <text x={lx(h)} y={LY + 20} fontSize={9} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">{h}</text>}
            </g>
          ))}
          <text x={lx(0)} y={LY + 36} fontSize={9} fill="var(--ink-faint)" textAnchor="middle" fontStyle="italic">far apart on the line…</text>

          {/* ── Right: hours on a clock-face circle via (sin, cos) ── */}
          <text x={CX - R} y={30} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">encoded: (sin, cos) on a circle</text>
          <circle cx={CX} cy={CY} r={R} fill="none" stroke="var(--rule)" strokeWidth={1.5} />
          {hours.map((h) => {
            const [px, py] = clockPos(h)
            return (
              <g key={h}>
                <circle cx={px} cy={py} r={h === hour ? 6 : 3.5} fill={COL(h)} stroke={h === hour ? 'var(--ink)' : 'var(--paper)'} strokeWidth={h === hour ? 1.5 : 0.6} />
                {h % 6 === 0 && <text x={CX + (px - CX) * 1.16} y={CY + (py - CY) * 1.16 + 3} fontSize={9} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">{h}</text>}
              </g>
            )
          })}
          {/* line from centre to current hour */}
          <line x1={CX} y1={CY} x2={clockPos(hour)[0]} y2={clockPos(hour)[1]} stroke="var(--ink-faint)" strokeWidth={1} />
          <text x={CX} y={CY + R + 26} fontSize={9} fill="var(--ink-faint)" textAnchor="middle" fontStyle="italic">…neighbours on the circle</text>
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>raw feature = <span className="text-ink">{hour}</span></div>
        <div>(sin, cos) = <span className="text-ink">({sinv.toFixed(2)}, {cosv.toFixed(2)})</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 17.4 — Putting a periodic feature on a circle. Encode the hour as
        the plain number 0–23 (left) and you tell the model that 23:00 and 00:00
        sit at opposite ends of the world, when they are a single hour apart —
        so it cannot learn the late-night pattern that straddles midnight. Map
        the hour through <span className="font-mono">(sin, cos)</span> onto a
        clock face (right) and the geometry tells the truth: 23:00 and 00:00
        become neighbours, and the wrap-around pattern is suddenly learnable.
        The encoding <em>is</em> the information.
      </figcaption>
    </figure>
  )
}
