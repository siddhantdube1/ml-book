'use client'

import { useEffect, useRef, useState } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { EditorView } from '@codemirror/view'
import { runPython } from '@/lib/pyodide'

type Props = {
  /** Starter code shown in the editor. */
  initialCode: string
  /** Python packages to load on first run (e.g. ['numpy']). */
  packages?: string[]
  /** Editor height. Defaults to fit content. */
  minHeight?: string
  /** Short caption shown under the figure. */
  caption?: string
}

export default function PyodideEditor({
  initialCode,
  packages = [],
  minHeight = '220px',
  caption,
}: Props) {
  const [code, setCode] = useState(initialCode)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState('')
  const [stdout, setStdout] = useState('')
  const [stderr, setStderr] = useState('')
  const [error, setError] = useState('')
  const [mounted, setMounted] = useState(false)
  const outputRef = useRef<HTMLPreElement>(null)

  useEffect(() => {
    setMounted(true)
  }, [])

  const run = async () => {
    if (running) return
    setRunning(true)
    setStdout('')
    setStderr('')
    setError('')
    setProgress('starting')
    try {
      const result = await runPython(code, packages, (s) => setProgress(s))
      setStdout(result.stdout)
      setStderr(result.stderr)
      if (result.error) setError(result.error)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setRunning(false)
      setProgress('')
      requestAnimationFrame(() => {
        outputRef.current?.scrollTo({ top: outputRef.current.scrollHeight })
      })
    }
  }

  const reset = () => {
    setCode(initialCode)
    setStdout('')
    setStderr('')
    setError('')
  }

  const editorTheme = EditorView.theme({
    '&': {
      fontSize: '14px',
      fontFamily: 'var(--font-mono)',
      background: 'transparent',
    },
    '.cm-content': {
      padding: '14px 0',
      caretColor: 'var(--ink)',
      color: 'var(--ink)',
    },
    '.cm-gutters': {
      background: 'transparent',
      color: 'var(--ink-faint)',
      border: 'none',
      paddingRight: '12px',
    },
    '.cm-activeLine': { background: 'transparent' },
    '.cm-activeLineGutter': { background: 'transparent' },
    '.cm-line': { padding: '0 12px' },
    '.cm-cursor': { borderLeftColor: 'var(--ink)' },
    '.cm-selectionBackground': {
      background: 'color-mix(in srgb, var(--accent) 22%, transparent) !important',
    },
    '.cm-focused': { outline: 'none' },
  })

  return (
    <figure className="not-prose my-10">
      <div className="rounded-lg border border-rule overflow-hidden">
        <div
          className="flex items-center justify-between px-4 py-2 border-b border-rule"
          style={{
            background: 'color-mix(in srgb, var(--ink) 3%, var(--paper))',
          }}
        >
          <span className="font-sans text-xs uppercase tracking-wider text-ink-muted">
            Python · runs in browser
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={reset}
              disabled={running}
              className="font-sans text-xs uppercase tracking-wider px-3 py-1 rounded text-ink-muted hover:text-ink transition-colors disabled:opacity-40"
            >
              Reset
            </button>
            <button
              onClick={run}
              disabled={running || !mounted}
              className="font-sans text-xs uppercase tracking-wider px-3 py-1 rounded text-paper transition-opacity disabled:opacity-60"
              style={{ background: 'var(--accent)' }}
            >
              {running ? progress || 'running' : 'Run'}
            </button>
          </div>
        </div>

        <div
          style={{
            minHeight,
            background: 'var(--paper)',
          }}
        >
          {mounted ? (
            <CodeMirror
              value={code}
              onChange={(v) => setCode(v)}
              extensions={[python(), editorTheme, EditorView.lineWrapping]}
              basicSetup={{
                lineNumbers: true,
                foldGutter: false,
                highlightActiveLine: false,
                highlightActiveLineGutter: false,
                indentOnInput: true,
              }}
              theme="none"
              style={{ fontFamily: 'var(--font-mono)' }}
            />
          ) : (
            <pre
              className="font-mono text-sm p-4 m-0 whitespace-pre"
              style={{ color: 'var(--ink)' }}
            >
              {code}
            </pre>
          )}
        </div>

        {(stdout || stderr || error) && (
          <div className="border-t border-rule">
            <div className="px-4 py-2 border-b border-rule">
              <span className="font-sans text-xs uppercase tracking-wider text-ink-muted">
                Output
              </span>
            </div>
            <pre
              ref={outputRef}
              className="font-mono text-sm px-4 py-3 m-0 whitespace-pre-wrap break-words"
              style={{
                color: 'var(--ink)',
                background: 'color-mix(in srgb, var(--ink) 2%, var(--paper))',
                maxHeight: '320px',
                overflowY: 'auto',
              }}
            >
              {stdout}
              {stderr && (
                <span style={{ color: 'var(--ink-muted)' }}>{stderr}</span>
              )}
              {error && (
                <span style={{ color: '#c7522a' }}>
                  {stdout || stderr ? '\n' : ''}
                  {error}
                </span>
              )}
            </pre>
          </div>
        )}
      </div>
      {caption && (
        <figcaption className="font-sans text-sm text-ink-muted mt-3 text-center">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}
