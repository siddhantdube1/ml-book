/**
 * Singleton Pyodide loader.
 *
 * Pyodide is roughly 6 MB on first download. We load it once, lazily, the
 * first time any PyodideEditor instance hits Run. Subsequent editors
 * (potentially many per chapter, across chapters) share the same instance.
 */

const PYODIDE_VERSION = '0.26.4'
const CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full`

declare global {
  interface Window {
    loadPyodide?: (opts: { indexURL: string }) => Promise<PyodideInstance>
  }
}

type PyodideInstance = {
  runPythonAsync: (code: string) => Promise<unknown>
  loadPackage: (packages: string[]) => Promise<void>
  setStdout: (opts: { batched: (s: string) => void }) => void
  setStderr: (opts: { batched: (s: string) => void }) => void
}

let pyodidePromise: Promise<PyodideInstance> | null = null
const loadedPackages = new Set<string>()

function injectPyodideScript(): Promise<void> {
  if (window.loadPyodide) return Promise.resolve()
  return new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.src = `${CDN}/pyodide.js`
    script.async = true
    script.onload = () => resolve()
    script.onerror = () =>
      reject(new Error('Could not load Pyodide from the CDN.'))
    document.head.appendChild(script)
  })
}

export function loadPyodideOnce(): Promise<PyodideInstance> {
  if (typeof window === 'undefined') {
    return Promise.reject(new Error('Pyodide requires a browser environment.'))
  }
  if (pyodidePromise) return pyodidePromise
  pyodidePromise = (async () => {
    await injectPyodideScript()
    if (!window.loadPyodide) {
      throw new Error('Pyodide script loaded but loadPyodide is undefined.')
    }
    return window.loadPyodide({ indexURL: `${CDN}/` })
  })()
  return pyodidePromise
}

export type RunResult = {
  stdout: string
  stderr: string
  error?: string
}

export async function runPython(
  code: string,
  packages: string[] = [],
  onProgress?: (status: string) => void,
): Promise<RunResult> {
  onProgress?.('loading Python runtime')
  const pyodide = await loadPyodideOnce()

  const toLoad = packages.filter((p) => !loadedPackages.has(p))
  if (toLoad.length > 0) {
    onProgress?.(`installing ${toLoad.join(', ')}`)
    await pyodide.loadPackage(toLoad)
    toLoad.forEach((p) => loadedPackages.add(p))
  }

  let stdout = ''
  let stderr = ''
  pyodide.setStdout({ batched: (s: string) => (stdout += s + '\n') })
  pyodide.setStderr({ batched: (s: string) => (stderr += s + '\n') })

  onProgress?.('running')
  try {
    await pyodide.runPythonAsync(code)
    return { stdout, stderr }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    return { stdout, stderr, error: message }
  }
}
