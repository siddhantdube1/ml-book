// Pure decoding math for the sampling widget (LLM Ch 14). Operates on the
// truncated logit list from a NextTokenFixture; the tail beyond the stored
// top-k is negligible for typical contexts, so re-normalising over the stored
// candidates is faithful for visualisation.

/** softmax(logits / T). T -> 0 sharpens toward greedy; large T flattens. */
export function temperatureProbs(logits: number[], temperature: number): number[] {
  const t = Math.max(temperature, 1e-6)
  const scaled = logits.map((l) => l / t)
  const max = Math.max(...scaled)
  const exps = scaled.map((s) => Math.exp(s - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

/** Keep only the k highest-probability entries; zero the rest; renormalise. */
export function topKMask(probs: number[], k: number): number[] {
  if (k <= 0 || k >= probs.length) return probs.slice()
  const cutoff = [...probs].sort((a, b) => b - a)[k - 1]
  const kept = probs.map((p) => (p >= cutoff ? p : 0))
  return renormalise(kept)
}

/**
 * Nucleus (top-p): keep the smallest set of highest-probability entries whose
 * cumulative mass reaches p; zero the rest; renormalise.
 */
export function topPMask(probs: number[], p: number): number[] {
  if (p >= 1) return probs.slice()
  const order = probs
    .map((prob, i) => ({ prob, i }))
    .sort((a, b) => b.prob - a.prob)
  const keep = new Set<number>()
  let cum = 0
  for (const { prob, i } of order) {
    keep.add(i)
    cum += prob
    if (cum >= p) break
  }
  const kept = probs.map((prob, i) => (keep.has(i) ? prob : 0))
  return renormalise(kept)
}

export function renormalise(probs: number[]): number[] {
  const sum = probs.reduce((a, b) => a + b, 0)
  if (sum === 0) return probs.slice()
  return probs.map((p) => p / sum)
}

export type DecodeOptions = {
  temperature: number
  topK?: number
  topP?: number
}

/** Compose temperature -> top-k -> top-p into a final distribution. */
export function decodeDistribution(
  logits: number[],
  { temperature, topK = 0, topP = 1 }: DecodeOptions,
): number[] {
  let probs = temperatureProbs(logits, temperature)
  if (topK > 0) probs = topKMask(probs, topK)
  if (topP < 1) probs = topPMask(probs, topP)
  return probs
}
