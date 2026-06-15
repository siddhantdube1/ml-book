// Shared types for the cached model fixtures. Mechanism fixtures (attention,
// next-token logits) are recorded offline from GPT-2; behaviour fixtures
// (completions, traces) are recorded from a frontier model. See
// docs/TRILOGY-DESIGN.md §4.

/** One decoded token plus its index in the model vocabulary. */
export type Token = {
  /** Display string (GPT-2 byte-tokens decoded; spaces/newlines visible). */
  text: string
  /** Vocabulary id in the recording model. */
  id: number
}

/**
 * Self-attention weights recorded from GPT-2 for one input string.
 * `attention[layer][head][query][key]` is in [0, 1]; each query row sums to 1.
 */
export type AttentionFixture = {
  id: string
  model: string
  text: string
  tokens: Token[]
  numLayers: number
  numHeads: number
  attention: number[][][][]
}

/** A candidate next token with its raw logit and softmax probability (T=1). */
export type TokenProb = {
  text: string
  id: number
  logit: number
  prob: number
}

/**
 * The next-token distribution after some context, truncated to the top-k
 * candidates by probability. The widget re-derives temperature/top-k/top-p
 * effects from the stored logits.
 */
export type NextTokenFixture = {
  id: string
  model: string
  context: string
  topk: TokenProb[]
}

/** One word in an embedding projection: 2D display coords + true neighbours. */
export type EmbeddingWord = {
  text: string
  group: string
  /** PCA-projected display coordinates, normalised to [0, 1]. */
  x: number
  y: number
  /** Nearest neighbours computed in the full embedding space (cosine). */
  neighbours: string[]
}

export type Analogy = {
  a: string
  b: string
  c: string
  want: string
  predicted: string[]
  hit: boolean
}

export type EmbeddingFixture = {
  model: string
  words: EmbeddingWord[]
  analogies: Analogy[]
}
