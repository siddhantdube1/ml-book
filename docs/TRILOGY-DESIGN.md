# The Trilogy — Design Document

> Master plan for turning the single ML book into a three-book interactive
> platform: **Classical ML → Large Language Models → Agents**. This doc is the
> source of truth for the architecture, the writing approach, and the chapter
> outlines of the two new books. It will move to the monorepo root when we
> extract the platform.
>
> Decisions locked with the author (2026-06-15):
> 1. Outline both new books now; **build the LLM book first**.
> 2. Interactive demos are **cached-first, live-optional** (no mandatory backend).
> 3. Codebase becomes a **monorepo with a shared platform package**.

---

## 1. The arc — why three books, in this order

The existing ML book ends at the multi-layer perceptron and backpropagation
(Ch 21–22). That is exactly where deep learning begins. The trilogy is one
continuous staircase:

| Book | Starts from | Ends at | The reader gains |
|---|---|---|---|
| **ML** (shipped) | "what is a model" | MLP + backprop | how models *learn* |
| **LLMs** (build first) | a neural net | a full transformer + its behaviour | how language models *work* |
| **Agents** (build second) | an LLM as a fixed capability | production agentic systems | how to make models *act* |

Each book hands the next its vocabulary. The LLM book opens by reminding the
reader they already built a neural net; the Agents book opens by treating the
LLM as a solved black box. Cross-book links are first-class — the trilogy
should feel *built*, not assembled, exactly like the chapters of one book do.

---

## 2. Core principle — port the soul, not the template

The ML book's 11-section structure and frame-playback widget are not the
magic. They are a perfect *fit* to classical ML's nature: small, iterative,
numerical algorithms over 2D data, runnable from scratch in numpy. That nature
does not survive the jump to LLMs and Agents, so the template must not be
copied blindly.

**What carries over verbatim (the soul):**
- The voice — confident, intuition-first, sensory hooks, British spelling,
  em-dashes, clinch endings. See `docs/VOICE-AND-STYLE.md`; it is unchanged.
- Intuition before formalism. (For LLMs/Agents: *mechanistic* intuition before
  architecture, rather than *geometric* intuition before math.)
- Manipulate the core object, observe the consequence — the reader's hand on
  the thing.
- Compounding structure: every chapter arms the next.
- The design system (paper/ink/accent tokens, the three fonts).

**What changes (the body):**
- **No rigid 11-section arc.** Chapters vary in shape because the subjects do.
- **No assumption that every chapter ends in a numpy implementation.** Each
  chapter declares one of three *build modes* (see §6): `numpy-from-scratch`,
  `prompt-experiment`, or `trace-analysis`.
- **A new widget vocabulary** (§5) replaces the 2D-plot-plus-scrubber default.
- **The interactive substrate moves from local numpy to a real hosted model**
  behind the page (§4). This is the single biggest architectural change.

If a chapter genuinely fits the ML mold (e.g. *Scaling laws* is just 2D plots),
reuse it wholesale. The rule is fit-to-subject, not novelty for its own sake.

---

## 3. Monorepo architecture

npm workspaces. Three thin book apps, one fat shared platform package.

```
trilogy/                          (repo root; was ml-book/)
├── package.json                  workspaces: ["apps/*", "packages/*"]
├── docs/
│   ├── TRILOGY-DESIGN.md         this file
│   ├── VOICE-AND-STYLE.md        unchanged, shared by all books
│   └── CHAPTER-RECIPE.md         updated per-book recipes
├── packages/
│   └── book-kit/                 THE PLATFORM — everything reusable
│       ├── theme/                globals.css tokens, ThemeToggle, no-FOUC script
│       ├── nav/                  ChapterNav, chapter registry types, link resolver
│       ├── mdx/                  mdx-components, remark-chapter-links (cross-book aware)
│       ├── code/                 PyodideEditor, Tex, pyodide singleton
│       ├── model/                cached-first model client + live-key provider (§4)
│       ├── viz/                  shared primitives: Axes, Scatter, Heatmap, Plot,
│       │                         usePlayback(frames), coord transforms
│       └── design.config.ts      tailwind preset (fonts + colour tokens)
└── apps/
    ├── ml/                       the existing book, near-untouched
    │   ├── app/chapters/...      22 chapters
    │   ├── components/           ML-specific widgets (the 84 we have)
    │   └── lib/                  ML algorithms (kmeans, gradient, logistic…)
    ├── llm/                      new
    └── agents/                   new
```

Each app is *thin*: its chapters, its book-specific widgets, its content
registry. Theme, nav, MDX wiring, the Pyodide harness, shared viz primitives,
and the model client all come from `@trilogy/book-kit`.

**Deploy:** each app is its own Vercel project with its own domain/subdomain.
The live `ml-book-seven.vercel.app` keeps working — `apps/ml` is the same Next
app with imports repointed at `book-kit`. Cross-book links are absolute URLs
resolved through one table in `book-kit/nav`.

**Migration (the one invasive step, done first, on a branch):**
1. Create the workspace skeleton; move current root into `apps/ml`.
2. Extract the genuinely-shared pieces into `packages/book-kit`
   (theme, ThemeToggle, ChapterNav, mdx-components, remark plugin, PyodideEditor,
   Tex, pyodide loader, the viz primitives that aren't ML-specific).
3. Repoint `apps/ml` imports. **Verify `apps/ml` builds byte-for-byte the same
   pages** (same 29 routes, same bundle shape) before writing a single new word.
   This refactor must be a no-op for the reader of the ML book.

---

## 4. The interactive substrate — cached-first, live-optional

ML widgets run real numpy locally because the algorithms are tiny. LLM/Agent
demos need a real model behind the page. We keep the site essentially static.

### 4.0 The mechanism / behaviour split (important)

Frontier-model APIs (Claude, GPT-4) **do not expose attention weights, hidden
states, or — for Anthropic — token logprobs.** So demos split by what they show:

- **Mechanism demos** (attention maps, embeddings, logits/sampling, the forward
  pass) need an *inspectable* model. Source: **GPT-2-class, recorded offline
  into JSON fixtures** (decided). Real, inspectable, deterministic, free at
  runtime, light bundle. Recorder uses transformers.js in Node (`output_
  attentions` / logits) — no Python/torch. No live mode (the open model's
  internals are the point; arbitrary-input live inference is a later nice-to-have).
- **Behaviour demos** (completions, base-vs-instruct, prompting, hallucination,
  reasoning, agent traces) use a real frontier model. Source: **latest Claude**
  (decided), recorded into fixtures + offered as opt-in live mode.

`book-kit/model` serves both behind one fixture-loading interface; widgets don't
care which kind they are.

**Default (cached):** canonical demos read *pre-recorded real model outputs*
committed as JSON fixtures — token distributions, attention matrices,
completions, agent traces. Deterministic, free at runtime, statically hostable,
always works offline. An offline script `scripts/record.ts` calls the real API
once per fixture and saves the result; fixtures are reviewed like any asset.

**Progressive enhancement (live):** a `LiveKeyProvider` lets a reader paste
*their own* API key (kept in `localStorage`, never sent to our servers).
Widgets detect a key and swap the fixture for a live call made
**browser → model API directly** (Anthropic supports a direct-browser-access
header; if a provider blocks CORS we add a thin stateless proxy later). No
backend, no key of ours, no unbounded cost. Live mode is a labelled toggle, not
the default.

```
book-kit/model/
├── client.ts        getCompletion(id) · getNextTokenDist(id) · getAttention(id) · getTrace(id)
├── fixtures/        committed JSON: the cached canonical outputs
├── LiveKeyProvider  context: reader's key, fixture⇄live switch, cost meter
└── record.ts        offline: regenerate a fixture from the real API
```

Every model-backed widget takes a `fixtureId` and renders from cache by
default; if a live key is present and the reader hits "run live", it calls
through. One code path, two data sources.

---

## 5. New widget vocabulary

Six reusable patterns cover almost everything across both books. Three are new;
three are ports of what the ML book already does well.

| Pattern | New? | Used for | Substrate |
|---|---|---|---|
| **Inspect-a-matrix** | new | attention heatmaps, embedding similarity, logit lens | cached fixture |
| **Distribution-reshaper** | new | sampling: temperature / top-k / top-p over a real vocab dist | cached + live |
| **Trace-stepper** | new | ReAct loops, tool calls, agent graphs (Agents flagship) | cached fixture |
| **Token-stream playback** | port | autoregressive generation, training-sample evolution | reuses `usePlayback` |
| **Pipeline-flow** | port-ish | transformer block dataflow, RAG retrieve→ground | client render |
| **2D-projection / plot** | port | embeddings in 2D, scaling-law curves | reuses ML viz primitives |

The flagship widgets — the ones that, if they land, prove the whole approach:
the **draggable attention heatmap** (LLM Ch 4), the **live sampling
distribution-reshaper** (LLM Ch 14), and the **ReAct trace-stepper** (Agents
Ch 3).

---

## 6. Build modes (replaces "every chapter ends in numpy")

Each chapter declares how the reader *builds* understanding by doing:

- **`numpy-from-scratch`** — kept where it genuinely teaches: implement
  attention / a tokenizer / softmax-sampling on toy inputs in Pyodide. The ML
  book's §10 ritual, intact, for the mechanistic chapters.
- **`prompt-experiment`** — the reader drives the real (cached/live) model:
  craft a prompt, change temperature, add few-shot examples, compare base vs
  instruct. The "implementation" is interaction with the real thing.
- **`trace-analysis`** — (Agents) the reader reads and perturbs a recorded
  agent trajectory: where did it branch, why did it loop, what if this tool
  failed.

Problems sections survive but adapt to the chapter's build mode.

---

## 7. LLM book — full outline (BUILD FIRST)

Working title: **"How Language Models Work."** Through-line: start from
next-token prediction and never let go — every chapter adds one layer (tokens →
vectors → attention → blocks → depth → training → behaviour) until the reader
can trace any LLM behaviour to a mechanism. Assumes the ML book (nets, GD,
softmax, cross-entropy, embeddings).

**Part I — From text to a language model**
1. **What is a language model?** Next-token prediction as the one objective;
   it's just classification over a vocabulary. *Widget:* next-token predictor
   (type a prefix → real top-k). *Build:* prompt-experiment. *Hook back:* ML
   Ch 7–8 (this is softmax classification, vocabulary-sized).
2. **Tokenization.** BPE; why models can't spell or count letters; the
   tokenizer as the model's senses. *Widget:* live tokenizer (fully
   client-side). *Build:* numpy-from-scratch (BPE merge loop).
3. **Embeddings: meaning as geometry.** Token IDs → vectors; analogy
   arithmetic; static vs contextual. *Widget:* 2D embedding explorer (reuses ML
   scatter). *Build:* numpy-from-scratch. *Hook back:* ML Ch 20 (PCA).

**Part II — The Transformer**
4. **Attention.** Tokens look at tokens; query/key/value; weighted average.
   *Widget:* **attention heatmap** — drag a query token, watch the distribution
   over keys (flagship). *Build:* numpy-from-scratch (attention on toy inputs).
5. **Multi-head self-attention.** Different heads, different relations
   (syntax, coreference). *Widget:* multi-head viewer (switch heads). *Build:*
   prompt-experiment.
6. **The Transformer block.** Residual stream + attention + MLP + layernorm;
   the residual stream as working memory. *Widget:* block dataflow
   (pipeline-flow). *Build:* numpy-from-scratch (a block forward pass).
7. **Depth.** Stacking blocks; early-syntax/late-semantics; the forward pass as
   iterative refinement. *Widget:* logit-lens — top prediction per layer.
   *Build:* trace-analysis (read the layers).
8. **Positional information.** Attention is order-blind; sinusoidal → learned →
   RoPE. *Widget:* positional-pattern viewer. *Build:* numpy-from-scratch.
9. **The full forward pass.** embed → N blocks → unembed → logits → sample →
   append → repeat. *Widget:* **token-by-token generation playback** (reuses
   `usePlayback`). *Build:* numpy-from-scratch (tie it all together, toy model).

**Part III — How they're trained**
10. **Pretraining.** The objective at scale; self-supervision (labels are
    free); the data. *Widget:* loss-curve + sample-quality scrubber. *Build:*
    prompt-experiment.
11. **Scaling laws.** Loss as a power law in params/data/compute;
    compute-optimal (Chinchilla). *Widget:* scaling-law explorer (pure 2D plots,
    reuses ML viz). *Build:* numpy-from-scratch (fit a power law). *Pure ML-mold
    chapter — reuse the template.*
12. **Fine-tuning & instruction-tuning.** Base (autocomplete) → assistant; SFT;
    why base models are weird. *Widget:* base-vs-instruct side-by-side.
    *Build:* prompt-experiment.
13. **Learning from preferences (RLHF / DPO).** Reward models, preference data,
    aligning behaviour. *Widget:* preference → policy shift (simulated).
    *Build:* prompt-experiment.

**Part IV — How they behave**
14. **Sampling & decoding.** Greedy / temperature / top-k / top-p; the
    creativity dial. *Widget:* **live sampling distribution-reshaper** (flagship,
    cached+live). *Build:* numpy-from-scratch (implement the samplers).
15. **The context window.** Quadratic attention; KV cache; lost-in-the-middle.
    *Widget:* attention-cost + U-shaped retrieval. *Build:* trace-analysis.
16. **Prompting & in-context learning.** Few-shot without weight updates; why it
    works (induction/pattern completion). *Widget:* few-shot playground.
    *Build:* prompt-experiment.
17. **Why models hallucinate.** Confabulation as a consequence of the objective;
    calibration; knowing vs predicting. *Widget:* calibration diagram (reuses ML
    Ch 10). *Build:* prompt-experiment. *Hook back:* ML Ch 7 §8.
18. **Evaluation.** Perplexity, benchmarks, contamination, the eval crisis.
    *Mostly prose.*

**Part V — The frontier**
19. **Architecture variations.** MoE, long-context, multimodality. *Prose +
    diagrams.*
20. **Reasoning models.** Chain-of-thought, test-time compute, thinking before
    answering. *Widget:* CoT trace. *Bridge:* explicitly foreshadows the Agents
    book — a model that reasons in steps can plan.
21. **What we don't understand.** Interpretability, superposition, features, the
    residual stream as a concept space. Honest close on the open problems.

**Prove-the-format-first set (mirrors how ML shipped Ch 6/7/18 before the
rest):** Ch 4 *Attention* (the flagship viz), Ch 14 *Sampling* (proves the
cached+live substrate), Ch 2 *Tokenization* (fully client-side real
interactivity, easy win). If these three land, the remaining 18 follow the
recipe.

---

## 8. Agents book — outline (BUILD SECOND)

Working title: **"Building Agents."** Through-line: the LLM is now a fixed
reasoner; an agent wraps it in a loop with tools, memory, and goals — each
chapter adds one element of the loop or one system concern.

**Part I — From model to agent**
1. **What is an agent?** The perceive→decide→act→observe loop; the LLM as the
   reasoner inside it. Clinch: "an agent is a language model that can press
   buttons." *Widget:* the agent loop (trace-stepper, toy task).
2. **Tool use / function calling.** The fundamental primitive; schemas; the
   model emits a call, you execute, return the result. *Widget:* tool-call
   trace.
3. **The ReAct loop.** Thought → Action → Observation, interleaved. *Widget:*
   **ReAct trace-stepper** (flagship). *Build:* trace-analysis.
4. **The system prompt.** The agent's constitution — role, tools, constraints,
   format. *Widget:* prompt → behaviour.

**Part II — Capabilities**
5. **Memory.** Working (context) vs persistent; what to keep/summarise/forget.
   *Widget:* memory inspector.
6. **Retrieval-augmented generation.** Embed → retrieve → ground; the most
   common pattern. *Widget:* RAG pipeline (toggle retrieval → watch grounding).
   *Hook back:* LLM Ch 3 (embeddings).
7. **Planning & decomposition.** Goals into steps; plan-and-execute vs ReAct.
   *Widget:* plan tree.
8. **Reflection & self-correction.** Self-critique, retry, verification.
   *Widget:* reflect loop.

**Part III — Systems**
9. **Multi-agent systems.** Orchestration, sub-agents, handoffs. *Widget:* agent
   graph.
10. **Environments & sandboxing.** Code execution, computer use, the danger
    surface. *Widget:* sandboxed tool run.
11. **Orchestration patterns.** Graphs, state machines, control flow.
12. **Human-in-the-loop.** Approval gates, oversight, when to ask. *Widget:*
    approval gate.

**Part IV — Making them work**
13. **Evaluating agents.** Trajectory evaluation — judging the process, not just
    the answer.
14. **Failure modes.** Loops, hallucinated tools, context rot, error cascades,
    goal drift. *Widget:* failure gallery.
15. **Cost, latency, reliability.** The engineering reality of production
    agents.
16. **Safety & permissions.** Sandboxing, least privilege, the stakes of acting
    in the world.

**Part V — The frontier**
17. **Continual-learning agents.** Memory that compounds; learning from
    experience.
18. **The road ahead.** Closes the trilogy.

**Prove-the-format-first set:** Ch 2 *Tool use*, Ch 3 *ReAct stepper* (the
flagship trace widget), Ch 6 *RAG* (tests cross-book compounding with LLM
embeddings).

---

## 9. Sequencing

1. **Platform extraction** — monorepo skeleton; `apps/ml` migration;
   `book-kit`. ML book must build identically. *(Invasive; on a branch;
   confirm before merging.)*
2. **Model substrate** — `book-kit/model` client, fixture format, `record.ts`,
   `LiveKeyProvider`. Build against one fixture (Ch 4 attention).
3. **LLM prove-the-format trio** — Ch 4, 14, 2, in that order. Review voice +
   widgets with the author after Ch 4.
4. **LLM remaining 18** — recipe-driven.
5. **Agents prove-the-format trio**, then the rest.

---

## 10. Open questions for later (not blocking)

- Domains/subdomains per book, and whether the three share a top-level landing.
- In-book disclosure of which models the fixtures came from (GPT-2 / Claude).
- Whether scaling-law / interp chapters want any real data vs illustrative.
- A shared "trilogy" home page tying the three books together.

*Last updated: 2026-06-15, planning session with Siddhant. Status: platform
extracted and verified — monorepo live (`apps/ml` builds identically, `apps/llm`
scaffolded on `book-kit`). Model substrate decided (GPT-2 fixtures for
mechanism, latest Claude for behaviour); building it next, then the LLM
prove-the-format trio (Ch 4, 14, 2).*
