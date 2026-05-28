# Voice & Style Guide

This file collects verbatim passages from the shipped chapters that exemplify the book's voice. **Read this before writing new prose.** The voice is the most important thing in the book — getting the prose right matters more than any visualization.

---

## The voice in one sentence

Confident, direct, intuition-first, opinionated, never condescending — like a smart, well-read colleague explaining something they understand well to a friend who's smart but new to the topic.

---

## Hallmarks (with examples)

### 1. Open with a sensory metaphor

Chapter 6 opens:

> Picture being dropped onto a hilly landscape, blindfolded. Your job: find the lowest point. You can't see anything. But you *can* feel the slope of the ground beneath your feet. So you take a small step in the steepest downhill direction, feel again, take another step. After many steps, you arrive somewhere low.
>
> That algorithm — *follow the slope down, repeatedly* — is gradient descent. And almost every machine learning model you'll meet for the rest of this book is trained using some variant of it.

Chapter 7 opens:

> So far the book has been about *regression* — predicting numbers. House prices. Temperatures. Heights. But a great many of the most useful machine-learning problems aren't about predicting numbers at all; they are about predicting *categories*. Is this email spam, or not? Is this transaction fraud, or not? Does this image contain a cat? These are *classification* problems.

The hook is concrete, not abstract. The reader knows what we're doing before we say what it's called.

### 2. Conversational asides; punchy sentences

> That's it. ML is mostly optimisation in a costume.

> That's it. That's the entire training algorithm.

> Learning slows to a crawl exactly when speedup is most needed.

> Local-minima panic was a 20th-century concern.

Short. Confident. Memorable. Use sparingly — once or twice per section.

### 3. Justify choices, never hedge

Bad:
> One approach we might consider is to perhaps try gradient descent...

Good (from Chapter 6):
> Gradient descent is the search algorithm of choice. It is not the only option (Newton's method, BFGS, evolutionary search, and others all exist), but it has a property the others lack: it scales.

State the position, then defend it. No "maybes" unless you genuinely mean it.

### 4. Set up tension, then resolve it

From Chapter 7 §2:

> The naive plan above almost works. But there are three problems with it, in increasing order of importance.

Then enumerate the three problems. Then resolve all three at once:

> The fix for all three problems is the same. We change the model in two small, related ways...

The reader is pulled along by the structure: tension → enumeration → resolution. They feel the *why* before the *how*.

### 5. Geometric intuition before formalism

From Chapter 6 §3:

> Here is the key fact, and it is worth pausing on: the gradient points in the direction of *steepest ascent*. Among all directions you could walk from the current $\theta$, the gradient direction is the one that increases $L$ the fastest. So to decrease $L$ the fastest, walk in the *opposite* direction: $-\nabla L$.

The math comes *after* the intuition. The reader is told what the gradient *does* before they see the formula for what it *is*.

### 6. End sections with a clinch, not a summary

From Chapter 7 §4 (after the DecisionBoundary widget):

> On **moons**, no straight line can separate the two interleaved arcs. This is the geometric reason logistic regression fails on non-linearly separable data. We'll see ways around this in later chapters (feature engineering, kernels, neural networks); for now, just feel the limit.

From Chapter 6 §1:

> Gradient descent is the search algorithm of choice. It is not the only option ... but it has a property the others lack: it scales. The cost per iteration stays manageable even when $\theta$ has billions of components, which is why it powers everything from linear models to LLMs.

Sections close on a sharp observation that earns the reader's attention for the next one.

### 7. British spelling

| Use | Not |
|---|---|
| optimise, optimisation | optimize, optimization |
| visualise, visualisation | visualize, visualization |
| behaviour | behavior |
| modelling, labelled | modeling, labeled |
| centre | center |
| analyse | analyze |

This is set across all three shipped chapters. Keep it consistent.

### 8. Em-dashes for emphasis

> The form is suspiciously similar to linear regression's gradient — that's not a coincidence.

> A subtle consequence: gradient descent will never reach the minimum in finite time on a smooth convex loss. It approaches asymptotically.

Em-dashes (— not – and not --) are the workhorses of mid-sentence emphasis. Use freely but not in every sentence.

### 9. Cross-reference past and future chapters

From Chapter 7:

> The training algorithm is now exactly Chapter 6's gradient descent.

> We will revisit metrics like accuracy, precision, recall, ROC, and calibration in Chapter 10.

> Convexity ... is in contrast to neural networks, which are non-convex.

The book is a unit. Remind the reader where they came from and where they're going. This is the *compounding* feel that makes the book feel built, not assembled.

### 10. Use the second person for interaction; third for facts

When guiding the reader through a widget or thought experiment:
> Drag the line or use the slider. Toggle the step function overlay to see how the sigmoid relates to its discontinuous counterpart.
> Try cranking sharpness up. Notice that the boundary doesn't move — only the steepness of the transition does.

When stating mathematical or technical facts:
> The decision boundary is a hyperplane.
> Cross-entropy is convex in $w$ and $b$.

Don't mix: "you should note that the loss is convex" — that's tutorial voice. Either "Note: the loss is convex." or "The loss is convex." is fine.

### 11. Foreshadow

From Chapter 7 §4:
> No straight line can separate the two interleaved arcs. This is the geometric reason logistic regression fails on non-linearly separable data. **We'll see ways around this in later chapters (feature engineering, kernels, neural networks); for now, just feel the limit.**

This single sentence sets up Chapters 15, 17, and 21-22. The reader knows there's more to come; they don't think the book just ran out of ideas when they hit the moons limit.

### 12. Numbered lists and bullets — use sparingly

Bullets are for genuinely parallel items, not for breaking up paragraphs. If a list could be a paragraph, make it one.

Good use (from Chapter 6 §4):

> The danger of too small:
> - Each step covers little ground.
> - Convergence requires many iterations.
> - For expensive losses (deep networks, large datasets), each iteration costs real time. A 10× smaller learning rate often means a 10× longer training run.

Bad use (don't do this):

> ❌ Logistic regression has several important properties:
> - It is convex
> - It is interpretable
> - It outputs probabilities
> - It is fast

The bullets here are weakly parallel; this should be prose: "Logistic regression has a clutch of pleasant properties: convexity (so training is robust), interpretability (each weight tells a story), probability outputs (not just labels), and speed."

---

## What to never do

- ❌ **Tutorial-speak**: "In this section we will learn..." "We are now going to explore..." "Let's discuss..."
- ❌ **Hedging without reason**: "It might be the case that..." "Some would argue..." "Perhaps..."
- ❌ **Over-formal academic register**: "We shall now proceed to derive..." "It follows from the foregoing that..."
- ❌ **Apologetic clarification**: "(Don't worry if this seems complicated...)" "(Bear with me here...)"
- ❌ **Marketing voice**: "powerful," "elegant," "groundbreaking," "revolutionary," "game-changing"
- ❌ **Throat-clearing transitions**: "Having now covered X, we move to Y." "Before we continue, let's recap." "So far we've seen..."
- ❌ **Filler words at the start of sentences**: "Now,..." "So,..." "Well,..." (occasional use is fine; not every paragraph)
- ❌ **"We" when you mean "you" or "the model"**: "Here we compute the gradient" → "The gradient is computed as..." or "You compute the gradient by..."

---

## Section transitions

End a section with a clinch (see #6 above). Start the next section with content, not with a backward reference. Don't write "Having seen X, let's now consider Y" — just start with Y.

Bad transition between §3 and §4:
> ❌ Now that we've seen the sigmoid function, we are ready to define the full logistic regression model.

Good transition (current chapter):
> [§3 ends with a widget showing the sigmoid.]
>
> ## 4. The model: σ(w·x + b)
>
> The logistic regression model is just linear regression's output piped through the sigmoid...

The reader doesn't need to be told the connection. The structure carries them.

---

## Opening every chapter

Every chapter opens with the same three structural elements, in order:

1. **Chapter label** (small uppercase tracking, e.g. "Chapter 7 · Part II — Linear models")
2. **H1 title** (sentence case: "Logistic regression" not "Logistic Regression")
3. **Italic one-line subtitle** that promises something concrete

Examples:

> *Finding the bottom of a hill, blindfolded. The algorithm that quietly powers nearly everything else in this book.*

> *Predicting probabilities instead of numbers. The moment the gradient descent we just built starts training real classifiers.*

Then **three opening paragraphs** that:
1. Set the sensory hook or motivating example.
2. Say what this chapter teaches.
3. Promise what the reader will have built by the end.

The third one often takes the form: "By the end of this chapter you will have trained a real classifier from scratch using nothing but the gradient descent of Chapter 6."

---

## Closing every chapter

Every chapter closes with a horizontal rule and a single italic line:

```mdx
---

*Next: Chapter N+1 — <Title>.* One-sentence hook that previews the next chapter's idea.
```

Examples:

> *Next: Chapter 7 — Logistic regression.* The first time we use gradient descent to train a model that actually classifies things.

> *Next: Chapter 8 — Multi-class classification.* What happens when you have three or more classes, and the trick (softmax) that lets logistic regression generalise to as many classes as you like.

No further closing text. The book ends each chapter on this forward-looking note.

---

## A model paragraph

Chapter 7 §5 paragraph that exemplifies everything together:

> Read the formula carefully. For each training example only one of the two terms is non-zero — the one where the indicator matches the label. If $y_i = 1$, the term is $-\log p_i$, which is small when $p_i$ is near 1 (the prediction matches) and grows large as $p_i \to 0$ (the prediction is confident and wrong). If $y_i = 0$, it's $-\log(1 - p_i)$, symmetric in the opposite direction. So cross-entropy punishes confident-wrong predictions extremely harshly — they push $\log p$ toward $-\infty$ — while letting near-the-threshold mistakes off with a small penalty. Exactly the shape we wanted.

What's good here:
- **Imperative opener** ("Read the formula carefully") — pulls the reader's attention
- **Structural parallelism** ("If $y_i = 1$... If $y_i = 0$...") — easy to follow
- **Geometric reading** of the math ("small when $p_i$ is near 1") — math is interpreted, not just stated
- **Voice clinch at the end** ("Exactly the shape we wanted") — sticks the landing
- **Em-dashes** for emphasis, used naturally
- **Concrete language** ("dangerous," "harshly," "small penalty") — not academic register
