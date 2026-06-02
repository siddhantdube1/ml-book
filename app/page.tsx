import Link from 'next/link'

type Chapter = {
  num: number
  title: string
  blurb: string
  href?: string
}

const parts: { title: string; chapters: Chapter[] }[] = [
  {
    title: 'I — Foundations',
    chapters: [
      { num: 1, title: 'What is machine learning?', blurb: 'Framing, vocabulary, and the supervised / unsupervised divide.' },
      { num: 2, title: 'The ML workflow', blurb: 'Data, train and test, generalisation, and the spectre of overfitting.' },
      { num: 3, title: 'A mathematical toolkit', blurb: 'Linear algebra, calculus, and probability — only what you need, exactly when you need it.' },
      { num: 4, title: 'Your first model: k-nearest neighbours', blurb: 'Building intuition for prediction without writing a single equation.' },
    ],
  },
  {
    title: 'II — Linear models',
    chapters: [
      { num: 5, title: 'Linear regression', blurb: 'The mother of all models. Closed form, gradient descent, and what they reveal.' },
      { num: 6, title: 'Gradient descent', blurb: 'The optimisation engine behind nearly every modern model.', href: '/chapters/6-gradient-descent' },
      { num: 7, title: 'Logistic regression', blurb: 'From regression to classification with one elegant change of perspective.', href: '/chapters/7-logistic-regression' },
      { num: 8, title: 'Multi-class classification', blurb: 'Softmax, one-vs-rest, and choosing between them.', href: '/chapters/8-multi-class-classification' },
      { num: 9, title: 'Regularisation', blurb: 'The geometric story of L1 and L2 and why they tame overfitting.', href: '/chapters/9-regularisation' },
    ],
  },
  {
    title: 'III — Evaluating models',
    chapters: [
      { num: 10, title: 'Evaluation metrics', blurb: 'Accuracy, precision, recall, ROC, and when each one lies.', href: '/chapters/10-evaluation-metrics' },
      { num: 11, title: 'Cross-validation and tuning', blurb: 'How to choose hyperparameters without fooling yourself.', href: '/chapters/11-cross-validation-and-tuning' },
    ],
  },
  {
    title: 'IV — Trees and ensembles',
    chapters: [
      { num: 12, title: 'Decision trees', blurb: 'Splitting, growing, pruning — and watching it all happen.', href: '/chapters/12-decision-trees' },
      { num: 13, title: 'Random forests', blurb: 'Why a crowd of imperfect trees beats one perfect one.', href: '/chapters/13-random-forests' },
      { num: 14, title: 'Gradient boosting', blurb: 'Sequential correction, and the road to XGBoost.' },
    ],
  },
  {
    title: 'V — More classical models',
    chapters: [
      { num: 15, title: 'Support vector machines', blurb: 'Maximum margins and the kernel trick, geometrically.' },
      { num: 16, title: 'Naive Bayes', blurb: 'Probability done plainly. A surprisingly strong baseline.' },
      { num: 17, title: 'Feature engineering', blurb: 'The unglamorous work that decides whether anything else matters.' },
    ],
  },
  {
    title: 'VI — Unsupervised learning',
    chapters: [
      { num: 18, title: 'k-means clustering', blurb: 'Finding structure without labels. Step through every iteration.', href: '/chapters/18-k-means' },
      { num: 19, title: 'Hierarchical and density-based clustering', blurb: 'When clusters are not blobs.' },
      { num: 20, title: 'PCA and dimensionality reduction', blurb: 'Seeing high-dimensional data as it really is.' },
    ],
  },
  {
    title: 'VII — Neural networks',
    chapters: [
      { num: 21, title: 'The perceptron and multilayer perceptrons', blurb: 'From a single neuron to a network.' },
      { num: 22, title: 'Backpropagation and training', blurb: 'How a network learns, derived from first principles.' },
    ],
  },
]

export default function Home() {
  return (
    <main className="min-h-screen px-6 py-16 md:py-24">
      <div className="mx-auto max-w-wide">
        <header className="mb-20 max-w-prose">
          <p className="font-sans text-xs uppercase tracking-[0.18em] text-ink-muted mb-6">
            An interactive handbook
          </p>
          <h1
            className="text-5xl md:text-6xl leading-[1.05] tracking-tight mb-6"
            style={{ fontWeight: 400 }}
          >
            Machine learning,
            <br />
            <em className="text-accent">step by step.</em>
          </h1>
          <p className="text-lg md:text-xl text-ink-muted leading-relaxed">
            Twenty-two chapters covering the foundations of machine learning,
            built around visualisations you can scrub through, code you can run
            in the browser, and problems that check your work. Designed for
            readers approaching the subject for the first time.
          </p>
        </header>

        <section>
          {parts.map((part) => (
            <div key={part.title} className="mb-14">
              <h2 className="font-sans text-xs uppercase tracking-[0.16em] text-ink-muted mb-6 pb-2 border-b border-rule">
                {part.title}
              </h2>
              <ul>
                {part.chapters.map((ch) => {
                  const inner = (
                    <div className="flex gap-6 py-4 border-b border-rule group">
                      <span className="font-sans text-sm text-ink-faint tabular-nums w-8 pt-0.5">
                        {String(ch.num).padStart(2, '0')}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p
                          className={`text-lg leading-snug ${
                            ch.href
                              ? 'text-ink group-hover:text-accent transition-colors'
                              : 'text-ink'
                          }`}
                          style={{ fontWeight: 500 }}
                        >
                          {ch.title}
                          {!ch.href && (
                            <span className="ml-2 font-sans text-xs uppercase tracking-wider text-ink-faint">
                              soon
                            </span>
                          )}
                        </p>
                        <p className="text-base text-ink-muted mt-1 leading-relaxed">
                          {ch.blurb}
                        </p>
                      </div>
                    </div>
                  )
                  return (
                    <li key={ch.num}>
                      {ch.href ? <Link href={ch.href}>{inner}</Link> : inner}
                    </li>
                  )
                })}
              </ul>
            </div>
          ))}
        </section>

        <footer className="mt-24 pt-8 border-t border-rule font-sans text-sm text-ink-faint">
          A work in progress.
        </footer>
      </div>
    </main>
  )
}
