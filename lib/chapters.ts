// Single source of truth for chapter order, slugs, and titles.
// Used by the prev/next navigation; slugs match the folders in app/chapters.

export type ChapterMeta = {
  num: number
  slug: string
  title: string
}

export const chapters: ChapterMeta[] = [
  { num: 1, slug: '1-what-is-machine-learning', title: 'What is machine learning?' },
  { num: 2, slug: '2-ml-workflow', title: 'The ML workflow' },
  { num: 3, slug: '3-mathematical-toolkit', title: 'A mathematical toolkit' },
  { num: 4, slug: '4-k-nearest-neighbours', title: 'Your first model: k-nearest neighbours' },
  { num: 5, slug: '5-linear-regression', title: 'Linear regression' },
  { num: 6, slug: '6-gradient-descent', title: 'Gradient descent' },
  { num: 7, slug: '7-logistic-regression', title: 'Logistic regression' },
  { num: 8, slug: '8-multi-class-classification', title: 'Multi-class classification' },
  { num: 9, slug: '9-regularisation', title: 'Regularisation' },
  { num: 10, slug: '10-evaluation-metrics', title: 'Evaluation metrics' },
  { num: 11, slug: '11-cross-validation-and-tuning', title: 'Cross-validation and tuning' },
  { num: 12, slug: '12-decision-trees', title: 'Decision trees' },
  { num: 13, slug: '13-random-forests', title: 'Random forests' },
  { num: 14, slug: '14-gradient-boosting', title: 'Gradient boosting' },
  { num: 15, slug: '15-support-vector-machines', title: 'Support vector machines' },
  { num: 16, slug: '16-naive-bayes', title: 'Naive Bayes' },
  { num: 17, slug: '17-feature-engineering', title: 'Feature engineering' },
  { num: 18, slug: '18-k-means', title: 'k-means clustering' },
  { num: 19, slug: '19-hierarchical-density-clustering', title: 'Hierarchical and density-based clustering' },
  { num: 20, slug: '20-pca-dimensionality-reduction', title: 'PCA and dimensionality reduction' },
  { num: 21, slug: '21-perceptron-and-mlps', title: 'The perceptron and multilayer perceptrons' },
  { num: 22, slug: '22-backpropagation-and-training', title: 'Backpropagation and training' },
]

export function chapterHref(slug: string): string {
  return `/chapters/${slug}`
}

/** Find a chapter by its folder slug. */
export function chapterBySlug(slug: string): ChapterMeta | undefined {
  return chapters.find((c) => c.slug === slug)
}

/** The chapters immediately before and after the given slug, if any. */
export function adjacentChapters(slug: string): {
  prev?: ChapterMeta
  next?: ChapterMeta
} {
  const i = chapters.findIndex((c) => c.slug === slug)
  if (i === -1) return {}
  return {
    prev: i > 0 ? chapters[i - 1] : undefined,
    next: i < chapters.length - 1 ? chapters[i + 1] : undefined,
  }
}
