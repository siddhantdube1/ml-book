import type { LossFn } from './gradient'

export type ContourSet = {
  level: number
  /** Each segment is [x1, y1, x2, y2] in domain coordinates. */
  segments: Array<[number, number, number, number]>
}

/**
 * Marching-squares contour generation for a 2D scalar field.
 *
 * Samples the loss on a (resolution + 1) × (resolution + 1) grid, then for
 * each cell checks which corners exceed the contour level and emits the
 * appropriate line segments. The corner-encoding convention here:
 *
 *   bit 0 = bottom-left, 1 = bottom-right, 2 = top-right, 3 = top-left
 *
 * Saddle cases (5 and 10) are resolved consistently to one orientation;
 * the alternative would split the saddle the other way. Either is correct.
 */
export function computeContours(
  loss: LossFn,
  levels: number[],
  resolution = 64,
): ContourSet[] {
  const [[xMin, yMin], [xMax, yMax]] = loss.domain
  const dx = (xMax - xMin) / resolution
  const dy = (yMax - yMin) / resolution

  // Pre-sample the field. grid[i][j] is the value at (xMin + i*dx, yMin + j*dy).
  const grid: number[][] = []
  for (let i = 0; i <= resolution; i++) {
    const row: number[] = new Array(resolution + 1)
    for (let j = 0; j <= resolution; j++) {
      row[j] = loss.eval([xMin + i * dx, yMin + j * dy])
    }
    grid.push(row)
  }

  return levels.map((level) => {
    const segments: ContourSet['segments'] = []

    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const vBL = grid[i][j]
        const vBR = grid[i + 1][j]
        const vTR = grid[i + 1][j + 1]
        const vTL = grid[i][j + 1]

        let idx = 0
        if (vBL > level) idx |= 1
        if (vBR > level) idx |= 2
        if (vTR > level) idx |= 4
        if (vTL > level) idx |= 8

        if (idx === 0 || idx === 15) continue

        const xL = xMin + i * dx
        const xR = xMin + (i + 1) * dx
        const yB = yMin + j * dy
        const yT = yMin + (j + 1) * dy

        // Edge crossings via linear interpolation.
        const eBottom = (): [number, number] => {
          const t = (level - vBL) / (vBR - vBL)
          return [xL + t * dx, yB]
        }
        const eRight = (): [number, number] => {
          const t = (level - vBR) / (vTR - vBR)
          return [xR, yB + t * dy]
        }
        const eTop = (): [number, number] => {
          const t = (level - vTL) / (vTR - vTL)
          return [xL + t * dx, yT]
        }
        const eLeft = (): [number, number] => {
          const t = (level - vBL) / (vTL - vBL)
          return [xL, yB + t * dy]
        }

        const push = (a: [number, number], b: [number, number]) => {
          segments.push([a[0], a[1], b[0], b[1]])
        }

        switch (idx) {
          case 1:
          case 14:
            push(eBottom(), eLeft())
            break
          case 2:
          case 13:
            push(eBottom(), eRight())
            break
          case 3:
          case 12:
            push(eLeft(), eRight())
            break
          case 4:
          case 11:
            push(eTop(), eRight())
            break
          case 6:
          case 9:
            push(eBottom(), eTop())
            break
          case 7:
          case 8:
            push(eTop(), eLeft())
            break
          case 5:
            // Saddle: BL + TR above. Connect bottom→right and top→left.
            push(eBottom(), eRight())
            push(eTop(), eLeft())
            break
          case 10:
            // Saddle: BR + TL above. Connect bottom→left and top→right.
            push(eBottom(), eLeft())
            push(eTop(), eRight())
            break
        }
      }
    }

    return { level, segments }
  })
}
