import { useMemo } from "react";

/**
 * GridDisplay
 * gridDim: { x: cols, y: rows }
 * status:  { x: currentCol, y: currentRow, run_complete: bool }
 *
 * Cells are scanned left-to-right, top-to-bottom.
 * A cell is "done" if it comes before the current (x,y) in scan order.
 * The cell AT (x,y) is "current" (yellow).
 */
export default function GridDisplay({ gridDim, status, runComplete }) {
  const { x: cols, y: rows } = gridDim;
  const { x: curCol, y: curRow, run_complete } = status;

  const cells = useMemo(() => {
    if (!cols || !rows) return [];
    const currentIndex = curRow * cols + curCol;
    const result = [];
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const idx = row * cols + col;
        let state = "idle";
        if (run_complete) {
          state = "done";
        } else if (idx < currentIndex) {
          state = "done";
        } else if (idx === currentIndex) {
          state = "current";
        }
        result.push({ row, col, idx, state });
      }
    }
    return result;
  }, [cols, rows, curCol, curRow, run_complete]);

  if (!cols || !rows) {
    return (
      <div className="flex items-center justify-center h-40 text-gray-600 text-xs">
        No grid data — start detection to load dimensions.
      </div>
    );
  }

  const cellSize = Math.min(
    Math.floor((520 - (cols - 1) * 2) / cols),
    Math.floor((360 - (rows - 1) * 2) / rows),
    48
  );

  return (
    <div className="overflow-auto">
      <div
        className="inline-grid gap-[2px]"
        style={{ gridTemplateColumns: `repeat(${cols}, ${cellSize}px)` }}
      >
        {cells.map((cell) => (
          <div
            key={cell.idx}
            title={`patch (${cell.col},${cell.row})`}
            style={{ width: cellSize, height: cellSize }}
            className={`rounded-sm transition-colors duration-200 border border-[#1a1a1a] ${
              cell.state === "done"
                ? "bg-green-600 border-green-500"
                : cell.state === "current"
                ? "bg-yellow-400 border-yellow-300 animate-pulse"
                : "bg-[#1e2028]"
            }`}
          />
        ))}
      </div>
      <div className="mt-3 flex gap-4 text-[10px] text-gray-500">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 bg-[#1e2028] border border-[#1a1a1a] rounded-sm inline-block" />
          Pending
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 bg-yellow-400 rounded-sm inline-block" />
          Processing
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 bg-green-600 rounded-sm inline-block" />
          Complete
        </span>
        <span className="ml-auto">
          {cols}×{rows} grid · {cols * rows} patches
        </span>
      </div>
    </div>
  );
}
