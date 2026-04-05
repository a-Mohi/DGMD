import { useState, useEffect } from "react";
import { BASE_URL } from "../App";

export default function StatsPanel({ selectedModel, runComplete }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!runComplete) return;
    setLoading(true);
    fetch(`${BASE_URL}/get-run-stats`)
      .then((r) => r.json())
      .then((data) => {
        setStats(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [runComplete]);

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Model info — always shown if model selected */}
      {selectedModel && (
        <div className="border border-[#1f2937] rounded p-3 bg-[#0f1117]">
          <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">
            Active Model
          </div>
          <StatRow label="Name" value={selectedModel.name} highlight />
          {selectedModel.id && (
            <StatRow label="ID" value={String(selectedModel.id)} />
          )}
          {selectedModel.params && (
            <StatRow label="Params" value={selectedModel.params} />
          )}
          {/* Render any extra model fields */}
          {Object.entries(selectedModel)
            .filter(([k]) => !["name", "id", "params"].includes(k))
            .map(([k, v]) => (
              <StatRow key={k} label={k} value={String(v)} />
            ))}
        </div>
      )}

      {/* Run stats */}
      {!runComplete && !loading && (
        <div className="flex items-center justify-center flex-1 text-gray-600 text-xs">
          Stats appear after detection completes.
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center flex-1 gap-2 text-gray-500 text-xs">
          <div className="w-3 h-3 border border-green-400 border-t-transparent rounded-full animate-spin" />
          Loading stats…
        </div>
      )}

      {stats && !loading && (
        <div className="border border-[#1f2937] rounded p-3 bg-[#0f1117] overflow-y-auto flex-1">
          <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">
            Run Statistics
          </div>
          {Object.entries(stats).map(([k, v]) => (
            <StatRow key={k} label={k} value={String(v)} />
          ))}
        </div>
      )}
    </div>
  );
}

function StatRow({ label, value, highlight }) {
  return (
    <div className="flex justify-between items-start py-1 border-b border-[#1a1f2a] last:border-0 gap-4">
      <span className="text-[10px] text-gray-500 uppercase tracking-wider shrink-0">
        {label}
      </span>
      <span
        className={`text-xs text-right break-all ${
          highlight ? "text-green-400 font-semibold" : "text-gray-300"
        }`}
      >
        {value}
      </span>
    </div>
  );
}
