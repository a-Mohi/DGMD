import { useState, useEffect } from "react";
import { BASE_URL } from "../App";

export default function ModelSelector({ selectedModel, setSelectedModel }) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [setting, setSetting] = useState(false);

  useEffect(() => {
    fetch(`${BASE_URL}/get-models`)
      .then((r) => r.json())
      .then((data) => {
        // Accept object { modelName: { id, ... } } or array
        if (Array.isArray(data)) {
          setModels(data);
        } else {
          const parsed = Object.entries(data).map(([name, val]) => ({
            name,
            id: typeof val === "object" ? val.id ?? name : val,
            ...( typeof val === "object" ? val : {} ),
          }));
          setModels(parsed);
        }
        setLoading(false);
      })
      .catch(() => {
        setError("Could not load models");
        setLoading(false);
      });
  }, []);

  const handleSelect = async (e) => {
    const modelId = e.target.value;
    const model = models.find((m) => String(m.id) === modelId);
    if (!model) return;
    setSetting(true);
    try {
      await fetch(`${BASE_URL}/set-model?model=${encodeURIComponent(modelId)}`);
      setSelectedModel(model);
    } catch {
      // still set locally
      setSelectedModel(model);
    } finally {
      setSetting(false);
    }
  };

  if (loading) {
    return (
      <div className="text-xs text-gray-500 animate-pulse">Loading models…</div>
    );
  }
  if (error) {
    return <div className="text-xs text-red-400">{error}</div>;
  }

  return (
    <div className="flex flex-col gap-2">
      <label className="text-[10px] text-gray-500 uppercase tracking-widest">
        Model
      </label>
      <div className="relative">
        <select
          onChange={handleSelect}
          defaultValue=""
          disabled={setting}
          className="w-full bg-[#111318] border border-[#2a2f3a] text-gray-200 text-xs rounded px-3 py-2 appearance-none cursor-pointer hover:border-gray-500 focus:outline-none focus:border-green-500 transition-colors"
        >
          <option value="" disabled>
            — Select a model —
          </option>
          {models.map((m) => (
            <option key={m.id} value={String(m.id)}>
              {m.name}
            </option>
          ))}
        </select>
        <div className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 text-xs">
          {setting ? "⟳" : "▾"}
        </div>
      </div>
      {selectedModel && (
        <div className="text-[10px] text-green-400">
          ✓ {selectedModel.name} selected
        </div>
      )}
    </div>
  );
}
