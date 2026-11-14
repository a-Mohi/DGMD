import { useState, useEffect, useRef, useCallback } from "react";
import GridDisplay from "./components/GridDisplay";
import UploadPanel from "./components/UploadPanel";
import ModelSelector from "./components/ModelSelector";
import PatchGallery from "./components/PatchGallery";
import StatsPanel from "./components/StatsPanel";

export const BASE_URL = "http://localhost:8000";

export default function App() {
  const [gridDim, setGridDim] = useState({ x: 0, y: 0 });
  const [status, setStatus] = useState({ x: 0, y: 0, run_complete: false });
  const [runComplete, setRunComplete] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [detectionError, setDetectionError] = useState("");
  const [detectionStarted, setDetectionStarted] = useState(false);
  const [patchCount, setPatchCount] = useState(0);
  const gridPollRef = useRef(null);

  // Fetch grid dimensions once on mount
  useEffect(() => {
    fetch(`${BASE_URL}/get-progress-dim`)
      .then((r) => r.json())
      .then((data) => setGridDim({ x: data.x, y: data.y }))
      .catch(() => { });
  }, []);

  // Called by UploadPanel once the server confirms the upload.
  // Polls /get-progress-dim every 300ms until the backend returns a
  // non-zero grid (the sampler computes it right after saving the file).
  const handleUploadSuccess = useCallback(() => {
    // Clear any previous poll
    if (gridPollRef.current) clearInterval(gridPollRef.current);

    // Reset stale grid while we wait
    setGridDim({ x: 0, y: 0 });

    gridPollRef.current = setInterval(() => {
      fetch(`${BASE_URL}/get-progress-dim`)
        .then((r) => r.json())
        .then((data) => {
          if (data.x > 0 && data.y > 0) {
            setGridDim({ x: data.x, y: data.y });
            clearInterval(gridPollRef.current);
            gridPollRef.current = null;
          }
        })
        .catch(() => { });
    }, 300);
  }, []);

  // Clean up grid poll on unmount
  useEffect(() => () => {
    if (gridPollRef.current) clearInterval(gridPollRef.current);
  }, []);

  // Poll /get-status every 500ms while detection is running
  useEffect(() => {
    if (!detectionStarted || runComplete) return;
    const interval = setInterval(() => {
      fetch(`${BASE_URL}/get-status`)
        .then((r) => r.json())
        .then((data) => {
          setStatus({ x: data.x, y: data.y, run_complete: data.run_complete });
          if (data.run_complete) {
            setRunComplete(true);
            clearInterval(interval);
            // Fetch patch count after completion
            fetch(`${BASE_URL}/get-patch-count`)
              .then((r) => r.json())
              .then((d) => setPatchCount(d.count ?? d.patch_count ?? 0))
              .catch(() => { });
          }
        })
        .catch(() => { });
    }, 500);
    return () => clearInterval(interval);
  }, [detectionStarted, runComplete]);

  const handleStart = useCallback(async () => {
    setDetectionError("");
    try {
      const res = await fetch(`${BASE_URL}/start`);
      if (res.status === 400) {
        setDetectionError("Upload image and select model first");
        return;
      }
      if (!res.ok) {
        setDetectionError("Failed to start detection.");
        return;
      }
      setRunComplete(false);
      setStatus({ x: 0, y: 0, run_complete: false });
      setPatchCount(0);
      setDetectionStarted(true);
    } catch {
      setDetectionError("Could not reach backend.");
    }
  }, []);

  return (
    <div className="min-h-screen bg-[#0d0d0f] text-gray-100 font-mono p-4">
      {/* Header */}
      <header className="mb-4 flex items-center gap-3 border-b border-[#1f2937] pb-3">
        <div className="flex gap-1">
          <span className="w-2 h-2 rounded-full bg-red-500" />
          <span className="w-2 h-2 rounded-full bg-yellow-500" />
          <span className="w-2 h-2 rounded-full bg-green-500" />
        </div>
        <h1 className="text-xs tracking-[0.25em] uppercase text-gray-400">
          Mitosis · Detection · Dashboard
        </h1>
        <div className="ml-auto flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${detectionStarted && !runComplete
              ? "bg-yellow-400 animate-pulse"
              : runComplete
                ? "bg-green-400"
                : "bg-gray-600"
              }`}
          />
          <span className="text-[10px] text-gray-500 uppercase tracking-widest">
            {detectionStarted && !runComplete
              ? "Running"
              : runComplete
                ? "Complete"
                : "Idle"}
          </span>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-2 gap-4">
        {/* TOP LEFT: Detection Grid */}
        <div className="panel">
          <PanelLabel>Detection Grid</PanelLabel>
          <GridDisplay
            gridDim={gridDim}
            status={status}
            runComplete={runComplete}
          />
        </div>

        {/* TOP RIGHT: Upload + Controls */}
        <div className="flex flex-col gap-4">
          <div className="panel">
            <PanelLabel>Upload Image</PanelLabel>
            <UploadPanel
              uploadedImage={uploadedImage}
              setUploadedImage={setUploadedImage}
              onUploadSuccess={handleUploadSuccess}
            />
          </div>

          {/* MIDDLE RIGHT: Model + Start */}
          <div className="panel">
            <PanelLabel>Model & Controls</PanelLabel>
            <ModelSelector
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
            />
            <div className="mt-4 flex flex-col gap-2">
              <button
                onClick={handleStart}
                className="w-full py-2 px-4 bg-green-700 hover:bg-green-600 active:scale-95 transition-all text-xs tracking-widest uppercase font-bold border border-green-500 rounded"
              >
                ▶ Start Detection
              </button>
              {detectionError && (
                <p className="text-red-400 text-xs text-center">
                  ⚠ {detectionError}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* BOTTOM LEFT: Patch Gallery */}
        <div className="panel min-h-[240px]">
          <PanelLabel>Detected Patch Gallery</PanelLabel>
          <PatchGallery patchCount={patchCount} runComplete={runComplete} />
        </div>

        {/* BOTTOM RIGHT: Stats */}
        <div className="panel min-h-[240px]">
          <PanelLabel>Stats & Model Info</PanelLabel>
          <StatsPanel selectedModel={selectedModel} runComplete={runComplete} />
        </div>
      </div>
    </div>
  );
}

function PanelLabel({ children }) {
  return (
    <div className="text-[10px] text-gray-500 uppercase tracking-[0.2em] mb-3 border-b border-[#1f2937] pb-2">
      {children}
    </div>
  );
}
