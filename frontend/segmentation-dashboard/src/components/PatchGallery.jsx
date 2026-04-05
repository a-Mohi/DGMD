import { useState, useEffect } from "react";
import { BASE_URL } from "../App";

export default function PatchGallery({ patchCount, runComplete }) {
  const [patches, setPatches] = useState([]);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    if (!runComplete || patchCount === 0) {
      setPatches([]);
      return;
    }
    // Build array of patch URLs
    const urls = Array.from(
      { length: patchCount },
      (_, i) => `${BASE_URL}/get-patch?id=${i}`
    );
    setPatches(urls);
  }, [runComplete, patchCount]);

  if (!runComplete) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-600 text-xs">
        {patches.length === 0
          ? "Patches will appear after detection completes."
          : ""}
      </div>
    );
  }

  if (patchCount === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-600 text-xs">
        No detected patches found.
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-4 gap-2 overflow-y-auto max-h-52 pr-1">
        {patches.map((url, i) => (
          <div
            key={i}
            onClick={() => setSelected(url)}
            className="cursor-pointer rounded border border-[#2a2f3a] overflow-hidden hover:border-green-500 transition-colors group"
          >
            <img
              src={url}
              alt={`Patch ${i}`}
              className="w-full h-14 object-cover group-hover:opacity-80 transition-opacity"
              loading="lazy"
            />
            <div className="text-[9px] text-center text-gray-600 py-0.5">
              #{i}
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox */}
      {selected && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center z-50"
          onClick={() => setSelected(null)}
        >
          <div
            className="relative max-w-2xl max-h-[80vh] bg-[#111318] p-2 rounded-lg border border-[#2a2f3a]"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setSelected(null)}
              className="absolute top-2 right-2 text-gray-400 hover:text-white text-lg leading-none"
            >
              ×
            </button>
            <img
              src={selected}
              alt="Patch enlarged"
              className="max-w-full max-h-[75vh] rounded object-contain"
            />
          </div>
        </div>
      )}

      <div className="mt-2 text-[10px] text-gray-500">
        {patchCount} patch{patchCount !== 1 ? "es" : ""} detected · click to enlarge
      </div>
    </>
  );
}
