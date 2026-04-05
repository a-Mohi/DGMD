import { useState, useRef, useCallback } from "react";
import { BASE_URL } from "../App";

export default function UploadPanel({ uploadedImage, setUploadedImage }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const inputRef = useRef(null);

  const uploadFile = useCallback(
    async (file) => {
      if (!file) return;
      setError("");
      setSuccess("");
      setUploading(true);
      try {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch(`${BASE_URL}/upload-image`, {
          method: "POST",
          body: formData,
        });
        if (!res.ok) throw new Error(`Server responded ${res.status}`);
        setSuccess(`Uploaded: ${file.name}`);
        setUploadedImage(URL.createObjectURL(file));
      } catch (e) {
        setError(e.message || "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [setUploadedImage]
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) uploadFile(file);
    },
    [uploadFile]
  );

  const onFileChange = useCallback(
    (e) => {
      const file = e.target.files?.[0];
      if (file) uploadFile(file);
    },
    [uploadFile]
  );

  return (
    <div className="flex flex-col gap-3">
      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`relative flex flex-col items-center justify-center h-36 border-2 border-dashed rounded cursor-pointer transition-all
          ${
            dragging
              ? "border-green-400 bg-green-900/20"
              : "border-[#2a2f3a] hover:border-gray-500 bg-[#111318]"
          }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={onFileChange}
        />
        {uploading ? (
          <div className="flex flex-col items-center gap-2">
            <div className="w-5 h-5 border-2 border-green-400 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs text-gray-400">Uploading…</span>
          </div>
        ) : uploadedImage ? (
          <img
            src={uploadedImage}
            alt="Preview"
            className="max-h-32 max-w-full rounded object-contain"
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-gray-500">
            <svg
              className="w-8 h-8"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <span className="text-xs">Drop image or click to upload</span>
          </div>
        )}
      </div>

      {error && <p className="text-red-400 text-xs">⚠ {error}</p>}
      {success && <p className="text-green-400 text-xs">✓ {success}</p>}
    </div>
  );
}
