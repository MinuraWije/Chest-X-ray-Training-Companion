import { useState } from "react";
import type { ReactNode } from "react";

interface CxrViewerProps {
  src: string;
  alt: string;
}

export function CxrViewer({ src, alt }: CxrViewerProps): ReactNode {
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [imgError, setImgError] = useState<boolean>(false);

  return (
    <div className="card overflow-hidden">
      <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50 px-3 py-2 text-sm">
        <span className="text-slate-700">Image viewer</span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="btn-ghost px-2 py-1"
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
            aria-label="Zoom out"
          >
            −
          </button>
          <span className="w-12 text-center tabular-nums">
            {Math.round(zoom * 100)}%
          </span>
          <button
            type="button"
            className="btn-ghost px-2 py-1"
            onClick={() => setZoom((z) => Math.min(4, z + 0.25))}
            aria-label="Zoom in"
          >
            +
          </button>
          <button
            type="button"
            className="btn-ghost px-2 py-1"
            onClick={() => {
              setZoom(1);
              setPan({ x: 0, y: 0 });
            }}
          >
            Reset
          </button>
        </div>
      </div>
      <div className="flex h-[480px] items-center justify-center overflow-hidden bg-slate-900">
        {imgError ? (
          <div className="px-4 text-center text-sm text-slate-300">
            <p className="font-semibold text-slate-100">Image unavailable</p>
            <p className="mt-1 text-xs text-slate-400">
              Could not load <code className="text-slate-300">{src}</code>.
            </p>
            <p className="mt-1 text-xs text-slate-400">
              Make sure the backend is running on{" "}
              <code className="text-slate-300">127.0.0.1:8000</code>.
            </p>
          </div>
        ) : (
          <img
            src={src}
            alt={alt}
            draggable={false}
            onError={() => setImgError(true)}
            onLoad={() => setImgError(false)}
            className="max-h-full max-w-full object-contain transition-transform"
            style={{
              transform: `scale(${zoom}) translate(${pan.x / 100}px, ${pan.y / 100}px)`,
            }}
          />
        )}
      </div>
    </div>
  );
}
