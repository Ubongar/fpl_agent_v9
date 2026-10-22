"use client";

import { useState, useEffect } from 'react';
import { pipelineStatus } from "@/lib/api";

export function SyncButton() {
  const [syncing, setSyncing] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");

  useEffect(() => {
    if (!syncing) return;

    const interval = setInterval(async () => {
      try {
        const res = await pipelineStatus();
        setStatusMsg(res.message);
        if (res.status === "idle" || res.status === "error") {
          setSyncing(false);
          setStatusMsg("");
          if (res.status === "error") alert("Pipeline failed. Check terminal.");
          else alert("Pipeline complete! Data updated.");
        }
      } catch (e) {
        // Use console.warn instead of console.error to prevent Next.js dev overlay
        console.warn("Polling status failed, retrying...");
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [syncing]);

  const handleSync = async () => {
    if (!window.confirm("Run the full model pipeline in the background? This takes about 7 minutes.")) return;
    
    setSyncing(true);
    setStatusMsg("Starting pipeline...");
    try {
      await fetch('/api/sync-data', { method: 'POST' });
    } catch (err) {
      alert("Error triggering pipeline.");
      setSyncing(false);
    }
  };

  return (
    <div className="flex flex-col items-end">
      <button 
        onClick={handleSync} 
        disabled={syncing}
        className="rounded-xl bg-[var(--accent)] px-3 py-2 text-[10px] font-bold uppercase tracking-wide text-black hover:opacity-90 disabled:opacity-50 transition-opacity"
      >
        {syncing ? "Syncing..." : "Sync Data"}
      </button>
      {syncing && <div className="mt-1 text-[9px] text-muted max-w-[200px] text-right truncate">{statusMsg}</div>}
    </div>
  );
}