"use client";

import { useState } from 'react';

export function SyncButton() {
  const [syncing, setSyncing] = useState(false);

  const handleSync = async () => {
    if (!window.confirm("Run the full model pipeline in the background? This takes about 7 minutes.")) return;
    
    setSyncing(true);
    try {
      const res = await fetch('/api/sync-data', { method: 'POST' });
      if (res.ok) {
        alert("Pipeline started! The UI will update automatically when it finishes in ~7 minutes.");
      } else {
        alert("Failed to start pipeline. Check the Python backend logs.");
      }
    } catch (err) {
      alert("Error triggering pipeline.");
    }
    setSyncing(false);
  };

  return (
    <button 
      onClick={handleSync} 
      disabled={syncing}
      className="rounded-xl bg-[var(--accent)] px-3 py-2 text-[10px] font-bold uppercase tracking-wide text-black hover:opacity-90 disabled:opacity-50 transition-opacity"
    >
      {syncing ? "Starting..." : "Sync Data"}
    </button>
  );
}