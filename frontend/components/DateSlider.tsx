/**
 * DateSlider — Full-width interactive date slider with playback controls.
 *
 * Features: range slider, step ◀▶ buttons, play/pause auto-advance,
 * regime badge, "Latest" reset button.
 */

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { fmtDateLong, regimeBg } from "../lib/metrics";

interface DateSliderProps {
  dates: string[];
  selectedDate: string | null;
  onDateChange: (date: string | null) => void;
  regimeAtDate?: string;
}

export default function DateSlider({
  dates,
  selectedDate,
  onDateChange,
  regimeAtDate,
}: DateSliderProps) {
  const [playing, setPlaying] = useState(false);

  const currentIndex = useMemo(() => {
    if (!selectedDate) return dates.length - 1;
    const idx = dates.indexOf(selectedDate);
    return idx >= 0 ? idx : dates.length - 1;
  }, [selectedDate, dates]);

  /* Auto-advance when playing */
  useEffect(() => {
    if (!playing || currentIndex >= dates.length - 1) {
      if (playing && currentIndex >= dates.length - 1) setPlaying(false);
      return;
    }
    const timer = setTimeout(() => onDateChange(dates[currentIndex + 1]), 300);
    return () => clearTimeout(timer);
  }, [playing, currentIndex, dates, onDateChange]);

  const step = useCallback(
    (dir: -1 | 1) => {
      const next = currentIndex + dir;
      if (next >= 0 && next < dates.length) onDateChange(dates[next]);
    },
    [currentIndex, dates, onDateChange],
  );

  if (dates.length === 0) return null;

  const isLatest = !selectedDate;

  return (
    <div className="bg-slate-900/60 border-b border-slate-700/30 px-6 py-2 flex items-center gap-3 flex-shrink-0">
      <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider whitespace-nowrap">
        Date
      </span>

      {/* Step back */}
      <button
        onClick={() => step(-1)}
        disabled={currentIndex <= 0}
        className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 disabled:opacity-30 transition-colors"
      >
        ◀
      </button>

      {/* Play/Pause */}
      <button
        onClick={() => setPlaying(!playing)}
        className={`p-1 rounded text-xs transition-colors ${
          playing
            ? "bg-amber-600/60 text-amber-200"
            : "bg-slate-800 hover:bg-slate-700 text-slate-300"
        }`}
      >
        {playing ? "⏸" : "▶"}
      </button>

      {/* Range slider */}
      <div className="flex-1 flex items-center gap-2">
        <span className="text-[9px] text-slate-600 whitespace-nowrap">
          {dates[0]?.slice(0, 7)}
        </span>
        <input
          type="range"
          min={0}
          max={dates.length - 1}
          value={currentIndex}
          onChange={(e) => onDateChange(dates[parseInt(e.target.value)])}
          className="flex-1 accent-indigo-500 h-1 cursor-pointer"
        />
        <span className="text-[9px] text-slate-600 whitespace-nowrap">
          {dates[dates.length - 1]?.slice(0, 7)}
        </span>
      </div>

      {/* Step forward */}
      <button
        onClick={() => step(1)}
        disabled={currentIndex >= dates.length - 1}
        className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 disabled:opacity-30 transition-colors"
      >
        ▶
      </button>

      {/* Current date display */}
      <div className="flex items-center gap-2 whitespace-nowrap">
        <span className="font-mono text-sm text-cyan-400">
          {fmtDateLong(dates[currentIndex])}
        </span>
        {regimeAtDate && (
          <span
            className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${regimeBg(regimeAtDate)}`}
          >
            {regimeAtDate}
          </span>
        )}
      </div>

      {/* Latest button */}
      <button
        onClick={() => onDateChange(null)}
        className={`text-[10px] px-2 py-1 rounded transition-colors whitespace-nowrap ${
          isLatest
            ? "bg-indigo-600/40 text-indigo-300 border border-indigo-500/30"
            : "bg-slate-800 hover:bg-slate-700 text-slate-400 border border-slate-700/30"
        }`}
      >
        Latest
      </button>
    </div>
  );
}
