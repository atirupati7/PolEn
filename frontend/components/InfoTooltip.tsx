import React, { useState, useRef, useEffect } from "react";

interface InfoTooltipProps {
  text: string;
  children?: React.ReactNode;
  position?: "top" | "bottom" | "left" | "right";
  className?: string;
}

export default function InfoTooltip({
  text,
  children,
  position = "top",
  className = "",
}: InfoTooltipProps) {
  const [visible, setVisible] = useState(false);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const positionClasses = {
    top: "bottom-full left-1/2 -translate-x-1/2 mb-2",
    bottom: "top-full left-1/2 -translate-x-1/2 mt-2",
    left: "right-full top-1/2 -translate-y-1/2 mr-2",
    right: "left-full top-1/2 -translate-y-1/2 ml-2",
  };

  const arrowClasses = {
    top: "top-full left-1/2 -translate-x-1/2 border-t-slate-700 border-l-transparent border-r-transparent border-b-transparent",
    bottom: "bottom-full left-1/2 -translate-x-1/2 border-b-slate-700 border-l-transparent border-r-transparent border-t-transparent",
    left: "left-full top-1/2 -translate-y-1/2 border-l-slate-700 border-t-transparent border-b-transparent border-r-transparent",
    right: "right-full top-1/2 -translate-y-1/2 border-r-slate-700 border-t-transparent border-b-transparent border-l-transparent",
  };

  return (
    <span
      className={`relative inline-flex items-center ${className}`}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children || (
        <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-slate-700 hover:bg-indigo-600 text-[9px] font-bold text-slate-300 hover:text-white cursor-help transition-all duration-200 select-none">
          ?
        </span>
      )}
      {visible && (
        <div
          ref={tooltipRef}
          className={`absolute z-50 ${positionClasses[position]} pointer-events-none`}
        >
          <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs text-slate-200 shadow-xl shadow-black/30 max-w-[280px] min-w-[160px] leading-relaxed whitespace-normal backdrop-blur-sm">
            {text}
          </div>
          <div
            className={`absolute w-0 h-0 border-[5px] ${arrowClasses[position]}`}
          />
        </div>
      )}
    </span>
  );
}
