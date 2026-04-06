import React, { useRef, useEffect } from 'react';

function logEntry(f) {
  const { distKm: d, yc: c, pn, bearing } = f;
  if (pn === 4)
    return { cls: 'log-kill', text: '[KILL] TARGET INTERCEPTED — visual confirm via YOLO' };
  if (pn === 3 && d < 5)
    return { cls: 'log-yolo-close', text: `[YOLO] UAV conf:${(c * 100).toFixed(0)}% | ${d}km | CLOSING — RL policy active` };
  if (pn === 3)
    return { cls: 'log-yolo', text: `[YOLO] UAV conf:${(c * 100).toFixed(0)}% | ${d}km — interceptor pursuing` };
  if (pn === 2)
    return { cls: 'log-rf', text: `[RF] Signal BRG:${bearing}° | ${d}km — camera slewing` };
  return { cls: 'log-scan', text: '[SCAN] No contacts — monitoring' };
}

export default function DetectionLog({ entries }) {
  const ref = useRef(null);

  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [entries]);

  return (
    <div className="detection-log" ref={ref}>
      {(!entries || entries.length === 0) && (
        <span className="log-scan">Awaiting RF signal...</span>
      )}
      {entries && entries.slice(-15).map((f, i) => {
        const e = logEntry(f);
        return <div key={i} className={e.cls}>{e.text}</div>;
      })}
    </div>
  );
}
