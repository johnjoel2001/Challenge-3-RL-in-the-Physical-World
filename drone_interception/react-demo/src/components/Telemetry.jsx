import React from 'react';

export default function Telemetry({ frame }) {
  if (!frame) {
    return (
      <div className="telemetry-row">
        {[['STEP', '0', ''], ['RANGE', '—', 'red'], ['BEARING', '—', ''],
          ['YOLO', '0%', ''], ['PHASE', 'STANDBY', 'yellow'], ['COST', '$350', 'green']].map(([l, v, c]) => (
          <div className="telem-box" key={l}>
            <div className="telem-label">{l}</div>
            <div className={`telem-val ${c}`}>{v}</div>
          </div>
        ))}
      </div>
    );
  }

  const pc = frame.pn === 4 ? 'green' : frame.pn >= 2 ? 'yellow' : '';

  return (
    <div className="telemetry-row">
      <div className="telem-box"><div className="telem-label">STEP</div><div className="telem-val">{frame.step}</div></div>
      <div className="telem-box"><div className="telem-label">RANGE</div><div className={`telem-val red`}>{frame.distKm}km</div></div>
      <div className="telem-box"><div className="telem-label">BEARING</div><div className="telem-val">{frame.bearing}°</div></div>
      <div className="telem-box"><div className="telem-label">YOLO</div><div className={`telem-val ${pc}`}>{(frame.yc * 100).toFixed(0)}%</div></div>
      <div className="telem-box"><div className="telem-label">PHASE</div><div className={`telem-val ${pc}`}>{frame.phase}</div></div>
      <div className="telem-box"><div className="telem-label">COST</div><div className="telem-val green">$350</div></div>
    </div>
  );
}
