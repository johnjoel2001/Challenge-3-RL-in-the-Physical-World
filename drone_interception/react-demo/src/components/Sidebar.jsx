import React from 'react';
import { AIRBASES } from '../scenario';

export default function Sidebar({
  baseName, setBaseName,
  seed, setSeed,
  speed, setSpeed,
  drEnabled, setDrEnabled,
  onLaunch, running,
}) {
  const randomSeed = () => setSeed(Math.floor(Math.random() * 9999));

  return (
    <div className="sidebar">
      <div className="sidebar-title">MISSION CONTROL</div>
      <hr className="sidebar-divider" />

      <div className="sidebar-label">AIRBASE</div>
      <select value={baseName} onChange={e => setBaseName(e.target.value)}>
        {Object.keys(AIRBASES).map(k => (
          <option key={k} value={k}>{k}</option>
        ))}
      </select>

      <div className="sidebar-label">RL ALGORITHM</div>
      <select disabled>
        <option>PPO (78% intercept)</option>
      </select>

      <div className="sidebar-label">SCENARIO SEED</div>
      <div className="seed-row">
        <input
          type="number" min={0} max={9999}
          value={seed}
          onChange={e => setSeed(+e.target.value || 0)}
        />
        <button onClick={randomSeed} title="Random seed">&#x21BB;</button>
      </div>

      <div className="sidebar-label">PLAYBACK SPEED</div>
      <div className="speed-row">
        {['SLOW', 'NORMAL', 'FAST'].map(s => (
          <button
            key={s}
            className={`speed-btn ${speed === s ? 'active' : ''}`}
            onClick={() => setSpeed(s)}
          >{s}</button>
        ))}
      </div>

      <div className="sidebar-label">SIM2REAL</div>
      <label className="dr-toggle">
        <input
          type="checkbox"
          checked={drEnabled}
          onChange={e => setDrEnabled(e.target.checked)}
        />
        Domain Randomization
      </label>

      <button
        className="launch-btn"
        onClick={onLaunch}
        disabled={running}
      >
        LAUNCH MISSION
      </button>

      <hr className="sidebar-divider" />

      <div className="sidebar-label">DETECTION CHAIN</div>
      <div className="sidebar-chain">
        1. <b>RF Sensor</b> &mdash; detects signal<br />
        2. <b>YOLO Camera</b> &mdash; confirms UAV<br />
        3. <b>RL Policy</b> &mdash; computes pursuit<br />
        4. <b>Interceptor</b> &mdash; kills target
      </div>

      <hr className="sidebar-divider" />

      <div className="sidebar-label">$ SYSTEM COST</div>
      <div className="sidebar-cost">
        <div className="cost-line"><span className="cost-label">RF Sensors</span><span className="cost-value">$5,000</span></div>
        <div className="cost-line"><span className="cost-label">PTZ + YOLO</span><span className="cost-value">$1,000</span></div>
        <div className="cost-line"><span className="cost-label">Jetson Orin</span><span className="cost-value">$100</span></div>
        <div className="cost-line"><span className="cost-label">Pursuit Drone</span><span className="cost-value">$300</span></div>
        <hr className="sidebar-divider" />
        <div className="cost-line"><span className="cost-total">Total</span><span className="cost-total">$6,400</span></div>
        <div className="cost-savings">vs Patriot: −99.999%</div>
      </div>
    </div>
  );
}
