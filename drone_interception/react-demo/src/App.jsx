import React, { useState, useRef, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import Telemetry from './components/Telemetry';
import DomainRand from './components/DomainRand';
import MapView from './components/MapView';
import YoloPanel from './components/YoloPanel';
import DetectionLog from './components/DetectionLog';
import Pipeline from './components/Pipeline';
import InterceptionScene from './components/InterceptionScene';
import { generateScenario, generateDomainRand, AIRBASES } from './scenario';

const API_URL = import.meta.env.VITE_API_URL || '';

const SPEED_DELAY = { SLOW: 220, NORMAL: 80, FAST: 20 };

export default function App() {
  const [baseName, setBaseName] = useState(Object.keys(AIRBASES)[0]);
  const [seed, setSeed] = useState(60);
  const [speed, setSpeed] = useState('FAST');
  const [drEnabled, setDrEnabled] = useState(true);
  const [running, setRunning] = useState(false);

  // Mission state
  const [scenario, setScenario] = useState(null);
  const [currentIdx, setCurrentIdx] = useState(-1);
  const [logEntries, setLogEntries] = useState([]);
  const [banner, setBanner] = useState(null);
  const [drParams, setDrParams] = useState(null);
  const [show3D, setShow3D] = useState(false);

  const cancelRef = useRef(false);

  const handleLaunch = useCallback(async () => {
    cancelRef.current = false;
    setRunning(true);
    setLogEntries([]);
    setCurrentIdx(-1);
    setBanner({ type: 'alert', text: 'ADVERSARY DRONE LAUNCHED FROM IRAN' });

    let sc;
    let dp;

    // Try the real PPO backend first, fall back to local JS generation
    try {
      const res = await fetch(`${API_URL}/api/scenario`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ baseName, seed, drEnabled }),
      });
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      const data = await res.json();
      sc = { frames: data.frames, iranLabel: data.iranLabel, iranLat: data.iranLat, iranLon: data.iranLon };
      dp = data.drParams || (drEnabled ? generateDomainRand(seed) : null);
      console.log(`PPO backend: ${data.totalSteps} env steps, intercepted=${data.intercepted}`);
    } catch (err) {
      console.warn('PPO backend unavailable, using local JS fallback:', err.message);
      sc = generateScenario(baseName, seed);
      dp = drEnabled ? generateDomainRand(seed) : null;
    }

    setScenario(sc);
    setDrParams(dp);

    const delay = SPEED_DELAY[speed] || 80;
    const { frames } = sc;

    let i = 0;
    const step = () => {
      if (cancelRef.current || i >= frames.length) {
        setRunning(false);
        return;
      }

      const f = frames[i];
      setCurrentIdx(i);
      setLogEntries(prev => [...prev, f]);

      // Banner
      const banners = {
        1: { type: 'standby', text: 'ADVERSARY CROSSING PERSIAN GULF \u2014 UNDETECTED' },
        2: { type: 'alert', text: 'RF SIGNAL DETECTED \u2014 CAMERA SLEWING TO BEARING' },
        3: { type: 'alert', text: 'YOLO CONFIRMED UAV \u2014 INTERCEPTOR PURSUING' },
        4: { type: 'success', text: 'KILL CONFIRMED \u2014 TARGET NEUTRALIZED' },
      };
      setBanner(banners[f.pn]);

      i++;
      setTimeout(step, delay);
    };

    // Start after a brief delay to let state settle
    setTimeout(step, 100);
  }, [baseName, seed, speed, drEnabled]);

  const frame = scenario && currentIdx >= 0 ? scenario.frames[currentIdx] : null;

  return (
    <div className="app">
      <Sidebar
        baseName={baseName} setBaseName={setBaseName}
        seed={seed} setSeed={setSeed}
        speed={speed} setSpeed={setSpeed}
        drEnabled={drEnabled} setDrEnabled={setDrEnabled}
        onLaunch={handleLaunch}
        running={running}
      />

      <div className="main-content">
        {/* Header */}
        <div className="header-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span>COUNTER-UAS COMMAND CENTER</span>
          <button
            onClick={() => setShow3D(true)}
            style={{
              background: 'linear-gradient(135deg, #00cc66, #009944)',
              color: '#fff',
              border: 'none',
              padding: '6px 14px',
              cursor: 'pointer',
              fontFamily: 'monospace',
              fontWeight: 'bold',
              fontSize: 11,
              letterSpacing: 1,
              borderRadius: 3,
            }}
          >
            3D INTERCEPTION VIEW
          </button>
        </div>

        {/* Scenario line */}
        {scenario ? (
          <div className="header-scenario">
            SCENARIO: {scenario.iranLabel.toUpperCase()}, IRAN &rarr; {baseName.toUpperCase()}
          </div>
        ) : (
          <div className="header-scenario">
            [ {baseName.toUpperCase()} ] — MONITORING IRANIAN AIRSPACE
          </div>
        )}

        {/* Domain Randomization row */}
        {drEnabled && (
          <DomainRand params={drParams || generateDomainRand(seed)} seed={seed} />
        )}

        {/* Banner */}
        {banner ? (
          <div className={`banner banner-${banner.type}`}>{banner.text}</div>
        ) : (
          <div className="banner banner-standby">SECTOR DEFENSE STANDBY // AWAITING LAUNCH COMMAND</div>
        )}

        {/* Telemetry */}
        <Telemetry frame={frame} />

        {/* Map + Right panel */}
        <div className="content-grid">
          <MapView
            baseName={baseName}
            frames={scenario ? scenario.frames : []}
            currentIdx={currentIdx}
            iranLabel={scenario ? scenario.iranLabel : ''}
            iranLat={scenario ? scenario.iranLat : null}
            iranLon={scenario ? scenario.iranLon : null}
          />

          <div className="right-panel">
            <div className="section-header">
              YOLO VISUAL ID <span className="highlight">// RF detects signal, YOLO confirms it's a drone</span>
            </div>
            {frame ? (
              <YoloPanel frame={frame} />
            ) : (
              <div className="camera-offline">
                <div className="camera-offline-icon">&#9678;</div>
                <div className="camera-offline-text">CAMERA OFFLINE</div>
                <div className="camera-offline-sub">
                  PTZ camera activates when RF detects a signal.<br />
                  YOLO then classifies the object as UAV / bird / aircraft.
                </div>
              </div>
            )}

            <div className="section-header">DETECTION LOG</div>
            <DetectionLog entries={logEntries} />
          </div>
        </div>

        {/* Pipeline */}
        <Pipeline activeStage={frame ? frame.pn : 0} />
      </div>

      {/* 3D Interception Modal */}
      <InterceptionScene
        visible={show3D}
        onClose={() => setShow3D(false)}
        frames={scenario ? scenario.frames : []}
        currentIdx={currentIdx}
        baseName={baseName}
        iranLabel={scenario ? scenario.iranLabel : ''}
        iranLat={scenario ? scenario.iranLat : null}
        iranLon={scenario ? scenario.iranLon : null}
        speed={speed}
      />
    </div>
  );
}
