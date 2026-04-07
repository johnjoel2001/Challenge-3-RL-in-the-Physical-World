import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Billboard, Text, Line } from '@react-three/drei';
import * as THREE from 'three';

/* ── Geo coordinate conversion ── */
function geoTo3D(lat, lon, alt, bounds) {
  const x = ((lon - bounds.minLon) / (bounds.maxLon - bounds.minLon || 1)) * 40;
  const z = ((lat - bounds.minLat) / (bounds.maxLat - bounds.minLat || 1)) * 40;
  const y = ((alt || 300) - 0) / 800 * 6;
  return [x, y, z];
}

function computeBounds(frames, iranLat, iranLon) {
  let minLat = Infinity, maxLat = -Infinity, minLon = Infinity, maxLon = -Infinity;
  for (const f of frames) {
    minLat = Math.min(minLat, f.advLat, f.intLat);
    maxLat = Math.max(maxLat, f.advLat, f.intLat);
    minLon = Math.min(minLon, f.advLon, f.intLon);
    maxLon = Math.max(maxLon, f.advLon, f.intLon);
  }
  if (iranLat != null && iranLon != null) {
    minLat = Math.min(minLat, iranLat); maxLat = Math.max(maxLat, iranLat);
    minLon = Math.min(minLon, iranLon); maxLon = Math.max(maxLon, iranLon);
  }
  if (frames.length > 0) {
    minLat = Math.min(minLat, frames[0].intLat); maxLat = Math.max(maxLat, frames[0].intLat);
    minLon = Math.min(minLon, frames[0].intLon); maxLon = Math.max(maxLon, frames[0].intLon);
  }
  const padLat = (maxLat - minLat) * 0.1 || 0.5;
  const padLon = (maxLon - minLon) * 0.1 || 0.5;
  return { minLat: minLat - padLat, maxLat: maxLat + padLat, minLon: minLon - padLon, maxLon: maxLon + padLon };
}

/* ── Drone quadcopter ── */
function Drone({ position, color, label, rotorSpeed = 12, scale = 1, showNet = false, netDeploy = 0 }) {
  const rotorRefs = [useRef(), useRef(), useRef(), useRef()];
  useFrame((_, dt) => rotorRefs.forEach(r => { if (r.current) r.current.rotation.y += rotorSpeed * dt; }));
  const arms = [[0.4,0,0.4],[-0.4,0,0.4],[0.4,0,-0.4],[-0.4,0,-0.4]];
  const netHang = 0.3 + netDeploy * 1.2;
  const netRadius = 0.15 + netDeploy * 0.6;
  return (
    <group position={position} scale={[scale, scale, scale]}>
      <mesh><boxGeometry args={[0.5, 0.15, 0.5]} /><meshStandardMaterial color={color} metalness={0.6} roughness={0.3} /></mesh>
      {arms.map((a, i) => (
        <group key={i} position={a}>
          <mesh><cylinderGeometry args={[0.03, 0.03, 0.6, 6]} /><meshStandardMaterial color="#444" /></mesh>
          <mesh position={[0,0.1,0]}><cylinderGeometry args={[0.07,0.07,0.06,8]} /><meshStandardMaterial color="#222" metalness={0.8} /></mesh>
          <mesh ref={rotorRefs[i]} position={[0,0.14,0]}><cylinderGeometry args={[0.22,0.22,0.015,12]} /><meshStandardMaterial color="#aaa" transparent opacity={0.35} /></mesh>
        </group>
      ))}
      <Billboard><Text fontSize={0.25} color={color} anchorY="bottom" position={[0,0.7,0]} font={undefined}>{label}</Text></Billboard>
      {showNet && (
        <group position={[0, -0.1, 0]}>
          {[[0.2,0,0.2],[-0.2,0,0.2],[0.2,0,-0.2],[-0.2,0,-0.2]].map((corner, i) => (
            <Line key={i} points={[corner, [0, -netHang, 0]]} color="#f5f5dc" lineWidth={1} transparent opacity={0.7} />
          ))}
          <mesh position={[0, -netHang * 0.5, 0]}><coneGeometry args={[netRadius, netHang, 8, 1, true]} /><meshStandardMaterial color="#f5f5dc" wireframe transparent opacity={0.6} side={THREE.DoubleSide} /></mesh>
          <mesh position={[0, -netHang, 0]}><sphereGeometry args={[0.06, 6, 6]} /><meshStandardMaterial color="#aaa" /></mesh>
        </group>
      )}
    </group>
  );
}

function Trail({ points, color }) {
  if (!points || points.length < 2) return null;
  return <Line points={points} color={color} lineWidth={2} transparent opacity={0.5} />;
}

function TangledNet({ position, active, fallProgress }) {
  if (!active) return null;
  const fall = fallProgress * 3, tangle = Math.min(fallProgress * 2, 1), radius = 0.8 - tangle * 0.3;
  return (
    <group position={[position[0], position[1] - fall, position[2]]}>
      <mesh><sphereGeometry args={[radius, 8, 6]} /><meshStandardMaterial color="#f5f5dc" wireframe transparent opacity={0.6 + tangle * 0.4} side={THREE.DoubleSide} /></mesh>
      {[0, 0.8, 1.6, 2.4, 3.2, 4.0].map((a, i) => (
        <Line key={i} points={[[Math.cos(a)*radius, radius*0.5, Math.sin(a)*radius],[0, -radius*0.3, 0],[Math.cos(a+1.5)*radius, -radius*0.5, Math.sin(a+1.5)*radius]]}
          color="#f5f5dc" lineWidth={1} transparent opacity={0.5 + tangle*0.5} />
      ))}
    </group>
  );
}

function CaptureEffect({ position, active }) {
  const [sparks] = useState(() => Array.from({length:50}, () => ({ off: [(Math.random()-0.5)*3,(Math.random()-0.5)*3,(Math.random()-0.5)*3], spd: 0.3 + Math.random()*1.2 })));
  const [t, setT] = useState(0);
  useFrame((_, dt) => { if (active) setT(p => p + dt); });
  if (!active) return null;
  return (
    <group position={position}>
      {sparks.map((s,i) => { const p = Math.min(t * s.spd, 1), fade = 1-p; return (
        <mesh key={i} position={[s.off[0]*p, s.off[1]*p, s.off[2]*p]}><sphereGeometry args={[0.06*fade,4,4]} /><meshStandardMaterial color="#ff6600" emissive="#ff3300" emissiveIntensity={3*fade} transparent opacity={fade} /></mesh>
      ); })}
    </group>
  );
}

/* ── Geographic 3D scene ── */
function GeoScene({ frames, currentIdx, baseName, iranLabel, iranLat, iranLon }) {
  const [netProgress, setNetProgress] = useState(0);
  const [captured, setCaptured] = useState(false);
  const [capturedTime, setCapturedTime] = useState(0);
  useFrame((_, dt) => { if (captured && capturedTime < 1) setCapturedTime(p => Math.min(p + dt * 0.5, 1)); });
  const intTrail = useRef([]);
  const tgtTrail = useRef([]);
  const prevIdx = useRef(-1);

  const bounds = React.useMemo(() => frames && frames.length > 0 ? computeBounds(frames, iranLat, iranLon) : null, [frames, iranLat, iranLon]);

  useEffect(() => { intTrail.current = []; tgtTrail.current = []; setNetProgress(0); setCaptured(false); setCapturedTime(0); prevIdx.current = -1; }, [frames]);

  useEffect(() => {
    if (!frames || !bounds || currentIdx < 0 || currentIdx === prevIdx.current) return;
    prevIdx.current = currentIdx;
    const f = frames[currentIdx];
    intTrail.current = [...intTrail.current, geoTo3D(f.intLat, f.intLon, f.intAlt, bounds)];
    tgtTrail.current = [...tgtTrail.current, geoTo3D(f.advLat, f.advLon, f.advAlt, bounds)];
    if (f.pn === 4 && !captured) { setCaptured(true); setNetProgress(1); }
    else if (f.pn === 3 && f.distKm < 15) { setNetProgress(Math.min(1, (15 - f.distKm) / 15)); }
  }, [currentIdx, frames, bounds, captured]);

  if (!frames || frames.length === 0 || !bounds || currentIdx < 0) return null;
  const f = frames[Math.min(currentIdx, frames.length - 1)];
  const iPos = geoTo3D(f.intLat, f.intLon, f.intAlt, bounds);
  const tPos = geoTo3D(f.advLat, f.advLon, f.advAlt, bounds);
  const basePos = geoTo3D(frames[0].intLat, frames[0].intLon, 0, bounds); basePos[1] = 0;
  const originPos = (iranLat != null && iranLon != null) ? geoTo3D(iranLat, iranLon, 0, bounds) : geoTo3D(frames[0].advLat, frames[0].advLon, 0, bounds); originPos[1] = 0;

  return (
    <>
      <ambientLight intensity={0.3} /><directionalLight position={[20, 25, 15]} intensity={1.0} /><pointLight position={[-10, 15, -10]} intensity={0.3} color="#4488ff" />
      <fog attach="fog" args={['#0a0f1a', 40, 80]} />
      <group>
        <gridHelper args={[44, 44, '#0a2a3a', '#081828']} position={[20, -0.01, 20]} />
        <mesh rotation={[-Math.PI/2,0,0]} position={[20, -0.05, 20]}><planeGeometry args={[50, 50]} /><meshStandardMaterial color="#040a14" /></mesh>
      </group>
      {/* Base marker */}
      <group position={basePos}>
        <mesh><cylinderGeometry args={[0.5, 0.5, 0.08, 16]} /><meshStandardMaterial color="#00cc66" emissive="#00cc66" emissiveIntensity={0.5} /></mesh>
        <mesh position={[0, 1, 0]}><boxGeometry args={[0.08, 2, 0.08]} /><meshStandardMaterial color="#00cc66" transparent opacity={0.4} /></mesh>
        <Billboard><Text fontSize={0.25} color="#00ff88" position={[0, 2.3, 0]} anchorY="bottom" font={undefined}>{baseName}</Text></Billboard>
      </group>
      {/* Origin marker */}
      <group position={originPos}>
        <mesh><cylinderGeometry args={[0.4, 0.4, 0.08, 16]} /><meshStandardMaterial color="#ff4444" emissive="#ff4444" emissiveIntensity={0.5} /></mesh>
        <Billboard><Text fontSize={0.2} color="#ff6666" position={[0, 0.8, 0]} anchorY="bottom" font={undefined}>{iranLabel || 'ORIGIN'}</Text></Billboard>
      </group>
      <Trail points={intTrail.current} color="#00ff88" /><Trail points={tgtTrail.current} color="#ff4444" />
      <Line points={[iPos, tPos]} color="#ffaa00" lineWidth={1} transparent opacity={0.3} />
      {!captured && <Drone position={tPos} color="#ff2222" label="ADVERSARY" rotorSpeed={14} />}
      {f.pn >= 2 && <Drone position={iPos} color="#00cc66" label={f.pn >= 3 ? 'INTERCEPTOR' : 'STANDBY'} rotorSpeed={f.pn >= 3 ? 22 : 8} showNet={f.pn >= 3 && !captured} netDeploy={netProgress} />}
      <TangledNet position={tPos} active={captured} fallProgress={capturedTime} />
      <CaptureEffect position={captured ? [tPos[0], tPos[1] - capturedTime * 3, tPos[2]] : tPos} active={captured} />
      <OrbitControls target={[(iPos[0]+tPos[0])/2, (iPos[1]+tPos[1])/2, (iPos[2]+tPos[2])/2]} enablePan enableZoom maxPolarAngle={Math.PI * 0.48} autoRotate={captured} autoRotateSpeed={1.5} />
    </>
  );
}

/* ── Exported component ── */
export default function InterceptionScene({ visible, onClose, frames, currentIdx, baseName, iranLabel, iranLat, iranLon, speed }) {
  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape' && onClose) onClose(); };
    if (visible) window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [visible, onClose]);

  if (!visible) return null;

  const mono = { fontFamily: 'monospace' };
  const f = frames && currentIdx >= 0 ? frames[Math.min(currentIdx, frames.length - 1)] : null;

  return (
    <div style={{ position:'fixed', top:0, left:0, width:'100vw', height:'100vh', zIndex:9999, background:'#0a0f1a' }}>
      {/* Header */}
      <div style={{ position:'absolute', top:0, left:0, right:0, height:48, background:'linear-gradient(90deg,#0d1117,#1a2332)',
        display:'flex', alignItems:'center', justifyContent:'space-between', padding:'0 16px', zIndex:10000, borderBottom:'1px solid #00ff8840' }}>
        <span style={{ ...mono, color:'#00ff88', fontSize:13, fontWeight:'bold', letterSpacing:2 }}>
          3D INTERCEPTION VIEW — {(baseName || '').toUpperCase()}
        </span>
        <button onClick={onClose}
          style={{ background:'#ff4444', color:'#fff', border:'none', padding:'6px 16px', cursor:'pointer', ...mono, fontWeight:'bold', fontSize:11, letterSpacing:1 }}>
          CLOSE [ESC]
        </button>
      </div>

      {/* HUD overlay */}
      {f && (() => {
        const phaseColor = f.pn === 4 ? '#00ff88' : f.pn === 3 ? '#ffaa00' : f.pn === 2 ? '#ff8800' : '#888';
        const phaseText = f.pn === 4 ? 'NET CAPTURE — TARGET NEUTRALIZED' : f.pn === 3 ? 'INTERCEPTOR PURSUING' : f.pn === 2 ? 'RF DETECTED — SLEWING CAMERA' : 'ADVERSARY IN TRANSIT';
        return (
          <div style={{ position:'absolute', top:58, right:16, background:'#0d1117dd', padding:'10px 16px', ...mono, fontSize:12, color:'#ccc', border:`1px solid ${phaseColor}40`, lineHeight:1.8, minWidth:220, zIndex:10001 }}>
            <div style={{ color:phaseColor, fontWeight:'bold', fontSize:13, marginBottom:4 }}>{phaseText}</div>
            <div>Distance: <span style={{color:'#ffaa00'}}>{f.distKm} km</span></div>
            <div>Altitude: <span style={{color:'#aaa'}}>{Math.round(f.advAlt)}m</span></div>
            <div>Bearing: <span style={{color:'#aaa'}}>{f.bearing}&deg;</span></div>
            {f.ax != null && (<>
              <div style={{borderTop:'1px solid #333', marginTop:6, paddingTop:6, color:'#ff6666', fontWeight:'bold', fontSize:11}}>ADVERSARY (arena)</div>
              <div>x: <span style={{color:'#ff4444'}}>{f.ax}</span>  y: <span style={{color:'#ff4444'}}>{f.ay}</span>  z: <span style={{color:'#ff4444'}}>{f.az}</span></div>
              <div style={{color:'#00cc66', fontWeight:'bold', fontSize:11, marginTop:4}}>THRUST (N)</div>
              <div>Fx: <span style={{color:'#00ff88'}}>{f.tx}</span>  Fy: <span style={{color:'#00ff88'}}>{f.ty}</span>  Fz: <span style={{color:'#00ff88'}}>{f.tz}</span></div>
            </>)}
            <div style={{color:'#555', marginTop:4}}>Step {f.step} / {frames.length}</div>
          </div>
        );
      })()}

      {/* 3D Canvas */}
      <Canvas camera={{ position: [30, 18, 35], fov: 50 }} style={{ marginTop:48, height:'calc(100vh - 48px)' }} gl={{ antialias:true }}>
        <GeoScene frames={frames} currentIdx={currentIdx} baseName={baseName} iranLabel={iranLabel} iranLat={iranLat} iranLon={iranLon} />
      </Canvas>

      {/* No scenario prompt */}
      {(!frames || frames.length === 0) && (
        <div style={{ position:'absolute', top:'50%', left:'50%', transform:'translate(-50%,-50%)', ...mono, color:'#555', fontSize:15, textAlign:'center', lineHeight:2 }}>
          No scenario loaded.<br/>Close this view, select a base, and click <span style={{color:'#00cc66'}}>LAUNCH</span> first.
        </div>
      )}

      {/* Legend */}
      <div style={{ position:'absolute', bottom:16, left:16, background:'#0d1117cc', padding:'10px 14px', ...mono, fontSize:11, color:'#ccc', border:'1px solid #333', lineHeight:1.9 }}>
        <div><span style={{color:'#00cc66'}}>[●]</span> Interceptor (PPO)</div>
        <div><span style={{color:'#ff2222'}}>[●]</span> Adversary</div>
        <div><span style={{color:'#f5f5dc'}}>[●]</span> Net capture</div>
        <div><span style={{color:'#ffaa00'}}>—</span> Distance</div>
        <div style={{ marginTop:4, color:'#555' }}>Drag to rotate · Scroll to zoom · ESC close</div>
      </div>
    </div>
  );
}
