import React from 'react';

const STAGES = [
  { num: '01', title: 'RF Detection', desc: 'Detects radio signal\nRange: 80km\nCost: $5,000' },
  { num: '02', title: 'YOLO ID', desc: "Confirms it's a UAV\nVisual classification\nCost: $1,000" },
  { num: '03', title: 'RL Policy', desc: 'PPO computes thrust\nInference <1ms\nCost: $100' },
  { num: '04', title: 'Intercept', desc: 'Drone pursues & kills\nSpeed: 5 m/s\nCost: $300' },
];

export default function Pipeline({ activeStage = 0 }) {
  return (
    <div className="pipeline-row">
      {STAGES.map((s, i) => (
        <div key={i} className={`pipe-box ${activeStage >= i + 1 ? 'active' : ''}`}>
          <div className="pipe-title"><span className="pipe-num">{s.num}</span> {s.title}</div>
          <div className="pipe-desc">{s.desc.split('\n').map((l, j) => (
            <span key={j}>{l}<br /></span>
          ))}</div>
        </div>
      ))}
    </div>
  );
}
