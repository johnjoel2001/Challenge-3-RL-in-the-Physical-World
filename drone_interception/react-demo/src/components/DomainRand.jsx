import React from 'react';

export default function DomainRand({ params, seed }) {
  if (!params) return null;

  const items = [
    ['Mass', `${params.drone_mass}kg`],
    ['Thrust', `${params.max_force}N`],
    ['Drag', `${params.drag_coeff}`],
    ['Evader', `${params.evader_speed}m/s`],
    ['Obstacles', `${params.num_obstacles}`],
    ['Noise', `σ${params.obs_noise_std}`],
    ['Delay', `${params.action_delay}step`],
    ['Gravity', `${params.gravity}`],
  ];

  return (
    <div>
      <div className="dr-row">
        {items.map(([label, val]) => (
          <div className="dr-box" key={label}>
            <div className="dr-label">{label}</div>
            <div className="dr-val">{val}</div>
          </div>
        ))}
      </div>
      <div className="dr-caption">
        DOMAIN RANDOMIZATION — physics randomized per episode for sim2real robustness (seed: {seed})
      </div>
    </div>
  );
}
