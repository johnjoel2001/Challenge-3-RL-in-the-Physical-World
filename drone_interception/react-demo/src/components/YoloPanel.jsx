import React, { useRef, useEffect } from 'react';

export default function YoloPanel({ frame }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    // Grainy dark background
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, W, H);

    // Noise grain
    const imgData = ctx.getImageData(0, 0, W, H);
    for (let i = 0; i < imgData.data.length; i += 4) {
      const n = Math.random() * 20;
      imgData.data[i] += n;
      imgData.data[i + 1] += n;
      imgData.data[i + 2] += n;
    }
    ctx.putImageData(imgData, 0, 0);

    // Camera info top-left
    ctx.font = '10px Courier New';
    ctx.fillStyle = '#446688';
    ctx.fillText('CAM-01  PTZ AUTO-TRACK', 8, 14);

    // Frame count top-right
    ctx.textAlign = 'right';
    ctx.fillText(`FRAME: ${frame ? frame.step : 0}`, W - 8, 14);
    ctx.textAlign = 'left';

    if (!frame || frame.pn <= 1) {
      // NO DETECTION
      ctx.font = '16px Courier New';
      ctx.fillStyle = '#334455';
      ctx.textAlign = 'center';
      ctx.fillText('SCANNING', W / 2, H / 2 - 10);
      ctx.font = '11px Courier New';
      ctx.fillText('Awaiting RF handoff...', W / 2, H / 2 + 10);
      ctx.textAlign = 'left';

      // REC indicator
      ctx.fillStyle = '#333';
      ctx.beginPath();
      ctx.arc(W - 16, H - 16, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = '9px Courier New';
      ctx.fillStyle = '#333';
      ctx.textAlign = 'right';
      ctx.fillText('STANDBY', W - 26, H - 12);
      ctx.textAlign = 'left';
    } else if (frame.pn === 2) {
      // RF DETECTED — camera slewing
      ctx.font = '14px Courier New';
      ctx.fillStyle = '#ffaa44';
      ctx.textAlign = 'center';
      ctx.fillText('RF SIGNAL RECEIVED', W / 2, H / 2 - 20);
      ctx.font = '11px Courier New';
      ctx.fillStyle = '#886633';
      ctx.fillText(`PTZ slewing to BRG ${frame.bearing}°`, W / 2, H / 2 + 5);
      ctx.fillText('Waiting for visual acquisition...', W / 2, H / 2 + 22);
      ctx.textAlign = 'left';

      // Animated crosshair
      const cx = W / 2;
      const cy = H / 2 + 40;
      ctx.strokeStyle = '#ffaa4466';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx - 20, cy); ctx.lineTo(cx + 20, cy);
      ctx.moveTo(cx, cy - 20); ctx.lineTo(cx, cy + 20);
      ctx.stroke();

      // REC blink
      ctx.fillStyle = '#ff8844';
      ctx.beginPath();
      ctx.arc(W - 16, H - 16, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = '9px Courier New';
      ctx.fillStyle = '#ff8844';
      ctx.textAlign = 'right';
      ctx.fillText('SLEWING', W - 26, H - 12);
      ctx.textAlign = 'left';
    } else {
      // YOLO ACTIVE (pn 3 or 4)
      const isKill = frame.pn === 4;
      const conf = frame.yc;

      // Draw drone silhouette
      const cx = W / 2;
      const cy = H / 2;
      ctx.fillStyle = '#333';
      // Body
      ctx.fillRect(cx - 8, cy - 3, 16, 6);
      // Arms
      ctx.fillRect(cx - 25, cy - 1, 50, 2);
      // Rotors
      for (const dx of [-22, -12, 12, 22]) {
        ctx.beginPath();
        ctx.arc(cx + dx, cy, 6, 0, Math.PI * 2);
        ctx.fill();
      }

      // Bounding box
      const boxColor = isKill ? '#44ff44' : '#ff4444';
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(cx - 50, cy - 40, 100, 80);

      // Class label
      ctx.font = 'bold 12px Courier New';
      ctx.fillStyle = boxColor;
      ctx.fillText(`UAV  ${(conf * 100).toFixed(0)}%`, cx - 48, cy - 44);

      // Confidence bar
      const barW = 80;
      const barH = 6;
      const barX = cx - 48;
      const barY = cy + 48;
      ctx.fillStyle = '#222';
      ctx.fillRect(barX, barY, barW, barH);
      ctx.fillStyle = boxColor;
      ctx.fillRect(barX, barY, barW * conf, barH);

      // Distance
      ctx.font = '10px Courier New';
      ctx.fillStyle = '#88aacc';
      ctx.fillText(`DIST: ${frame.distKm}km`, cx - 48, barY + 18);

      // Threat level
      const threatColor = isKill ? '#44ff44' : '#ff4444';
      ctx.font = 'bold 10px Courier New';
      ctx.fillStyle = threatColor;
      ctx.textAlign = 'right';
      ctx.fillText(isKill ? 'NEUTRALIZED' : 'THREAT: HIGH', W - 12, barY + 18);
      ctx.textAlign = 'left';

      // Status line at bottom
      ctx.font = '10px Courier New';
      ctx.fillStyle = isKill ? '#44ff44' : '#ff8844';
      ctx.textAlign = 'center';
      ctx.fillText(
        isKill ? 'TARGET NEUTRALIZED' : 'CONFIRMED: Hostile UAV by YOLOv8',
        W / 2, H - 10
      );
      ctx.textAlign = 'left';

      // REC indicator
      ctx.fillStyle = isKill ? '#44ff44' : '#ff0000';
      ctx.beginPath();
      ctx.arc(W - 16, H - 40, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = '9px Courier New';
      ctx.fillStyle = isKill ? '#44ff44' : '#ff0000';
      ctx.textAlign = 'right';
      ctx.fillText('REC ●', W - 26, H - 36);
      ctx.textAlign = 'left';
    }
  }, [frame]);

  return (
    <div className="yolo-panel">
      <canvas
        ref={canvasRef}
        className="yolo-canvas"
        width={440}
        height={240}
      />
    </div>
  );
}
