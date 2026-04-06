import React, { useRef, useEffect } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { AIRBASES } from '../scenario';

const PHASE_YOLO = 0.72;

// Satellite + labels — same look as Mapbox satellite-streets used by pydeck
const MAP_STYLE = {
  version: 8,
  glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
  sources: {
    'esri-sat': {
      type: 'raster',
      tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
      tileSize: 256, maxzoom: 18, attribution: 'Esri',
    },
    'esri-lbl': {
      type: 'raster',
      tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'],
      tileSize: 256, maxzoom: 18,
    },
  },
  layers: [
    { id: 'sat', type: 'raster', source: 'esri-sat' },
    { id: 'lbl', type: 'raster', source: 'esri-lbl' },
  ],
};

// Generate ring coordinates
function ring(lon, lat, deg, pts = 80) {
  const coords = [];
  for (let i = 0; i <= pts; i++) {
    const a = (i / pts) * 2 * Math.PI;
    coords.push([lon + deg * Math.cos(a), lat + deg * Math.sin(a)]);
  }
  return coords;
}

// Helper: set or update a GeoJSON source
function setSource(map, id, geojson) {
  const src = map.getSource(id);
  if (src) { src.setData(geojson); }
  else { map.addSource(id, { type: 'geojson', data: geojson }); }
}

// Build all GeoJSON data + layers for the current state
function updateMap(map, baseName, frames, currentIdx, iranLabel, iranLat, iranLon) {
  const base = AIRBASES[baseName];
  if (!base) return;
  const blon = base.lon, blat = base.lat;
  const n = (frames && currentIdx >= 0) ? Math.min(currentIdx + 1, frames.length) : 0;
  const cur = n > 0 ? frames[n - 1] : null;

  // ── ADVERSARY TRAIL ──
  const advCoords = n >= 2 ? frames.slice(0, n).map(f => [f.advLon, f.advLat]) : [];
  setSource(map, 'adv-trail', { type: 'Feature', geometry: { type: 'LineString', coordinates: advCoords.length >= 2 ? advCoords : [[0,0],[0,0]] }, properties: {} });

  // ── INTERCEPTOR TRAIL ── (include base as start so full pursuit path is visible)
  let intCoords = [];
  if (cur && cur.pn >= 3) {
    intCoords = [[blon, blat]].concat(
      frames.slice(0, n).filter(f => f.t >= PHASE_YOLO).map(f => [f.intLon, f.intLat])
    );
  }
  setSource(map, 'int-trail', { type: 'Feature', geometry: { type: 'LineString', coordinates: intCoords.length >= 2 ? intCoords : [[0,0],[0,0]] }, properties: {} });

  // ── THREAT ARC (straight line Iran→Base) ──
  const arcCoords = (iranLat && iranLon) ? [[iranLon, iranLat], [blon, blat]] : [[0,0],[0,0]];
  setSource(map, 'threat-arc', { type: 'Feature', geometry: { type: 'LineString', coordinates: arcCoords }, properties: {} });

  // ── RF RING ──
  setSource(map, 'rf-ring', { type: 'Feature', geometry: { type: 'LineString', coordinates: ring(blon, blat, 0.72) }, properties: {} });

  // ── YOLO RING ──
  setSource(map, 'yolo-ring', { type: 'Feature', geometry: { type: 'LineString', coordinates: ring(blon, blat, 0.27) }, properties: {} });

  // ── POINT MARKERS (base, iran, drones, kill) ──
  const points = [];
  // Base
  points.push({ type: 'Feature', geometry: { type: 'Point', coordinates: [blon, blat] },
    properties: { kind: 'base', label: baseName.split(',')[0], color: '#33ddff' } });
  // Iran
  if (iranLat && iranLon) {
    points.push({ type: 'Feature', geometry: { type: 'Point', coordinates: [iranLon, iranLat] },
      properties: { kind: 'iran', label: `${iranLabel}, Iran`, color: '#ff5555' } });
  }
  // Adversary drone
  if (cur) {
    points.push({ type: 'Feature', geometry: { type: 'Point', coordinates: [cur.advLon, cur.advLat] },
      properties: { kind: 'adv', label: `HOSTILE UAV | ${cur.distKm}km`, color: '#ff4444' } });
  }
  // Interceptor drone
  if (cur && cur.pn >= 3) {
    points.push({ type: 'Feature', geometry: { type: 'Point', coordinates: [cur.intLon, cur.intLat] },
      properties: { kind: 'int', label: `RL INTERCEPTOR`, color: '#44ccff' } });
  }
  // Kill marker
  if (cur && cur.pn === 4) {
    points.push({ type: 'Feature', geometry: { type: 'Point', coordinates: [cur.advLon, cur.advLat] },
      properties: { kind: 'kill', label: '✅ KILL CONFIRMED', color: '#44ff44' } });
  }
  setSource(map, 'markers', { type: 'FeatureCollection', features: points });

  // ── LABELS ──
  const labels = [
    { type: 'Feature', geometry: { type: 'Point', coordinates: [blon + 0.74, blat] }, properties: { text: 'RF 80km', color: '#ffcc33' } },
    { type: 'Feature', geometry: { type: 'Point', coordinates: [blon + 0.29, blat] }, properties: { text: 'YOLO 30km', color: '#44ff88' } },
  ];
  setSource(map, 'ring-labels', { type: 'FeatureCollection', features: labels });

}

// Add all map layers (called once after map loads)
function addLayers(map) {
  // Adversary trail — glow + core
  map.addLayer({ id: 'adv-trail-glow', type: 'line', source: 'adv-trail',
    paint: { 'line-color': 'rgba(255,60,40,0.25)', 'line-width': 8, 'line-blur': 4 } });
  map.addLayer({ id: 'adv-trail-core', type: 'line', source: 'adv-trail',
    paint: { 'line-color': '#ff6644', 'line-width': 3 } });

  // Interceptor trail — glow + core
  map.addLayer({ id: 'int-trail-glow', type: 'line', source: 'int-trail',
    paint: { 'line-color': 'rgba(50,180,255,0.25)', 'line-width': 8, 'line-blur': 4 } });
  map.addLayer({ id: 'int-trail-core', type: 'line', source: 'int-trail',
    paint: { 'line-color': '#44ccff', 'line-width': 3 } });

  // Threat arc
  map.addLayer({ id: 'threat-arc-line', type: 'line', source: 'threat-arc',
    paint: { 'line-color': 'rgba(255,80,60,0.35)', 'line-width': 2, 'line-dasharray': [4, 4] } });

  // RF ring
  map.addLayer({ id: 'rf-ring-glow', type: 'line', source: 'rf-ring',
    paint: { 'line-color': 'rgba(255,200,50,0.15)', 'line-width': 6, 'line-blur': 3 } });
  map.addLayer({ id: 'rf-ring-core', type: 'line', source: 'rf-ring',
    paint: { 'line-color': 'rgba(255,200,50,0.6)', 'line-width': 1.5, 'line-dasharray': [6, 4] } });

  // YOLO ring
  map.addLayer({ id: 'yolo-ring-glow', type: 'line', source: 'yolo-ring',
    paint: { 'line-color': 'rgba(50,255,120,0.15)', 'line-width': 6, 'line-blur': 3 } });
  map.addLayer({ id: 'yolo-ring-core', type: 'line', source: 'yolo-ring',
    paint: { 'line-color': 'rgba(50,255,120,0.6)', 'line-width': 1.5, 'line-dasharray': [6, 4] } });

  // Marker halos (larger circles behind dots)
  map.addLayer({ id: 'marker-halos', type: 'circle', source: 'markers',
    paint: { 'circle-radius': 12, 'circle-color': ['get', 'color'], 'circle-opacity': 0.25, 'circle-blur': 0.5 } });

  // Marker dots
  map.addLayer({ id: 'marker-dots', type: 'circle', source: 'markers',
    filter: ['!=', ['get', 'kind'], 'kill'],
    paint: { 'circle-radius': 6, 'circle-color': ['get', 'color'], 'circle-stroke-width': 2, 'circle-stroke-color': '#000' } });

  // Kill marker — big green pulsing dot
  map.addLayer({ id: 'kill-dot', type: 'circle', source: 'markers',
    filter: ['==', ['get', 'kind'], 'kill'],
    paint: { 'circle-radius': 14, 'circle-color': '#44ff44', 'circle-opacity': 0.7,
             'circle-stroke-width': 3, 'circle-stroke-color': '#44ff44' } });

  // Marker labels
  map.addLayer({ id: 'marker-labels', type: 'symbol', source: 'markers',
    layout: {
      'text-field': ['get', 'label'], 'text-size': 11,
      'text-font': ['Open Sans Semibold'],
      'text-anchor': 'bottom', 'text-offset': [0, -1.5],
      'text-allow-overlap': true, 'text-ignore-placement': true,
    },
    paint: { 'text-color': ['get', 'color'], 'text-halo-color': '#000', 'text-halo-width': 2 } });

  // Ring labels
  map.addLayer({ id: 'ring-labels', type: 'symbol', source: 'ring-labels',
    layout: {
      'text-field': ['get', 'text'], 'text-size': 11,
      'text-font': ['Open Sans Semibold'],
      'text-anchor': 'left', 'text-allow-overlap': true,
    },
    paint: { 'text-color': ['get', 'color'], 'text-halo-color': '#000', 'text-halo-width': 2 } });
}

/**
 * MapLibre GL JS map — same rendering engine as pydeck/Streamlit.
 */
export default function MapView({ baseName, frames, currentIdx, iranLabel, iranLat, iranLon }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const readyRef = useRef(false);

  const base = AIRBASES[baseName];

  // ── Create map once ──
  useEffect(() => {
    if (!containerRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: MAP_STYLE,
      center: [base.lon, base.lat],
      zoom: 7,
      pitch: 45,
      bearing: 10,
      attributionControl: false,
    });
    map.addControl(new maplibregl.NavigationControl(), 'top-right');
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right');

    map.on('load', () => {
      // Initialize all sources with empty data
      const emptyLine = { type: 'Feature', geometry: { type: 'LineString', coordinates: [[0,0],[0,0]] }, properties: {} };
      const emptyPts = { type: 'FeatureCollection', features: [] };
      map.addSource('adv-trail', { type: 'geojson', data: emptyLine });
      map.addSource('int-trail', { type: 'geojson', data: emptyLine });
      map.addSource('threat-arc', { type: 'geojson', data: emptyLine });
      map.addSource('rf-ring', { type: 'geojson', data: emptyLine });
      map.addSource('yolo-ring', { type: 'geojson', data: emptyLine });
      map.addSource('markers', { type: 'geojson', data: emptyPts });
      map.addSource('ring-labels', { type: 'geojson', data: emptyPts });
      addLayers(map);
      readyRef.current = true;
      // Draw initial state immediately
      updateMap(map, baseName, frames, currentIdx, iranLabel, iranLat, iranLon);
    });

    mapRef.current = map;
    return () => { readyRef.current = false; map.remove(); mapRef.current = null; };
  }, []);

  // ── Update data on every prop change ──
  useEffect(() => {
    if (!readyRef.current || !mapRef.current) return;
    updateMap(mapRef.current, baseName, frames, currentIdx, iranLabel, iranLat, iranLon);
  }, [baseName, frames, currentIdx, iranLabel, iranLat, iranLon]);

  // ── Fly to mission area when Iran site appears ──
  useEffect(() => {
    if (!mapRef.current || !iranLat || !iranLon) return;
    const midLat = (base.lat + iranLat) / 2;
    const midLon = (base.lon + iranLon) / 2;
    const span = Math.max(Math.abs(base.lat - iranLat), Math.abs(base.lon - iranLon));
    const zoom = span < 5 ? 6.5 : span < 8 ? 5.8 : 5.2;
    mapRef.current.flyTo({ center: [midLon, midLat], zoom, pitch: 45, bearing: 10, duration: 1500 });
  }, [iranLat, iranLon]);

  // ── Re-center when base changes ──
  useEffect(() => {
    if (!mapRef.current) return;
    const b = AIRBASES[baseName];
    if (b) mapRef.current.flyTo({ center: [b.lon, b.lat], zoom: 7, duration: 1000 });
  }, [baseName]);

  return <div className="map-panel" ref={containerRef} />;
}
