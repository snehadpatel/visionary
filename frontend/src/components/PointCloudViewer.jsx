import React, { useRef, useMemo, useEffect, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import * as THREE from 'three';

/* ────────────────────────────────────────────────────
   Depth-Displaced Mesh — creates a real 3D surface
   from point-cloud data with smooth vertex coloring
   ──────────────────────────────────────────────────── */
const DepthMesh = ({ data }) => {
  const meshRef = useRef();

  const geometry = useMemo(() => {
    const count = data.count;
    const positions = new Float32Array(data.positions);
    const colors = new Float32Array(data.colors);

    // Find bounds for normalization
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (let i = 0; i < count; i++) {
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
    }

    // Normalize positions to a -2..2 cube
    const scale = 4 / Math.max(maxX - minX, maxY - minY, maxZ - minZ || 1);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;

    const normPos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      normPos[i * 3]     = (positions[i * 3]     - cx) * scale;
      normPos[i * 3 + 1] = (positions[i * 3 + 1] - cy) * scale;
      normPos[i * 3 + 2] = (positions[i * 3 + 2] - cz) * scale;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(normPos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeBoundingSphere();
    return geo;
  }, [data]);

  // Gentle rotation
  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.03;
    }
  });

  return (
    <points ref={meshRef} geometry={geometry}>
      <pointsMaterial
        size={0.012}
        vertexColors
        transparent
        opacity={0.92}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
};

/* ────────────────────────────────────────────────────
   Ambient Particles — floating dust for atmosphere
   ──────────────────────────────────────────────────── */
const AmbientParticles = () => {
  const ref = useRef();
  const count = 800;

  const [positions, sizes] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const sz = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      pos[i * 3]     = (Math.random() - 0.5) * 12;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 12;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 12;
      sz[i] = Math.random() * 0.02 + 0.005;
    }
    return [pos, sz];
  }, []);

  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y = state.clock.elapsedTime * 0.01;
      ref.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.005) * 0.1;
    }
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={count} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.015}
        color="#4466ff"
        transparent
        opacity={0.15}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
};

/* ────────────────────────────────────────────────────
   Walkthrough Camera — cinematic flythrough animation
   ──────────────────────────────────────────────────── */
const WalkthroughCamera = ({ isPlaying, onComplete, phase }) => {
  const { camera } = useThree();
  const timeRef = useRef(0);
  const startedRef = useRef(false);

  // Cinematic camera path: approach → orbit → dolly back
  const getWalkthroughPosition = useCallback((t) => {
    const duration = 12; // seconds
    const progress = Math.min(t / duration, 1);
    
    // Smooth easing
    const ease = progress < 0.5
      ? 2 * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 2) / 2;

    if (progress < 0.35) {
      // Phase 1: Approach from far → close
      const p = ease / 0.35;
      return {
        pos: new THREE.Vector3(
          6 - p * 4,
          2.5 - p * 1,
          5 - p * 3
        ),
        target: new THREE.Vector3(0, 0, 0)
      };
    } else if (progress < 0.7) {
      // Phase 2: Orbit around
      const p = (progress - 0.35) / 0.35;
      const angle = p * Math.PI * 0.8 - 0.2;
      return {
        pos: new THREE.Vector3(
          Math.cos(angle) * 3.5,
          1 + Math.sin(p * Math.PI) * 0.5,
          Math.sin(angle) * 3.5
        ),
        target: new THREE.Vector3(0, 0, 0)
      };
    } else {
      // Phase 3: Final position
      const p = (progress - 0.7) / 0.3;
      return {
        pos: new THREE.Vector3(
          3 * Math.cos(Math.PI * 0.6),
          1.5 + p * 0.5,
          3 * Math.sin(Math.PI * 0.6)
        ),
        target: new THREE.Vector3(0, -0.2, 0)
      };
    }
  }, []);

  useFrame((_, delta) => {
    if (!isPlaying) return;

    timeRef.current += delta;
    const { pos, target } = getWalkthroughPosition(timeRef.current);

    camera.position.lerp(pos, 0.04);
    const currentTarget = new THREE.Vector3();
    camera.getWorldDirection(currentTarget);
    currentTarget.add(camera.position);
    currentTarget.lerp(target, 0.04);
    camera.lookAt(target);

    if (timeRef.current >= 12 && !startedRef.current) {
      startedRef.current = true;
      onComplete?.();
    }
  });

  return null;
};

/* ────────────────────────────────────────────────────
   Grid Floor — subtle architectural reference grid
   ──────────────────────────────────────────────────── */
const ArchGrid = () => {
  const ref = useRef();
  useFrame((state) => {
    if (ref.current) {
      ref.current.material.opacity = 0.08 + Math.sin(state.clock.elapsedTime * 0.5) * 0.02;
    }
  });
  return (
    <gridHelper
      ref={ref}
      args={[16, 32, 0x2244ff, 0x112244]}
      position={[0, -2.5, 0]}
      rotation={[0, 0, 0]}
    />
  );
};

/* ────────────────────────────────────────────────────
   Main Viewer Component
   ──────────────────────────────────────────────────── */
const PointCloudViewer = ({ pcdUrl, onClose }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [walkthroughActive, setWalkthroughActive] = useState(true);
  const [walkthroughPhase, setWalkthroughPhase] = useState(0);
  const [showControls, setShowControls] = useState(false);
  const [pointCount, setPointCount] = useState(0);
  const controlsRef = useRef();

  useEffect(() => {
    if (!pcdUrl) return;

    setLoading(true);
    const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8080';
    fetch(`${backendUrl}${pcdUrl}`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(json => {
        setData(json);
        setPointCount(json.count || 0);
        setLoading(false);
        // Start walkthrough after a brief pause
        setTimeout(() => setWalkthroughActive(true), 500);
      })
      .catch(err => {
        console.error("Failed to load point cloud:", err);
        setError("Could not load 3D spatial data");
        setLoading(false);
      });
  }, [pcdUrl]);

  const handleWalkthroughComplete = useCallback(() => {
    setWalkthroughActive(false);
    setShowControls(true);
  }, []);

  const handleReplay = useCallback(() => {
    setWalkthroughActive(true);
    setShowControls(false);
  }, []);

  return (
    <div className="fixed inset-0 z-50 flex flex-col" style={{ background: 'linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a0f1e 100%)' }}>
      {/* ─── Cinematic Header ─── */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '16px 24px',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        background: 'rgba(0,0,0,0.4)',
        backdropFilter: 'blur(20px)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            width: 40, height: 40,
            borderRadius: 12,
            background: 'linear-gradient(135deg, #4466ff, #7744ff)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 18,
            boxShadow: '0 0 20px rgba(68,102,255,0.3)',
          }}>
            🧊
          </div>
          <div>
            <h2 style={{ fontSize: 16, fontWeight: 700, color: '#fff', margin: 0, letterSpacing: '-0.02em' }}>
              Spatial Walkthrough
            </h2>
            <p style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)', margin: 0, fontFamily: 'monospace', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              {walkthroughActive ? '● cinematic flythrough' : '● free exploration'}
              {pointCount > 0 && ` • ${(pointCount / 1000).toFixed(1)}K vertices`}
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {!walkthroughActive && (
            <button
              onClick={handleReplay}
              style={{
                padding: '8px 16px',
                background: 'rgba(68,102,255,0.15)',
                border: '1px solid rgba(68,102,255,0.3)',
                borderRadius: 10,
                color: '#7799ff',
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s',
                letterSpacing: '0.05em',
              }}
              onMouseOver={(e) => { e.target.style.background = 'rgba(68,102,255,0.3)'; e.target.style.color = '#fff'; }}
              onMouseOut={(e) => { e.target.style.background = 'rgba(68,102,255,0.15)'; e.target.style.color = '#7799ff'; }}
            >
              ↻ Replay Walkthrough
            </button>
          )}
          <button
            onClick={onClose}
            style={{
              width: 36, height: 36,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 10,
              color: 'rgba(255,255,255,0.5)',
              fontSize: 18,
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            onMouseOver={(e) => { e.target.style.background = 'rgba(255,60,60,0.15)'; e.target.style.color = '#ff6666'; }}
            onMouseOut={(e) => { e.target.style.background = 'rgba(255,255,255,0.05)'; e.target.style.color = 'rgba(255,255,255,0.5)'; }}
          >
            ✕
          </button>
        </div>
      </div>

      {/* ─── 3D Canvas ─── */}
      <div style={{ flex: 1, position: 'relative' }}>
        {/* Loading State */}
        {loading && (
          <div style={{
            position: 'absolute', inset: 0, zIndex: 10,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            gap: 16,
          }}>
            <div style={{
              width: 56, height: 56,
              border: '3px solid rgba(68,102,255,0.15)',
              borderTop: '3px solid #4466ff',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
            }} />
            <div style={{ textAlign: 'center' }}>
              <p style={{ color: '#fff', fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
                Reconstructing Geometry
              </p>
              <p style={{ color: 'rgba(255,255,255,0.3)', fontSize: 11, fontFamily: 'monospace' }}>
                Building 3D spatial mesh from depth data...
              </p>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div style={{
            position: 'absolute', inset: 0, zIndex: 10,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            gap: 12,
          }}>
            <div style={{ fontSize: 32 }}>⚠️</div>
            <p style={{ color: '#ff6666', fontSize: 14 }}>{error}</p>
            <button
              onClick={onClose}
              style={{
                padding: '8px 20px', background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8,
                color: '#fff', fontSize: 12, cursor: 'pointer',
              }}
            >
              Go Back
            </button>
          </div>
        )}

        {/* Three.js Scene */}
        {data && (
          <Canvas
            dpr={[1, 2]}
            gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
            style={{ background: 'transparent' }}
          >
            <PerspectiveCamera makeDefault position={[6, 2.5, 5]} fov={45} near={0.1} far={100} />

            <WalkthroughCamera
              isPlaying={walkthroughActive}
              onComplete={handleWalkthroughComplete}
              phase={walkthroughPhase}
            />

            {!walkthroughActive && (
              <OrbitControls
                ref={controlsRef}
                makeDefault
                enableDamping
                dampingFactor={0.05}
                rotateSpeed={0.6}
                zoomSpeed={0.8}
                minDistance={1}
                maxDistance={12}
                target={[0, 0, 0]}
              />
            )}

            {/* Lighting */}
            <ambientLight intensity={0.3} color="#4466ff" />
            <pointLight position={[5, 5, 5]} intensity={0.5} color="#ffffff" />
            <pointLight position={[-5, 3, -3]} intensity={0.3} color="#4466ff" />
            <pointLight position={[0, -3, 5]} intensity={0.2} color="#7744ff" />

            {/* Room Geometry */}
            <DepthMesh data={data} />

            {/* Atmosphere */}
            <AmbientParticles />
            <ArchGrid />

            {/* Fog for depth */}
            <fog attach="fog" args={['#0a0a1a', 6, 18]} />
          </Canvas>
        )}

        {/* ─── Walkthrough Phase Indicator ─── */}
        {walkthroughActive && data && (
          <div style={{
            position: 'absolute',
            bottom: 80, left: '50%', transform: 'translateX(-50%)',
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12,
            animation: 'fadeIn 1s ease',
          }}>
            <p style={{
              color: 'rgba(255,255,255,0.6)',
              fontSize: 13,
              fontWeight: 500,
              letterSpacing: '0.05em',
              textShadow: '0 2px 10px rgba(0,0,0,0.5)',
            }}>
              ✦ Cinematic walkthrough in progress...
            </p>
            <div style={{
              width: 200, height: 2,
              background: 'rgba(255,255,255,0.1)',
              borderRadius: 4,
              overflow: 'hidden',
            }}>
              <div style={{
                height: '100%',
                background: 'linear-gradient(90deg, #4466ff, #7744ff)',
                borderRadius: 4,
                animation: 'progressBar 12s linear',
                boxShadow: '0 0 10px rgba(68,102,255,0.5)',
              }} />
            </div>
            <button
              onClick={() => { setWalkthroughActive(false); setShowControls(true); }}
              style={{
                padding: '6px 14px',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 8,
                color: 'rgba(255,255,255,0.4)',
                fontSize: 10,
                cursor: 'pointer',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
              }}
            >
              Skip →
            </button>
          </div>
        )}

        {/* ─── Controls Legend ─── */}
        {showControls && (
          <div style={{
            position: 'absolute',
            bottom: 24, right: 24,
            padding: '16px 20px',
            background: 'rgba(10,10,26,0.6)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 16,
            backdropFilter: 'blur(16px)',
            animation: 'fadeIn 0.5s ease',
          }}>
            <p style={{ fontSize: 9, fontWeight: 700, color: 'rgba(255,255,255,0.3)', textTransform: 'uppercase', letterSpacing: '0.15em', marginBottom: 10 }}>
              Navigation Controls
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                { icon: '◎', label: 'Orbit', key: 'Left Click + Drag', color: '#4466ff' },
                { icon: '⊞', label: 'Pan', key: 'Right Click + Drag', color: '#44cc88' },
                { icon: '◉', label: 'Zoom', key: 'Scroll Wheel', color: '#aa66ff' },
              ].map(c => (
                <div key={c.label} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ color: c.color, fontSize: 12, width: 16, textAlign: 'center' }}>{c.icon}</span>
                  <span style={{ color: 'rgba(255,255,255,0.5)', fontSize: 11, flex: 1 }}>{c.label}</span>
                  <span style={{ color: 'rgba(255,255,255,0.2)', fontSize: 9, fontFamily: 'monospace', background: 'rgba(255,255,255,0.04)', padding: '2px 6px', borderRadius: 4 }}>{c.key}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ─── Spatial Stats ─── */}
        {data && (
          <div style={{
            position: 'absolute',
            bottom: 24, left: 24,
            padding: '12px 16px',
            background: 'rgba(10,10,26,0.6)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 12,
            backdropFilter: 'blur(16px)',
          }}>
            <div style={{ display: 'flex', gap: 20 }}>
              {[
                { label: 'Vertices', value: `${(pointCount / 1000).toFixed(1)}K` },
                { label: 'Density', value: pointCount > 30000 ? 'High' : pointCount > 15000 ? 'Medium' : 'Sparse' },
                { label: 'Render', value: 'WebGL 2.0' },
              ].map(s => (
                <div key={s.label} style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: 14, fontWeight: 700, color: '#fff', margin: 0 }}>{s.value}</p>
                  <p style={{ fontSize: 8, color: 'rgba(255,255,255,0.3)', margin: 0, fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.1em' }}>{s.label}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ─── CSS Animations ─── */}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes progressBar {
          from { width: 0%; }
          to { width: 100%; }
        }
      `}</style>
    </div>
  );
};

export default PointCloudViewer;
