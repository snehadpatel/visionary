import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { PerspectiveCamera, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

/* ── Room Image Wall: project redesigned image as the room environment ── */
const RoomEnvironment = ({ resultUrl }) => {
  const [tex, setTex] = useState(null);
  useEffect(() => {
    if (!resultUrl) return;
    const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8080';
    const url = resultUrl.startsWith('http') ? resultUrl : `${backendUrl}${resultUrl}`;
    let active = true;
    let loadedTex = null;

    new THREE.TextureLoader().load(url, t => {
      if (active) {
        t.colorSpace = THREE.SRGBColorSpace;
        setTex(t);
        loadedTex = t;
      } else {
        t.dispose();
      }
    });

    return () => {
      active = false;
      if (loadedTex) loadedTex.dispose();
    };
  }, [resultUrl]);

  if (!tex) return null;
  const aspect = tex.image ? tex.image.width / tex.image.height : 16/9;
  const h = 5, w = h * aspect;

  return (
    <group>
      {/* Main back wall — the redesigned room image */}
      <mesh position={[0, 0, -4]}>
        <planeGeometry args={[w, h]} />
        <meshBasicMaterial map={tex} side={THREE.FrontSide} />
      </mesh>
      {/* Floor */}
      <mesh rotation={[-Math.PI/2, 0, 0]} position={[0, -h/2, 0]}>
        <planeGeometry args={[w * 2, 12]} />
        <meshStandardMaterial color="#0c0e16" roughness={0.9} metalness={0.1} />
      </mesh>
      {/* Ceiling */}
      <mesh rotation={[Math.PI/2, 0, 0]} position={[0, h/2, 0]}>
        <planeGeometry args={[w * 2, 12]} />
        <meshStandardMaterial color="#0a0c14" roughness={1} />
      </mesh>
      {/* Left wall */}
      <mesh rotation={[0, Math.PI/2, 0]} position={[-w/2, 0, 0]}>
        <planeGeometry args={[12, h]} />
        <meshStandardMaterial color="#0b0d15" roughness={0.95} />
      </mesh>
      {/* Right wall */}
      <mesh rotation={[0, -Math.PI/2, 0]} position={[w/2, 0, 0]}>
        <planeGeometry args={[12, h]} />
        <meshStandardMaterial color="#0b0d15" roughness={0.95} />
      </mesh>
    </group>
  );
};

/* ── Depth Layers: split image into depth slices for parallax ── */
const DepthLayers = ({ resultUrl, depthUrl }) => {
  const groupRef = useRef();
  const [layers, setLayers] = useState([]);

  useEffect(() => {
    if (!resultUrl || !depthUrl) return;
    const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8080';
    const rUrl = resultUrl.startsWith('http') ? resultUrl : `${backendUrl}${resultUrl}`;
    const dUrl = depthUrl.startsWith('http') ? depthUrl : `${backendUrl}${depthUrl}`;

    let active = true;
    let createdTextures = [];

    const colorImg = new Image(); colorImg.crossOrigin = 'anonymous';
    const depthImg = new Image(); depthImg.crossOrigin = 'anonymous';
    let loaded = 0;

    const process = () => {
      if (++loaded < 2) return;
      if (!active) return;
      const W = 256, H = 256;
      // Draw color
      const cc = document.createElement('canvas'); cc.width = W; cc.height = H;
      const cctx = cc.getContext('2d'); cctx.drawImage(colorImg, 0, 0, W, H);
      const colorData = cctx.getImageData(0, 0, W, H);
      // Draw depth
      const dc = document.createElement('canvas'); dc.width = W; dc.height = H;
      const dctx = dc.getContext('2d'); dctx.drawImage(depthImg, 0, 0, W, H);
      const depthData = dctx.getImageData(0, 0, W, H);

      // Create 4 depth layers
      const layerDefs = [
        { min: 0, max: 60, z: -1.5, opacity: 0.9 },    // very close foreground
        { min: 60, max: 120, z: -2.5, opacity: 0.85 },  // mid foreground
        { min: 120, max: 190, z: -3.2, opacity: 0.8 },  // mid background
        { min: 190, max: 256, z: -3.8, opacity: 0.7 },  // far background
      ];

      const result = layerDefs.map(def => {
        const lc = document.createElement('canvas'); lc.width = W; lc.height = H;
        const lctx = lc.getContext('2d');
        const imgData = lctx.createImageData(W, H);
        for (let i = 0; i < W * H; i++) {
          const d = depthData.data[i * 4]; // depth value (0=far, 255=near for MiDaS)
          const inRange = d >= def.min && d < def.max;
          imgData.data[i*4]   = colorData.data[i*4];
          imgData.data[i*4+1] = colorData.data[i*4+1];
          imgData.data[i*4+2] = colorData.data[i*4+2];
          imgData.data[i*4+3] = inRange ? 255 : 0;
        }
        lctx.putImageData(imgData, 0, 0);
        const tex = new THREE.CanvasTexture(lc);
        tex.colorSpace = THREE.SRGBColorSpace;
        createdTextures.push(tex);
        return { tex, z: def.z, opacity: def.opacity };
      });
      setLayers(result);
    };

    colorImg.onload = process;
    depthImg.onload = process;
    colorImg.src = rUrl;
    depthImg.src = dUrl;

    return () => {
      active = false;
      createdTextures.forEach(t => t.dispose());
    };
  }, [resultUrl, depthUrl]);

  // Subtle floating animation
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.02;
    }
  });

  const aspect = 16/9;
  return (
    <group ref={groupRef}>
      {layers.map((l, i) => (
        <mesh key={i} position={[0, 0, l.z]}>
          <planeGeometry args={[5 * aspect * (0.85 + i * 0.05), 5 * (0.85 + i * 0.05)]} />
          <meshBasicMaterial map={l.tex} transparent opacity={l.opacity} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      ))}
    </group>
  );
};

/* ── First Person Controls (game-style WASD + mouse) ── */
const FPSControls = ({ active, speed = 2.5 }) => {
  const { camera, gl } = useThree();
  const keys = useRef({});
  const euler = useRef(new THREE.Euler(0, 0, 0, 'YXZ'));
  const locked = useRef(false);

  useEffect(() => {
    if (!active) return;
    const kd = e => { keys.current[e.code] = true; };
    const ku = e => { keys.current[e.code] = false; };
    const click = () => gl.domElement.requestPointerLock?.();
    const lockChange = () => { locked.current = document.pointerLockElement === gl.domElement; };
    const move = e => {
      if (!locked.current) return;
      euler.current.y -= e.movementX * 0.0018;
      euler.current.x = Math.max(-1.2, Math.min(1.2, euler.current.x - e.movementY * 0.0018));
      camera.quaternion.setFromEuler(euler.current);
    };
    window.addEventListener('keydown', kd); window.addEventListener('keyup', ku);
    gl.domElement.addEventListener('click', click);
    document.addEventListener('pointerlockchange', lockChange);
    document.addEventListener('mousemove', move);
    return () => {
      window.removeEventListener('keydown', kd); window.removeEventListener('keyup', ku);
      gl.domElement.removeEventListener('click', click);
      document.removeEventListener('pointerlockchange', lockChange);
      document.removeEventListener('mousemove', move);
      if (document.pointerLockElement) document.exitPointerLock();
    };
  }, [active, camera, gl]);

  useFrame((_, dt) => {
    if (!active) return;
    const s = speed * dt;
    const dir = new THREE.Vector3(); camera.getWorldDirection(dir);
    const right = new THREE.Vector3().crossVectors(dir, camera.up).normalize();
    const k = keys.current;
    if (k.KeyW || k.ArrowUp) camera.position.addScaledVector(dir, s);
    if (k.KeyS || k.ArrowDown) camera.position.addScaledVector(dir, -s);
    if (k.KeyA || k.ArrowLeft) camera.position.addScaledVector(right, -s);
    if (k.KeyD || k.ArrowRight) camera.position.addScaledVector(right, s);
    if (k.Space) camera.position.y += s;
    if (k.ShiftLeft) camera.position.y -= s;
    // Clamp position inside room
    camera.position.x = Math.max(-3.5, Math.min(3.5, camera.position.x));
    camera.position.y = Math.max(-1.5, Math.min(2, camera.position.y));
    camera.position.z = Math.max(-3.5, Math.min(5, camera.position.z));
  });
  return null;
};

/* ── Cinematic Auto-Camera ── */
const CinematicCamera = ({ active, onDone }) => {
  const { camera } = useThree();
  const t = useRef(0);
  const finished = useRef(false);

  useFrame((_, dt) => {
    if (!active || finished.current) return;
    t.current += dt;
    const dur = 8, p = Math.min(t.current / dur, 1);
    // Smooth approach from behind into the room
    const ease = 1 - Math.pow(1 - p, 3);
    const z = 6 - ease * 5.5;  // 6 → 0.5
    const y = 0.8 + Math.sin(p * Math.PI) * 0.4;
    const x = Math.sin(p * Math.PI * 0.4) * 0.6;
    camera.position.set(x, y, z);
    camera.lookAt(0, 0, -3);
    if (p >= 1 && !finished.current) { finished.current = true; onDone?.(); }
  });
  return null;
};

/* ── Floor Grid ── */
const FloorGrid = () => (
  <gridHelper args={[20, 30, 0x1a2744, 0x0d1520]} position={[0, -2.5, 0]} />
);

/* ── Ambient Dust Particles ── */
const Dust = () => {
  const ref = useRef();
  const positions = useMemo(() => {
    const p = new Float32Array(400 * 3);
    for (let i = 0; i < 400; i++) {
      p[i*3] = (Math.random()-0.5) * 10;
      p[i*3+1] = (Math.random()-0.5) * 6;
      p[i*3+2] = (Math.random()-0.5) * 10;
    }
    return p;
  }, []);
  useFrame(s => { if(ref.current) ref.current.rotation.y = s.clock.elapsedTime * 0.01; });
  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={400} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial size={0.008} color="#667fff" transparent opacity={0.15}
        depthWrite={false} blending={THREE.AdditiveBlending} sizeAttenuation />
    </points>
  );
};

/* ── Spatial Object: Individual 3D entities ── */
const SpatialObject = ({ obj }) => {
  const [tex, setTex] = useState(null);
  const [hovered, setHover] = useState(false);
  const meshRef = useRef();

  useEffect(() => {
    if (!obj.texture_url) return;
    const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8080';
    const url = `${backendUrl}${obj.texture_url}`;
    let active = true;
    let loadedTex = null;

    new THREE.TextureLoader().load(url, t => {
      if (active) {
        t.colorSpace = THREE.SRGBColorSpace;
        setTex(t);
        loadedTex = t;
      } else {
        t.dispose();
      }
    });

    return () => {
      active = false;
      if (loadedTex) loadedTex.dispose();
    };
  }, [obj.texture_url]);

  if (!tex) return null;

  return (
    <mesh 
      ref={meshRef} 
      position={obj.position} 
      scale={obj.scale}
      onPointerOver={() => setHover(true)}
      onPointerOut={() => setHover(false)}
    >
      <planeGeometry args={[1, 1]} />
      <meshStandardMaterial 
        map={tex} 
        transparent 
        side={THREE.DoubleSide} 
        alphaTest={0.5}
        depthWrite={true}
        emissive={hovered ? "#4466ff" : "#000000"}
        emissiveIntensity={hovered ? 0.5 : 0}
      />
      {/* Label above object when hovered */}
      {hovered && (
        <group position={[0, 0.6, 0]} scale={[1/obj.scale[0], 1/obj.scale[1], 1]}>
          <mesh position={[0, 0, 0.01]}>
            <planeGeometry args={[0.6, 0.18]} />
            <meshBasicMaterial color="#4466ff" transparent opacity={0.8} />
          </mesh>
        </group>
      )}
    </mesh>
  );
};

/* ════════════════════════════════════════════════════════
   MAIN COMPONENT
   ════════════════════════════════════════════════════════ */
const RoomWalkthrough = ({ pcdUrl, resultUrl, depthUrl, originalUrl, roomInfo, targetStyle, objects3D, onClose }) => {
  const [mode, setMode] = useState('cinematic');
  const [showHUD, setShowHUD] = useState(true);

  const onCinematicDone = useCallback(() => setMode('fps'), []);

  const css = {
    hud: { position:'absolute',padding:'10px 14px',background:'rgba(6,8,18,0.8)',
      border:'1px solid rgba(255,255,255,0.06)',borderRadius:12,backdropFilter:'blur(16px)' },
    label: { fontSize:8,color:'rgba(255,255,255,0.3)',fontFamily:'monospace',
      textTransform:'uppercase',letterSpacing:'0.12em',margin:0 },
    val: { fontSize:12,fontWeight:700,color:'#fff',margin:0,textTransform:'capitalize' },
  };

  return (
    <div style={{position:'fixed',inset:0,zIndex:50,display:'flex',flexDirection:'column',
      background:'#050510'}}>

      {/* ── Top Bar ── */}
      <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',
        padding:'10px 16px',borderBottom:'1px solid rgba(255,255,255,0.05)',
        background:'rgba(0,0,0,0.6)',backdropFilter:'blur(20px)',zIndex:20}}>
        <div style={{display:'flex',alignItems:'center',gap:12}}>
          <div style={{width:32,height:32,borderRadius:8,
            background:'linear-gradient(135deg,#4466ff,#7744ff)',
            display:'flex',alignItems:'center',justifyContent:'center',fontSize:15,
            boxShadow:'0 0 16px rgba(68,102,255,0.25)'}}>🥽</div>
          <div>
            <h2 style={{fontSize:14,fontWeight:700,color:'#fff',margin:0}}>3D Room Walkthrough</h2>
            <p style={{fontSize:9,color:'rgba(255,255,255,0.3)',margin:0,fontFamily:'monospace',
              letterSpacing:'0.08em',textTransform:'uppercase'}}>
              {mode === 'cinematic' ? '● cinematic entry' : mode === 'fps' ? '● first-person mode' : '● orbit mode'}
              {targetStyle && ` • ${targetStyle}`}
            </p>
          </div>
        </div>

        <div style={{display:'flex',gap:6,alignItems:'center'}}>
          {[
            {id:'cinematic',icon:'🎬',l:'Cinematic'},
            {id:'fps',icon:'🎮',l:'Walk (FPS)'},
            {id:'orbit',icon:'🌐',l:'Orbit'},
          ].map(m => (
            <button key={m.id} onClick={() => setMode(m.id)}
              style={{padding:'5px 10px',borderRadius:7,fontSize:9,fontWeight:700,
                cursor:'pointer',transition:'all 0.2s',letterSpacing:'0.06em',
                border:'1px solid',
                ...(mode===m.id
                  ? {background:'rgba(68,102,255,0.2)',borderColor:'rgba(68,102,255,0.4)',color:'#88aaff'}
                  : {background:'rgba(255,255,255,0.02)',borderColor:'rgba(255,255,255,0.06)',color:'rgba(255,255,255,0.3)'})
              }}>{m.icon} {m.l}</button>
          ))}
          <button onClick={onClose}
            style={{width:30,height:30,display:'flex',alignItems:'center',justifyContent:'center',
              background:'rgba(255,255,255,0.03)',border:'1px solid rgba(255,255,255,0.07)',
              borderRadius:8,color:'rgba(255,255,255,0.35)',fontSize:14,cursor:'pointer',
              marginLeft:6}}>✕</button>
        </div>
      </div>

      {/* ── 3D Scene ── */}
      <div style={{flex:1,position:'relative',cursor:mode==='fps'?'crosshair':'grab'}}>
        <Canvas dpr={[1,2]} gl={{antialias:true,alpha:true}} style={{background:'#050510'}}>
          <PerspectiveCamera makeDefault position={[0,0.5,5]} fov={60} near={0.05} far={50} />

          {mode === 'cinematic' && <CinematicCamera active onDone={onCinematicDone} />}
          {mode === 'fps' && <FPSControls active />}
          {mode === 'orbit' && <OrbitControls makeDefault enableDamping dampingFactor={0.05}
            target={[0,0,-2]} minDistance={0.3} maxDistance={10} />}

          {/* Lighting */}
          <ambientLight intensity={0.5} color="#c8d4ff" />
          <directionalLight position={[3,5,2]} intensity={0.6} color="#fff8ee" />
          <pointLight position={[-2,2,0]} intensity={0.3} color="#4466ff" />
          <hemisphereLight args={['#aabbff','#111122',0.3]} />

          {/* Room environment — image projected on walls */}
          <RoomEnvironment resultUrl={resultUrl} />

          {/* Individual 3D Objects */}
          {objects3D && objects3D.map(obj => (
            <SpatialObject key={obj.id} obj={obj} />
          ))}

          {/* Depth parallax layers — creates game-like depth feel */}
          <DepthLayers resultUrl={resultUrl} depthUrl={depthUrl} />

          {/* Atmosphere */}
          <Dust />
          <FloorGrid />
          <fog attach="fog" args={['#050510',6,18]} />
        </Canvas>

        {/* ── FPS Crosshair ── */}
        {mode === 'fps' && (
          <div style={{position:'absolute',top:'50%',left:'50%',transform:'translate(-50%,-50%)',
            pointerEvents:'none',zIndex:10}}>
            <div style={{width:20,height:20,position:'relative'}}>
              <div style={{position:'absolute',top:9,left:2,width:6,height:2,background:'rgba(255,255,255,0.4)',borderRadius:1}} />
              <div style={{position:'absolute',top:9,right:2,width:6,height:2,background:'rgba(255,255,255,0.4)',borderRadius:1}} />
              <div style={{position:'absolute',left:9,top:2,width:2,height:6,background:'rgba(255,255,255,0.4)',borderRadius:1}} />
              <div style={{position:'absolute',left:9,bottom:2,width:2,height:6,background:'rgba(255,255,255,0.4)',borderRadius:1}} />
              <div style={{position:'absolute',top:9,left:9,width:2,height:2,background:'#4466ff',borderRadius:'50%',
                boxShadow:'0 0 6px #4466ff'}} />
            </div>
            <p style={{color:'rgba(255,255,255,0.2)',fontSize:9,fontFamily:'monospace',textAlign:'center',
              marginTop:12,letterSpacing:'0.1em',textShadow:'0 1px 4px rgba(0,0,0,0.8)'}}>
              CLICK TO LOCK MOUSE • WASD TO MOVE
            </p>
          </div>
        )}

        {/* ── Cinematic Progress ── */}
        {mode === 'cinematic' && (
          <div style={{position:'absolute',bottom:60,left:'50%',transform:'translateX(-50%)',
            display:'flex',flexDirection:'column',alignItems:'center',gap:8,zIndex:10}}>
            <p style={{color:'rgba(255,255,255,0.5)',fontSize:11,letterSpacing:'0.06em'}}>
              ✦ Entering your redesigned room...
            </p>
            <div style={{width:160,height:2,background:'rgba(255,255,255,0.08)',borderRadius:2,overflow:'hidden'}}>
              <div style={{height:'100%',background:'linear-gradient(90deg,#4466ff,#7744ff)',
                animation:'prog 8s linear forwards',boxShadow:'0 0 8px rgba(68,102,255,0.4)'}} />
            </div>
            <button onClick={() => setMode('fps')}
              style={{padding:'4px 10px',background:'rgba(255,255,255,0.03)',
                border:'1px solid rgba(255,255,255,0.06)',borderRadius:6,
                color:'rgba(255,255,255,0.3)',fontSize:9,cursor:'pointer'}}>Skip →</button>
          </div>
        )}

        {/* ── Room Info Panel ── */}
        {roomInfo && showHUD && (
          <div style={{...css.hud,bottom:16,left:16}}>
            <p style={{...css.label,marginBottom:6}}>Room Intel</p>
            {[{k:'Type',v:roomInfo.room_type},{k:'Style',v:roomInfo.current_style},
              {k:'Light',v:roomInfo.natural_light},{k:'Target',v:targetStyle}]
              .filter(x=>x.v).map(x => (
              <div key={x.k} style={{display:'flex',justifyContent:'space-between',gap:16,
                padding:'3px 0',borderBottom:'1px solid rgba(255,255,255,0.03)'}}>
                <span style={css.label}>{x.k}</span>
                <span style={{fontSize:10,fontWeight:600,color:'#fff',textTransform:'capitalize'}}>{x.v}</span>
              </div>
            ))}
          </div>
        )}

        {/* ── Controls Help ── */}
        <div style={{...css.hud,bottom:16,right:16}}>
          <p style={{...css.label,marginBottom:6}}>
            {mode === 'fps' ? 'FPS Controls' : 'Controls'}
          </p>
          {(mode === 'fps' ? [
            {k:'W A S D',d:'Move',c:'#4466ff'},{k:'Mouse',d:'Look',c:'#44cc88'},
            {k:'Space',d:'Up',c:'#aa66ff'},{k:'Shift',d:'Down',c:'#ff6644'},
          ] : [
            {k:'Left Drag',d:'Rotate',c:'#4466ff'},{k:'Right Drag',d:'Pan',c:'#44cc88'},
            {k:'Scroll',d:'Zoom',c:'#aa66ff'},
          ]).map(x => (
            <div key={x.k} style={{display:'flex',alignItems:'center',gap:6,marginBottom:3}}>
              <span style={{fontSize:8,fontFamily:'monospace',padding:'1px 4px',
                background:'rgba(255,255,255,0.04)',borderRadius:3,color:x.c}}>{x.k}</span>
              <span style={{fontSize:9,color:'rgba(255,255,255,0.3)'}}>{x.d}</span>
            </div>
          ))}
        </div>

        {/* ── Mode Badge ── */}
        <div style={{position:'absolute',top:12,left:'50%',transform:'translateX(-50%)',
          padding:'5px 14px',background:'rgba(6,8,18,0.7)',border:'1px solid rgba(255,255,255,0.05)',
          borderRadius:16,backdropFilter:'blur(12px)',display:'flex',alignItems:'center',gap:6,zIndex:10}}>
          <div style={{width:5,height:5,borderRadius:'50%',
            background:mode==='fps'?'#44cc88':mode==='cinematic'?'#ff8844':'#4466ff',
            boxShadow:`0 0 6px ${mode==='fps'?'#44cc88':mode==='cinematic'?'#ff8844':'#4466ff'}`}} />
          <span style={{fontSize:9,fontWeight:600,color:'rgba(255,255,255,0.5)',
            fontFamily:'monospace',textTransform:'uppercase',letterSpacing:'0.08em'}}>
            {mode === 'cinematic' ? 'AR Walkthrough' : mode === 'fps' ? 'VR First-Person' : '3D Orbit'}
          </span>
        </div>
      </div>

      <style>{`
        @keyframes prog { from{width:0%} to{width:100%} }
      `}</style>
    </div>
  );
};

export default RoomWalkthrough;
