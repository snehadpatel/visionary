import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, Palette, CheckCircle, ArrowRight, ShoppingCart, Home, Sparkles, Loader2, RefreshCw } from 'lucide-react';
import './index.css';

const STYLES = [
  { id: 'scandinavian', name: 'Scandinavian', color: '#E0F2FE' },
  { id: 'bohemian', name: 'Bohemian', color: '#FFEDD5' },
  { id: 'industrial', name: 'Industrial', color: '#F1F5F9' },
  { id: 'minimalist', name: 'Minimalist', color: '#FFFFFF' },
];

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [style, setStyle] = useState('scandinavian');
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [jobId, setJobId] = useState(null);
  const [result, setResult] = useState(null);

  const onFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handleRedesign = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setResult(null);
    setJobId(null);
    setStatus('Queueing...');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('style', style);
    formData.append('prompt', prompt);

    try {
      const response = await axios.post('http://localhost:8000/redesign', formData);
      setJobId(response.data.job_id);
    } catch (err) {
      console.error(err);
      alert('Redesign failed. Ensure backend is running.');
      setLoading(false);
    }
  };

  // Polling Effect
  useEffect(() => {
    let interval;
    if (jobId && loading) {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`http://localhost:8000/status/${jobId}`);
          setStatus(res.data.status);
          if (res.data.status === 'done') {
            setResult(res.data.result);
            setLoading(false);
            clearInterval(interval);
          } else if (res.data.status === 'error') {
            alert('Error: ' + res.data.error);
            setLoading(false);
            clearInterval(interval);
          }
        } catch (err) {
          console.error('Polling error', err);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [jobId, loading]);

  return (
    <div className="app-container">
      <header className="header">
        <div className="logo-section">
          <div className="logo-box">
            <Home size={20} />
          </div>
          <span className="logo-text">Visionary</span>
        </div>
        <div style={{display: 'flex', gap: '1rem', alignItems: 'center'}}>
          <div className="status-indicator">
            <div className="dot pulse" style={{background: '#10B981'}} />
            <span>M4 Neural Engine Active</span>
          </div>
          <button className="style-btn active" style={{padding: '0.6rem 1.2rem'}}>Phase 6 Beta</button>
        </div>
      </header>

      <main className="main-content">
        <section className="animate">
          <h1 className="hero-title">
            Redesign your room with <span className="hero-highlight">High-Fidelity AI</span>
          </h1>
          <p style={{color: '#64748B', fontSize: '1.1rem', marginBottom: '2.5rem', maxWidth: '500px'}}>
            Upload a photo, pick a style, and watch Stable Diffusion + ControlNet transform your space while preserving its unique 3D layout.
          </p>

          <div className="card">
            <div className={`upload-zone ${preview ? 'active' : ''}`} onClick={() => !preview && document.getElementById('file-input').click()}>
              {preview ? (
                <div style={{position: 'relative'}}>
                  <img src={preview} alt="Preview" className="preview-img" />
                  <button 
                    onClick={(e) => { e.stopPropagation(); setSelectedFile(null); setPreview(null); }}
                    style={{position: 'absolute', top: '-10px', right: '-10px', background: 'white', border: '1px solid #ddd', borderRadius: '50%', width: '30px', height: '30px', cursor: 'pointer', boxShadow: '0 2px 5px rgba(0,0,0,0.2)'}}
                  >×</button>
                </div>
              ) : (
                <>
                  <div style={{background: '#EEF2FF', color: '#4F46E5', width: '60px', height: '60px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyCenter: 'center', margin: '0 auto 1rem'}}>
                    <Upload size={24} style={{margin: 'auto'}} />
                  </div>
                  <p style={{fontWeight: 700, color: '#334155'}}>Click to upload photo</p>
                  <p style={{fontSize: '0.8rem', color: '#94A3B8', marginTop: '0.25rem'}}>JPG, PNG or WEBP</p>
                </>
              )}
              <input type="file" id="file-input" className="hidden" style={{display: 'none'}} onChange={onFileChange} accept="image/*" />
            </div>

            <div style={{marginTop: '2rem'}}>
              <p className="input-label">Target Aesthetic</p>
              <div className="style-grid">
                {STYLES.map(s => (
                  <div 
                    key={s.id} 
                    className={`style-btn ${style === s.id ? 'active' : ''}`}
                    onClick={() => setStyle(s.id)}
                  >
                    <div className="dot" style={{background: s.color}} />
                    {s.name}
                  </div>
                ))}
              </div>
            </div>

            <div style={{marginTop: '1.5rem'}}>
              <p className="input-label">Describe your vision (optional)</p>
              <input 
                type="text" 
                placeholder="e.g. Add a velvet blue sofa and warm lighting" 
                className="text-input"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </div>

            <button 
              className="btn-primary" 
              style={{marginTop: '2.5rem'}}
              disabled={!selectedFile || loading}
              onClick={handleRedesign}
            >
              {loading ? (
                <><Loader2 className="spinner" size={18} /> {status === 'processing' ? 'Synthesizing...' : 'Reconstructing Scene...'}</>
              ) : (
                <><Sparkles size={18} /> Generate HD Redesign</>
              )}
            </button>
          </div>
        </section>

        <section className="result-viewport animate">
          {!result && !loading && (
            <div className="glass-card">
               <Palette size={48} style={{color: '#818CF8', opacity: 0.5, marginBottom: '1.5rem'}} />
               <h2 style={{fontSize: '1.75rem', marginBottom: '1rem'}}>Studio Ready</h2>
               <p style={{color: '#94A3B8', maxWidth: '300px', margin: '0 auto'}}>
                 Our neural back-projector is waiting for your input to reconstruct the 3D room.
               </p>
            </div>
          )}

          {loading && (
            <div style={{textAlign: 'center'}} className="animate">
              <div className="pipeline-steps">
                <div className={`step-item ${status === 'queued' ? 'active' : 'done'}`}>
                  <div className="step-num">{status === 'queued' ? <RefreshCw className="spinner" size={14} /> : <CheckCircle size={14} />}</div>
                  <span>Queueing Job</span>
                </div>
                <div className={`step-item ${status === 'processing' ? 'active' : ''}`}>
                  <div className="step-num">{status === 'processing' ? <Loader2 className="spinner" size={14} /> : '2'}</div>
                  <span>3D Scene & SD Synthesis</span>
                </div>
              </div>
              <div className="pulse-loader" style={{margin: '3rem auto'}} />
              <h3 style={{fontSize: '1.5rem', textTransform: 'capitalize'}}>{status}...</h3>
              <p style={{color: '#64748B', marginTop: '0.5rem', fontSize: '0.9rem'}}>This can take up to 20-30 seconds on M4 GPU</p>
            </div>
          )}

          {result && (
            <div className="animate">
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '1.5rem'}}>
                  <h2 style={{color: '#A5B4FC'}}>Redesign Complete</h2>
                  <span className="badge">
                    {result.replacements.length} neural Swaps
                  </span>
               </div>
               
               {/* Visualized Result (The "Wow" factor) */}
               {result.visualized_url && (
                 <div className="visualization-container">
                   <p className="section-subtitle">Photorealistic Neural Transformation</p>
                   <img src={`http://localhost:8000${result.visualized_url}`} className="visualized-img" alt="Redesigned Room" />
                 </div>
               )}

               <div style={{display: 'flex', gap: '1rem', marginTop: '2rem'}}>
                 <div style={{flex: 1}}>
                    <p className="section-subtitle">Original Reconstruction</p>
                    <img src={`http://localhost:8000${result.image_url}`} className="result-img" />
                 </div>
                 
                 <div style={{flex: 1}}>
                   <div className="inspiration-panel" style={{height: '100%', margin: 0}}>
                      <p className="section-subtitle">Extracted Style Palette</p>
                      <div className="palette-strip">
                        {result.inspiration.palette.map((c, i) => (
                          <div key={i} className="color-swatch" style={{background: c}} title={c} />
                        ))}
                      </div>
                      <div className="replacements-mini-list">
                        {result.replacements.slice(0, 3).map((r, i) => (
                          <div key={i} className="mini-item">
                            <span>{r.replacement_name}</span>
                            <ShoppingCart size={14} />
                          </div>
                        ))}
                        {result.replacements.length > 3 && (
                          <div className="mini-item" style={{justifyContent: 'center', opacity: 0.6}}>
                            + {result.replacements.length - 3} more items
                          </div>
                        )}
                      </div>
                   </div>
                 </div>
               </div>
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        Apple M4 Neuro-Enhanced Pipeline • Photorealistic Phase 6 Live
      </footer>
    </div>
  );
}
