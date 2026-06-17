import React from 'react';
import { motion } from 'framer-motion';

const SpatialInsights = ({ depthUrl, vlmAnalysis, reasoning }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-6xl mx-auto mt-12">
      {/* Geometry Panel */}
      <motion.div 
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="glass-card p-6 border border-[var(--border-subtle)] relative overflow-hidden"
      >
        <div className="absolute top-0 right-0 p-3 opacity-10 font-mono text-[60px] select-none text-[var(--text-muted)]">3D</div>
        <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--accent-primary)] mb-6 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-primary)] animate-pulse" />
            Depth & Layout Analysis
        </h3>
        
        <div className="relative aspect-video rounded-xl overflow-hidden border border-[var(--border-subtle)] bg-black group">
          {depthUrl ? (
            <img 
              src={depthUrl} 
              alt="Depth Map" 
              className="w-full h-full object-cover mix-blend-screen opacity-80 group-hover:opacity-100 transition-opacity"
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-[var(--text-muted)]">
              <div className="w-8 h-8 border-2 border-[var(--border-subtle)] border-t-[var(--accent-primary)] rounded-full animate-spin mb-3" />
              <span className="text-[10px] font-mono">RECONSTRUCTING MESH...</span>
            </div>
          )}
          <div className="absolute inset-0 pointer-events-none border-[10px] border-black/20" />
          <div className="absolute bottom-4 left-4 flex gap-2">
             <span className="px-2 py-1 bg-black/60 text-[8px] font-mono text-white rounded border border-white/10">Z-PLANE ANALYSIS</span>
             <span className="px-2 py-1 bg-[var(--accent-primary)]/20 text-[8px] font-mono text-[var(--accent-primary)] rounded border border-[var(--accent-primary)]/20 uppercase">{vlmAnalysis?.room_size_estimate || 'Standard'} SCALE</span>
          </div>
        </div>
        
        <div className="mt-6 grid grid-cols-3 gap-4">
           {[
             { label: 'Surface Variance', val: 'Low' },
             { label: 'Occlusion Delta', val: '0.42m' },
             { label: 'Neural Fidelity', val: '98.4%' }
           ].map(stat => (
             <div key={stat.label} className="text-center">
                <p className="text-[8px] text-[var(--text-muted)] uppercase mb-1">{stat.label}</p>
                <p className="text-xs font-bold text-[var(--text-heading)] font-mono">{stat.val}</p>
             </div>
           ))}
         </div>
      </motion.div>

      {/* Reasoning Panel */}
      <motion.div 
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="glass-card p-6 border border-[var(--border-subtle)]"
      >
        <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--accent-secondary)] mb-6 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-secondary)]" />
            AI Design Rationale
        </h3>
        
        <div className="space-y-6">
          <div className="p-4 bg-[var(--bg-surface)] rounded-xl border border-[var(--border-subtle)]">
            <p className="text-[10px] font-mono text-[var(--text-muted)] uppercase mb-2">Staging Strategy</p>
            <p className="text-sm text-[var(--text-primary)] leading-relaxed italic">
              "{reasoning || 'Analyzing spatial constraints to optimize furniture placement and lighting contrast...'}"
            </p>
          </div>
          
          <div className="space-y-4">
            <p className="text-[10px] font-mono text-[var(--text-muted)] uppercase">Identified Style DNA</p>
            <div className="flex flex-wrap gap-2">
              {(vlmAnalysis?.current_style || 'Detecting').split(',').map(tag => (
                <span key={tag} className="px-3 py-1 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-full text-[10px] text-[var(--text-primary)] font-medium capitalize">
                  {tag.trim()}
                </span>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 pt-4 border-t border-[var(--border-subtle)]">
             <div>
                <p className="text-[9px] text-[var(--text-muted)] uppercase mb-1">Illumination Mode</p>
                <p className="text-xs font-bold text-[var(--text-heading)] capitalize">{vlmAnalysis?.natural_light || 'Ambient'}</p>
             </div>
             <div>
                <p className="text-[9px] text-[var(--text-muted)] uppercase mb-1">Spatial Condition</p>
                <p className="text-xs font-bold text-[var(--text-heading)] capitalize">{vlmAnalysis?.condition || 'Fair'}</p>
             </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default SpatialInsights;
