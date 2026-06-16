import React from 'react';
import { motion } from 'framer-motion';

const SpatialInsights = ({ depthUrl, vlmAnalysis, reasoning }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-6xl mx-auto mt-12">
      {/* Geometry Panel */}
      <motion.div 
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="glass-card p-6 border border-white/5 relative overflow-hidden"
      >
        <div className="absolute top-0 right-0 p-3 opacity-10 font-mono text-[60px] select-none">3D</div>
        <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-accent-primary mb-6 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-primary animate-pulse" />
            Spatial Geometry Telemetry
        </h3>
        
        <div className="relative aspect-video rounded-xl overflow-hidden border border-white/10 bg-black group">
          {depthUrl ? (
            <img 
              src={depthUrl} 
              alt="Depth Map" 
              className="w-full h-full object-cover mix-blend-screen opacity-80 group-hover:opacity-100 transition-opacity"
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-text-muted">
              <div className="w-8 h-8 border-2 border-white/10 border-t-accent-primary rounded-full animate-spin mb-3" />
              <span className="text-[10px] font-mono">RECONSTRUCTING MESH...</span>
            </div>
          )}
          <div className="absolute inset-0 pointer-events-none border-[10px] border-black/20" />
          <div className="absolute bottom-4 left-4 flex gap-2">
             <span className="px-2 py-1 bg-black/60 text-[8px] font-mono text-white rounded border border-white/10">Z-PLANE ANALYSIS</span>
             <span className="px-2 py-1 bg-accent-primary/20 text-[8px] font-mono text-accent-primary rounded border border-accent-primary/20 uppercase">{vlmAnalysis?.room_size_estimate || 'Standard'} SCALE</span>
          </div>
        </div>
        
        <div className="mt-6 grid grid-cols-3 gap-4">
           {[
             { label: 'Surface Variance', val: 'Low' },
             { label: 'Occlusion Delta', val: '0.42m' },
             { label: 'Neural Fidelity', val: '98.4%' }
           ].map(stat => (
             <div key={stat.label} className="text-center">
                <p className="text-[8px] text-text-muted uppercase mb-1">{stat.label}</p>
                <p className="text-xs font-bold text-white font-mono">{stat.val}</p>
             </div>
           ))}
        </div>
      </motion.div>

      {/* Reasoning Panel */}
      <motion.div 
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="glass-card p-6 border border-white/5"
      >
        <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-accent-secondary mb-6 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-secondary" />
            Neural Design Reasoning
        </h3>
        
        <div className="space-y-6">
          <div className="p-4 bg-white/5 rounded-xl border border-white/5">
            <p className="text-[10px] font-mono text-text-muted uppercase mb-2">Architectural Logic</p>
            <p className="text-sm text-white leading-relaxed italic">
              "{reasoning || 'Analyzing spatial constraints to optimize furniture placement and lighting contrast...'}"
            </p>
          </div>
          
          <div className="space-y-4">
            <p className="text-[10px] font-mono text-text-muted uppercase">Detected Style DNA</p>
            <div className="flex flex-wrap gap-2">
              {(vlmAnalysis?.current_style || 'Detecting').split(',').map(tag => (
                <span key={tag} className="px-3 py-1 bg-white/5 border border-white/10 rounded-full text-[10px] text-white font-medium capitalize">
                  {tag.trim()}
                </span>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 pt-4 border-t border-white/5">
             <div>
                <p className="text-[9px] text-text-muted uppercase mb-1">Illumination Mode</p>
                <p className="text-xs font-bold text-white capitalize">{vlmAnalysis?.natural_light || 'Ambient'}</p>
             </div>
             <div>
                <p className="text-[9px] text-text-muted uppercase mb-1">Spatial Condition</p>
                <p className="text-xs font-bold text-white capitalize">{vlmAnalysis?.condition || 'Fair'}</p>
             </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default SpatialInsights;
