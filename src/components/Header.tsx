import { useState } from 'react';
import { Settings, Cpu, Play, Loader2, Link, CheckCircle, AlertCircle, ChevronDown } from 'lucide-react';
import type { Language, GPUType } from '../types';
import { LANGUAGE_OPTIONS, GPU_OPTIONS } from '../types';

interface HeaderProps {
  language: Language;
  onLanguageChange: (lang: Language) => void;
  onSettingsClick: () => void;
  onRun: () => void;
  isRunning: boolean;
  isConfigured: boolean;
  gpu: GPUType;
  onGpuChange: (gpu: GPUType) => void;
  endpoint: string | null;
  onEndpointChange: (endpoint: string) => void;
  deploymentStatus: 'idle' | 'checking' | 'deploying' | 'deployed' | 'error';
}

export function Header({
  language,
  onLanguageChange,
  onSettingsClick,
  onRun,
  isRunning,
  isConfigured,
  gpu,
  onGpuChange,
  endpoint,
  onEndpointChange,
  deploymentStatus,
}: HeaderProps) {
  const [showEndpointInput, setShowEndpointInput] = useState(false);
  const [tempEndpoint, setTempEndpoint] = useState(endpoint || '');

  const handleEndpointSave = () => {
    onEndpointChange(tempEndpoint);
    setShowEndpointInput(false);
  };

  return (
    <header
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 24px',
        background: 'linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%)',
        borderBottom: '1px solid var(--border-color)',
        position: 'relative',
      }}
    >
      {/* Left section - Logo and Language */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div 
            style={{ 
              background: 'var(--gradient-accent)',
              borderRadius: '10px',
              padding: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 12px var(--accent-glow)',
            }}
          >
            <Cpu size={20} color="white" />
          </div>
          <span style={{ fontSize: '20px', fontWeight: 700, letterSpacing: '-0.5px' }}>
            Kernel<span style={{ color: 'var(--accent)' }}>IDE</span>
          </span>
        </div>

        <div style={{ height: '24px', width: '1px', background: 'var(--border-color)' }} />

        <select
          className="select"
          value={language}
          onChange={(e) => onLanguageChange(e.target.value as Language)}
          style={{ minWidth: '160px' }}
        >
          {LANGUAGE_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      {/* Center section - GPU and Endpoint */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {/* GPU Selection */}
        <div style={{ position: 'relative' }}>
          <select
            className="select"
            value={gpu}
            onChange={(e) => onGpuChange(e.target.value as GPUType)}
            style={{ 
              minWidth: '180px',
              background: 'var(--bg-tertiary)',
              paddingRight: '40px',
            }}
          >
            {GPU_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label} ({opt.vram})
              </option>
            ))}
          </select>
        </div>

        {/* Endpoint Status */}
        <div style={{ position: 'relative' }}>
          <button
            className="btn-secondary"
            onClick={() => {
              setTempEndpoint(endpoint || '');
              setShowEndpointInput(!showEndpointInput);
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 14px',
            }}
          >
            <Link size={14} />
            <span style={{ maxWidth: '150px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {endpoint ? 'Connected' : 'Set Endpoint'}
            </span>
            {deploymentStatus === 'deployed' && <CheckCircle size={14} color="var(--success)" />}
            {deploymentStatus === 'error' && <AlertCircle size={14} color="var(--error)" />}
            {deploymentStatus === 'idle' && !endpoint && <AlertCircle size={14} color="var(--warning)" />}
            <ChevronDown size={14} />
          </button>

          {/* Endpoint Input Dropdown */}
          {showEndpointInput && (
            <div
              style={{
                position: 'absolute',
                top: '100%',
                left: '50%',
                transform: 'translateX(-50%)',
                marginTop: '8px',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: '12px',
                padding: '16px',
                width: '400px',
                zIndex: 100,
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
              }}
            >
              <label style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px', display: 'block' }}>
                Modal Endpoint URL
              </label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <input
                  className="input"
                  type="text"
                  placeholder="https://your-workspace--kernelide-executor-api.modal.run"
                  value={tempEndpoint}
                  onChange={(e) => setTempEndpoint(e.target.value)}
                  style={{ flex: 1, fontSize: '13px' }}
                />
                <button className="btn-primary" onClick={handleEndpointSave} style={{ padding: '8px 16px' }}>
                  Save
                </button>
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px' }}>
                Deploy with: <code style={{ background: 'var(--bg-tertiary)', padding: '2px 6px', borderRadius: '4px' }}>modal deploy modal_executor.py</code>
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Right section - Run and Settings */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <button
          className="btn-primary"
          onClick={onRun}
          disabled={isRunning || !isConfigured}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            minWidth: '120px',
            justifyContent: 'center',
          }}
        >
          {isRunning ? (
            <>
              <Loader2 size={16} className="spin" />
              Running...
            </>
          ) : (
            <>
              <Play size={16} fill="white" />
              Run
            </>
          )}
        </button>

        <button
          className="btn-icon"
          onClick={onSettingsClick}
          title="Settings"
        >
          <Settings size={18} />
        </button>
      </div>

      {/* Click outside to close endpoint input */}
      {showEndpointInput && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 99,
          }}
          onClick={() => setShowEndpointInput(false)}
        />
      )}
    </header>
  );
}
