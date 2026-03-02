import { useState, useEffect } from 'react';
import { X, CheckCircle, AlertCircle, Loader2, ExternalLink } from 'lucide-react';
import type { Settings, GPUType } from '../types';
import { GPU_OPTIONS } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: Settings;
  onSave: (settings: Settings) => void;
  endpoint: string | null;
  onEndpointChange: (endpoint: string) => void;
  connectionStatus: 'disconnected' | 'checking' | 'connected';
}

export function SettingsModal({
  isOpen,
  onClose,
  settings,
  onSave,
  endpoint,
  onEndpointChange,
  connectionStatus,
}: SettingsModalProps) {
  const [localSettings, setLocalSettings] = useState<Settings>(settings);
  const [localEndpoint, setLocalEndpoint] = useState(endpoint || '');

  useEffect(() => {
    setLocalSettings(settings);
    setLocalEndpoint(endpoint || '');
  }, [settings, endpoint]);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave(localSettings);
    if (localEndpoint !== (endpoint || '')) {
      onEndpointChange(localEndpoint);
    }
    onClose();
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.8)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        className="panel"
        style={{
          width: '500px',
          maxHeight: '90vh',
          overflow: 'auto',
          padding: '24px',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '24px',
          }}
        >
          <h2 style={{ fontSize: '20px', fontWeight: 600 }}>Settings</h2>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              color: 'var(--text-secondary)',
              padding: '4px',
            }}
          >
            <X size={20} />
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Endpoint Configuration */}
          <section>
            <h3 style={sectionTitleStyle}>
              Modal Endpoint
            </h3>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '12px' }}>
              Deploy with <code style={codeStyle}>modal deploy modal_executor.py</code> then paste the URL below.{' '}
              <a
                href="https://modal.com/settings"
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: 'var(--accent)', textDecoration: 'none' }}
              >
                modal.com <ExternalLink size={11} style={{ display: 'inline' }} />
              </a>
            </p>

            <input
              className="input"
              type="text"
              placeholder="https://your-workspace--kernelide-executor-api.modal.run"
              value={localEndpoint}
              onChange={(e) => setLocalEndpoint(e.target.value)}
              style={{ marginBottom: '8px' }}
            />

            <div
              style={{
                padding: '10px 12px',
                background: 'var(--bg-tertiary)',
                borderRadius: '6px',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                fontSize: '13px',
              }}
            >
              {connectionStatus === 'connected' && (
                <>
                  <CheckCircle size={16} color="var(--success)" />
                  <span style={{ color: 'var(--success)' }}>Connected and ready</span>
                </>
              )}
              {connectionStatus === 'checking' && (
                <>
                  <Loader2 size={16} color="var(--accent)" className="spin" />
                  <span>Verifying endpoint...</span>
                </>
              )}
              {connectionStatus === 'disconnected' && (
                <>
                  <AlertCircle size={16} color={endpoint ? 'var(--error)' : 'var(--warning)'} />
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {endpoint ? 'Could not connect to endpoint' : 'No endpoint configured'}
                  </span>
                </>
              )}
            </div>
          </section>

          {/* GPU Selection */}
          <section>
            <h3 style={sectionTitleStyle}>GPU Selection</h3>
            <select
              className="select"
              value={localSettings.gpu}
              onChange={(e) =>
                setLocalSettings({ ...localSettings, gpu: e.target.value as GPUType })
              }
              style={{ width: '100%' }}
            >
              {GPU_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label} ({opt.vram}) - {opt.price}
                </option>
              ))}
            </select>
          </section>

          {/* Timeout */}
          <section>
            <h3 style={sectionTitleStyle}>Execution Timeout</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <input
                type="range"
                min="5"
                max="300"
                step="5"
                value={localSettings.timeout}
                onChange={(e) =>
                  setLocalSettings({ ...localSettings, timeout: Number(e.target.value) })
                }
                style={{ flex: 1, accentColor: 'var(--accent)' }}
              />
              <span style={{ minWidth: '60px', textAlign: 'right', color: 'var(--text-secondary)' }}>
                {localSettings.timeout}s
              </span>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '8px' }}>
              Maximum execution time before the kernel is terminated (5-300 seconds)
            </p>
          </section>

          <button className="btn-primary" onClick={handleSave} style={{ marginTop: '8px' }}>
            Save Settings
          </button>
        </div>

        <style>{`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          .spin {
            animation: spin 1s linear infinite;
          }
        `}</style>
      </div>
    </div>
  );
}

const sectionTitleStyle: React.CSSProperties = {
  fontSize: '14px',
  fontWeight: 600,
  marginBottom: '12px',
  color: 'var(--text-secondary)',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
};

const codeStyle: React.CSSProperties = {
  background: 'var(--bg-tertiary)',
  padding: '2px 6px',
  borderRadius: '4px',
  fontSize: '12px',
};
