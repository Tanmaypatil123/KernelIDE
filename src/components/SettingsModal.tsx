import { useState, useEffect } from 'react';
import { X, Eye, EyeOff, CheckCircle, AlertCircle, Loader2, ExternalLink } from 'lucide-react';
import type { Settings, GPUType } from '../types';
import { GPU_OPTIONS } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: Settings;
  onSave: (settings: Settings) => void;
  deploymentStatus: 'idle' | 'checking' | 'deploying' | 'deployed' | 'error';
  deploymentError?: string;
  onDeploy: () => void;
}

export function SettingsModal({
  isOpen,
  onClose,
  settings,
  onSave,
  deploymentStatus,
  deploymentError,
  onDeploy,
}: SettingsModalProps) {
  const [localSettings, setLocalSettings] = useState<Settings>(settings);
  const [showApiKey, setShowApiKey] = useState(false);
  const [showApiSecret, setShowApiSecret] = useState(false);

  useEffect(() => {
    setLocalSettings(settings);
  }, [settings]);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave(localSettings);
  };

  const handleDeployAndSave = () => {
    onSave(localSettings);
    onDeploy();
  };

  const isConfigured = localSettings.apiKey && localSettings.apiSecret;

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
          {/* Modal API Credentials */}
          <section>
            <h3
              style={{
                fontSize: '14px',
                fontWeight: 600,
                marginBottom: '12px',
                color: 'var(--text-secondary)',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}
            >
              Modal.com Credentials
            </h3>
            <p
              style={{
                fontSize: '13px',
                color: 'var(--text-muted)',
                marginBottom: '12px',
              }}
            >
              Get your API credentials from{' '}
              <a
                href="https://modal.com/settings"
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: 'var(--accent)', textDecoration: 'none' }}
              >
                modal.com/settings <ExternalLink size={12} style={{ display: 'inline' }} />
              </a>
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <label
                  style={{
                    fontSize: '13px',
                    color: 'var(--text-secondary)',
                    display: 'block',
                    marginBottom: '6px',
                  }}
                >
                  API Key (Token ID)
                </label>
                <div style={{ position: 'relative' }}>
                  <input
                    className="input"
                    type={showApiKey ? 'text' : 'password'}
                    value={localSettings.apiKey}
                    onChange={(e) =>
                      setLocalSettings({ ...localSettings, apiKey: e.target.value })
                    }
                    placeholder="ak-..."
                    style={{ paddingRight: '40px' }}
                  />
                  <button
                    onClick={() => setShowApiKey(!showApiKey)}
                    style={{
                      position: 'absolute',
                      right: '8px',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      background: 'none',
                      color: 'var(--text-muted)',
                      padding: '4px',
                    }}
                  >
                    {showApiKey ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              <div>
                <label
                  style={{
                    fontSize: '13px',
                    color: 'var(--text-secondary)',
                    display: 'block',
                    marginBottom: '6px',
                  }}
                >
                  API Secret (Token Secret)
                </label>
                <div style={{ position: 'relative' }}>
                  <input
                    className="input"
                    type={showApiSecret ? 'text' : 'password'}
                    value={localSettings.apiSecret}
                    onChange={(e) =>
                      setLocalSettings({ ...localSettings, apiSecret: e.target.value })
                    }
                    placeholder="as-..."
                    style={{ paddingRight: '40px' }}
                  />
                  <button
                    onClick={() => setShowApiSecret(!showApiSecret)}
                    style={{
                      position: 'absolute',
                      right: '8px',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      background: 'none',
                      color: 'var(--text-muted)',
                      padding: '4px',
                    }}
                  >
                    {showApiSecret ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>
            </div>
          </section>

          {/* Deployment Status */}
          <section>
            <h3
              style={{
                fontSize: '14px',
                fontWeight: 600,
                marginBottom: '12px',
                color: 'var(--text-secondary)',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}
            >
              Deployment Status
            </h3>

            <div
              style={{
                padding: '12px',
                background: 'var(--bg-tertiary)',
                borderRadius: '6px',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
              }}
            >
              {deploymentStatus === 'idle' && (
                <>
                  <AlertCircle size={20} color="var(--warning)" />
                  <span style={{ color: 'var(--text-secondary)' }}>
                    Not deployed. Click "Deploy" to set up the executor on your Modal account.
                  </span>
                </>
              )}
              {deploymentStatus === 'checking' && (
                <>
                  <Loader2 size={20} color="var(--accent)" className="spin" />
                  <span>Checking deployment status...</span>
                </>
              )}
              {deploymentStatus === 'deploying' && (
                <>
                  <Loader2 size={20} color="var(--accent)" className="spin" />
                  <span>Deploying executor to Modal...</span>
                </>
              )}
              {deploymentStatus === 'deployed' && (
                <>
                  <CheckCircle size={20} color="var(--success)" />
                  <span style={{ color: 'var(--success)' }}>
                    Deployed and ready!
                  </span>
                </>
              )}
              {deploymentStatus === 'error' && (
                <>
                  <AlertCircle size={20} color="var(--error)" />
                  <span style={{ color: 'var(--error)' }}>
                    {deploymentError || 'Deployment failed'}
                  </span>
                </>
              )}
            </div>

            {isConfigured && deploymentStatus !== 'deployed' && (
              <button
                className="btn-primary"
                onClick={handleDeployAndSave}
                disabled={deploymentStatus === 'deploying' || deploymentStatus === 'checking'}
                style={{ marginTop: '12px', width: '100%' }}
              >
                {deploymentStatus === 'deploying' ? 'Deploying...' : 'Deploy Executor'}
              </button>
            )}
          </section>

          {/* GPU Selection */}
          <section>
            <h3
              style={{
                fontSize: '14px',
                fontWeight: 600,
                marginBottom: '12px',
                color: 'var(--text-secondary)',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}
            >
              GPU Selection
            </h3>

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
            <h3
              style={{
                fontSize: '14px',
                fontWeight: 600,
                marginBottom: '12px',
                color: 'var(--text-secondary)',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}
            >
              Execution Timeout
            </h3>

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
              <span
                style={{
                  minWidth: '60px',
                  textAlign: 'right',
                  color: 'var(--text-secondary)',
                }}
              >
                {localSettings.timeout}s
              </span>
            </div>
            <p
              style={{
                fontSize: '12px',
                color: 'var(--text-muted)',
                marginTop: '8px',
              }}
            >
              Maximum execution time before the kernel is terminated (5-300 seconds)
            </p>
          </section>

          {/* Save Button */}
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
