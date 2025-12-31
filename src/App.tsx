import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { Editor } from './components/Editor';
import { OutputPanel } from './components/OutputPanel';
import { SettingsModal } from './components/SettingsModal';
import type { Language, Settings, ExecutionResult } from './types';
import { loadSettings, saveSettings, defaultCode, getStoredEndpoint, setStoredEndpoint } from './store';
import { executeCode, checkDeployment } from './services/modal';

function App() {
  const [language, setLanguage] = useState<Language>('cuda');
  const [code, setCode] = useState<string>(defaultCode.cuda);
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [showSettings, setShowSettings] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<ExecutionResult | null>(null);
  const [deploymentStatus, setDeploymentStatus] = useState<
    'idle' | 'checking' | 'deploying' | 'deployed' | 'error'
  >('idle');
  const [deploymentError, setDeploymentError] = useState<string>();
  const [endpoint, setEndpoint] = useState<string | null>(getStoredEndpoint);

  // Check if API is configured
  const isConfigured = Boolean(settings.apiKey && settings.apiSecret && endpoint);

  // Show settings on first load if not configured
  useEffect(() => {
    if (!settings.apiKey || !settings.apiSecret) {
      setShowSettings(true);
    } else if (endpoint) {
      // Check if deployment is still valid
      checkDeploymentStatus();
    }
  }, []);

  const checkDeploymentStatus = useCallback(async () => {
    if (!settings.apiKey || !settings.apiSecret) return;

    setDeploymentStatus('checking');
    const result = await checkDeployment(settings);

    if (result.deployed && result.endpoint) {
      setDeploymentStatus('deployed');
      setEndpoint(result.endpoint);
      setStoredEndpoint(result.endpoint);
    } else {
      setDeploymentStatus('idle');
    }
  }, [settings]);

  const handleLanguageChange = (lang: Language) => {
    setLanguage(lang);
    // Load default code for the language if current code is default
    if (code === defaultCode[language]) {
      setCode(defaultCode[lang]);
    }
  };

  const handleSettingsSave = (newSettings: Settings) => {
    setSettings(newSettings);
    saveSettings(newSettings);
  };

  const handleDeploy = async () => {
    setDeploymentStatus('deploying');
    setDeploymentError(undefined);

    // For now, show manual deployment instructions
    // In a full implementation, this would use Modal's API or a backend service
    setDeploymentStatus('error');
    setDeploymentError(
      'Please deploy manually using Modal CLI. Save the executor code and run: modal deploy executor.py'
    );
  };

  const handleRun = async () => {
    if (!endpoint) {
      setResult({
        success: false,
        stdout: '',
        stderr: '',
        error: 'No endpoint configured. Please deploy the executor first.',
      });
      return;
    }

    setIsRunning(true);
    setResult(null);

    try {
      const executionResult = await executeCode(code, language, settings, endpoint);
      setResult(executionResult);
    } catch (e) {
      setResult({
        success: false,
        stdout: '',
        stderr: '',
        error: e instanceof Error ? e.message : 'Unknown error occurred',
      });
    } finally {
      setIsRunning(false);
    }
  };

  // Keyboard shortcut for running code
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        if (isConfigured && !isRunning) {
          handleRun();
        }
      }
      if ((e.metaKey || e.ctrlKey) && e.key === ',') {
        e.preventDefault();
        setShowSettings(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isConfigured, isRunning, code, language, settings, endpoint]);

  const handleGpuChange = (gpu: typeof settings.gpu) => {
    const newSettings = { ...settings, gpu };
    setSettings(newSettings);
    saveSettings(newSettings);
  };

  const handleEndpointChange = (newEndpoint: string) => {
    setEndpoint(newEndpoint);
    if (newEndpoint) {
      setStoredEndpoint(newEndpoint);
      checkDeploymentStatus();
    }
  };

  return (
    <>
      <Header
        language={language}
        onLanguageChange={handleLanguageChange}
        onSettingsClick={() => setShowSettings(true)}
        onRun={handleRun}
        isRunning={isRunning}
        isConfigured={isConfigured}
        gpu={settings.gpu}
        onGpuChange={handleGpuChange}
        endpoint={endpoint}
        onEndpointChange={handleEndpointChange}
        deploymentStatus={deploymentStatus}
      />

      <main
        style={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns: '1fr 400px',
          overflow: 'hidden',
        }}
      >
        <div style={{ overflow: 'hidden' }}>
          <Editor language={language} code={code} onChange={setCode} />
        </div>

        <OutputPanel result={result} isRunning={isRunning} />
      </main>

      <footer
        style={{
          padding: '10px 24px',
          background: 'linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%)',
          borderTop: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '12px',
          color: 'var(--text-muted)',
        }}
      >
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          <span style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '6px',
            padding: '4px 10px',
            background: 'var(--bg-tertiary)',
            borderRadius: '6px',
            border: '1px solid var(--border-color)',
          }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--success)' }} />
            {settings.gpu}
          </span>
          <span>Timeout: {settings.timeout}s</span>
        </div>
        <div style={{ display: 'flex', gap: '16px', color: 'var(--text-secondary)' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <kbd style={{ 
              padding: '2px 6px', 
              background: 'var(--bg-tertiary)', 
              borderRadius: '4px',
              border: '1px solid var(--border-color)',
              fontSize: '11px',
            }}>⌘</kbd>
            <kbd style={{ 
              padding: '2px 6px', 
              background: 'var(--bg-tertiary)', 
              borderRadius: '4px',
              border: '1px solid var(--border-color)',
              fontSize: '11px',
            }}>Enter</kbd>
            <span style={{ marginLeft: '4px' }}>Run</span>
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <kbd style={{ 
              padding: '2px 6px', 
              background: 'var(--bg-tertiary)', 
              borderRadius: '4px',
              border: '1px solid var(--border-color)',
              fontSize: '11px',
            }}>⌘</kbd>
            <kbd style={{ 
              padding: '2px 6px', 
              background: 'var(--bg-tertiary)', 
              borderRadius: '4px',
              border: '1px solid var(--border-color)',
              fontSize: '11px',
            }}>,</kbd>
            <span style={{ marginLeft: '4px' }}>Settings</span>
          </span>
        </div>
      </footer>

      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        settings={settings}
        onSave={handleSettingsSave}
        deploymentStatus={deploymentStatus}
        deploymentError={deploymentError}
        onDeploy={handleDeploy}
      />

      {/* Manual Endpoint Input Modal - shown when deployment fails */}
      {deploymentStatus === 'error' && showSettings && (
        <div
          style={{
            position: 'fixed',
            bottom: '20px',
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'var(--bg-tertiary)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            padding: '16px 20px',
            maxWidth: '600px',
            zIndex: 1001,
          }}
        >
          <h4 style={{ marginBottom: '12px', color: 'var(--text-primary)' }}>
            Manual Endpoint Configuration
          </h4>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '12px' }}>
            After deploying with Modal CLI, enter your endpoint URL:
          </p>
          <div style={{ display: 'flex', gap: '8px' }}>
            <input
              className="input"
              type="text"
              placeholder="https://your-workspace--kernelide-executor-api.modal.run"
              value={endpoint || ''}
              onChange={(e) => {
                setEndpoint(e.target.value);
                if (e.target.value) {
                  setStoredEndpoint(e.target.value);
                }
              }}
              style={{ flex: 1 }}
            />
            <button
              className="btn-primary"
              onClick={() => {
                if (endpoint) {
                  checkDeploymentStatus();
                }
              }}
            >
              Verify
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
