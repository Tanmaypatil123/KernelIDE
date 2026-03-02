import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { Editor } from './components/Editor';
import { OutputPanel } from './components/OutputPanel';
import { SettingsModal } from './components/SettingsModal';
import type { Language, Settings, ExecutionResult } from './types';
import { loadSettings, saveSettings, defaultCode, getStoredEndpoint, setStoredEndpoint, clearStoredEndpoint } from './store';
import { executeCode, verifyEndpoint } from './services/modal';

function App() {
  const [language, setLanguage] = useState<Language>('cuda');
  const [code, setCode] = useState<string>(defaultCode.cuda);
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [showSettings, setShowSettings] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<ExecutionResult | null>(null);
  const [endpoint, setEndpoint] = useState<string | null>(getStoredEndpoint);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'checking' | 'connected'>('disconnected');

  const isConfigured = Boolean(endpoint) && connectionStatus === 'connected';

  const checkConnection = useCallback(async (url: string) => {
    setConnectionStatus('checking');
    const ok = await verifyEndpoint(url);
    setConnectionStatus(ok ? 'connected' : 'disconnected');
    return ok;
  }, []);

  // Verify stored endpoint on load
  useEffect(() => {
    if (endpoint) {
      checkConnection(endpoint);
    }
  }, []);

  const handleEndpointChange = async (newEndpoint: string) => {
    const trimmed = newEndpoint.trim().replace(/\/+$/, ''); // trim trailing slashes
    if (!trimmed) {
      setEndpoint(null);
      clearStoredEndpoint();
      setConnectionStatus('disconnected');
      return;
    }
    setEndpoint(trimmed);
    setStoredEndpoint(trimmed);
    await checkConnection(trimmed);
  };

  const handleLanguageChange = (lang: Language) => {
    setLanguage(lang);
    if (code === defaultCode[language]) {
      setCode(defaultCode[lang]);
    }
  };

  const handleSettingsSave = (newSettings: Settings) => {
    setSettings(newSettings);
    saveSettings(newSettings);
  };

  const handleRun = async () => {
    if (!endpoint) {
      setResult({
        success: false,
        stdout: '',
        stderr: '',
        error: 'No endpoint configured. Paste your Modal endpoint URL in the header.',
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

  // Keyboard shortcuts
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
        connectionStatus={connectionStatus}
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
            }}>Ctrl</kbd>
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
            }}>Ctrl</kbd>
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
        endpoint={endpoint}
        onEndpointChange={handleEndpointChange}
        connectionStatus={connectionStatus}
      />
    </>
  );
}

export default App;
