import { Terminal, Clock, Cpu, AlertTriangle, CheckCircle } from 'lucide-react';
import type { ExecutionResult } from '../types';

interface OutputPanelProps {
  result: ExecutionResult | null;
  isRunning: boolean;
}

export function OutputPanel({ result, isRunning }: OutputPanelProps) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: 'var(--bg-secondary)',
        borderLeft: '1px solid var(--border-color)',
      }}
    >
      <div
        style={{
          padding: '12px 16px',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}
      >
        <Terminal size={16} color="var(--text-secondary)" />
        <span style={{ fontWeight: 500 }}>Output</span>
        {result && (
          <span
            style={{
              marginLeft: 'auto',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              fontSize: '12px',
              color: result.success ? 'var(--success)' : 'var(--error)',
            }}
          >
            {result.success ? (
              <CheckCircle size={14} />
            ) : (
              <AlertTriangle size={14} />
            )}
            {result.success ? 'Success' : 'Failed'}
          </span>
        )}
      </div>

      {/* Metrics bar */}
      {result && (result.executionTime !== undefined || result.gpuMetrics) && (
        <div
          style={{
            padding: '8px 16px',
            borderBottom: '1px solid var(--border-color)',
            display: 'flex',
            gap: '20px',
            fontSize: '12px',
            color: 'var(--text-secondary)',
            background: 'var(--bg-tertiary)',
          }}
        >
          {result.executionTime !== undefined && (
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Clock size={12} />
              {result.executionTime.toFixed(2)}s
            </span>
          )}
          {result.gpuMetrics?.memoryUsed && (
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Cpu size={12} />
              {result.gpuMetrics.memoryUsed}
            </span>
          )}
        </div>
      )}

      {/* Output content */}
      <div
        style={{
          flex: 1,
          overflow: 'auto',
          padding: '16px',
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          fontSize: '13px',
          lineHeight: '1.6',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {isRunning && (
          <div style={{ color: 'var(--text-muted)' }}>
            <span className="pulse">●</span> Executing kernel...
          </div>
        )}

        {!isRunning && !result && (
          <div style={{ color: 'var(--text-muted)' }}>
            Click "Run" to execute your kernel. Output will appear here.
          </div>
        )}

        {result && (
          <>
            {result.stdout && (
              <div style={{ color: 'var(--text-primary)' }}>{result.stdout}</div>
            )}
            {result.stderr && (
              <div
                style={{
                  color: 'var(--error)',
                  marginTop: result.stdout ? '12px' : 0,
                  paddingTop: result.stdout ? '12px' : 0,
                  borderTop: result.stdout
                    ? '1px solid var(--border-color)'
                    : 'none',
                }}
              >
                <div
                  style={{
                    fontSize: '11px',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    marginBottom: '8px',
                    color: 'var(--error)',
                    opacity: 0.8,
                  }}
                >
                  stderr
                </div>
                {result.stderr}
              </div>
            )}
            {result.error && (
              <div style={{ color: 'var(--error)' }}>
                <div
                  style={{
                    fontSize: '11px',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    marginBottom: '8px',
                    opacity: 0.8,
                  }}
                >
                  Error
                </div>
                {result.error}
              </div>
            )}
          </>
        )}
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
        .pulse {
          animation: pulse 1.5s ease-in-out infinite;
          color: var(--accent);
          margin-right: 8px;
        }
      `}</style>
    </div>
  );
}
