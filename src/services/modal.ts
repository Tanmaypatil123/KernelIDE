import type { Settings, Language, ExecutionResult } from '../types';

const MODAL_APP_NAME = 'kernelide-executor';

export async function checkDeployment(_settings: Settings): Promise<{ deployed: boolean; endpoint?: string }> {
  const endpoint = getEndpointUrl();
  
  if (!endpoint) {
    return { deployed: false };
  }

  try {
    const response = await fetch(`${endpoint}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (response.ok) {
      const data = await response.json();
      if (data.app === 'kernelide-executor') {
        return { deployed: true, endpoint };
      }
    }
  } catch {
    // Endpoint doesn't exist or isn't accessible
  }
  
  return { deployed: false };
}

function getEndpointUrl(): string | null {
  const stored = localStorage.getItem('kernelide_endpoint');
  return stored;
}

export async function deployToModal(_settings: Settings): Promise<{ success: boolean; endpoint?: string; error?: string }> {
  return {
    success: false,
    error: 'Automatic deployment requires Modal CLI. Please deploy manually using: modal deploy modal_executor.py',
  };
}

export async function executeCode(
  code: string,
  language: Language,
  settings: Settings,
  endpoint: string
): Promise<ExecutionResult> {
  try {
    const response = await fetch(`${endpoint}/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        code,
        language,
        gpu_type: settings.gpu,
        timeout: settings.timeout,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      return {
        success: false,
        stdout: '',
        stderr: '',
        error: `Request failed: ${response.status} ${error}`,
      };
    }

    const result = await response.json();
    return {
      success: result.success,
      stdout: result.stdout,
      stderr: result.stderr,
      executionTime: result.execution_time,
    };
  } catch (e) {
    return {
      success: false,
      stdout: '',
      stderr: '',
      error: `Network error: ${e instanceof Error ? e.message : 'Unknown error'}`,
    };
  }
}

export { MODAL_APP_NAME };
