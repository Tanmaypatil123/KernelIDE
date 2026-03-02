import type { Settings, Language, ExecutionResult } from '../types';

export async function verifyEndpoint(endpoint: string): Promise<boolean> {
  try {
    const response = await fetch(`${endpoint}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });

    if (response.ok) {
      const data = await response.json();
      return data.app === 'kernelide-executor';
    }
  } catch {
    // Endpoint unreachable
  }
  return false;
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
