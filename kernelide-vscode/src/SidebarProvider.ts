import * as vscode from 'vscode';

export class SidebarProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly _context: vscode.ExtensionContext
    ) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'saveSettings': {
                    const config = vscode.workspace.getConfiguration('kernelide');
                    await config.update('modalEndpoint', data.endpoint, vscode.ConfigurationTarget.Global);
                    await config.update('defaultGpu', data.gpu, vscode.ConfigurationTarget.Global);
                    await config.update('defaultLanguage', data.language, vscode.ConfigurationTarget.Global);
                    await config.update('timeout', data.timeout, vscode.ConfigurationTarget.Global);
                    vscode.window.showInformationMessage('KernelIDE settings saved!');
                    break;
                }
                case 'runKernel': {
                    const editor = vscode.window.activeTextEditor;
                    if (!editor) {
                        this._view?.webview.postMessage({ type: 'error', message: 'No active editor found' });
                        return;
                    }
                    const code = editor.document.getText();
                    await this._executeKernel(code, data.gpu, data.language, data.timeout);
                    break;
                }
                case 'getSettings': {
                    this._sendSettings();
                    break;
                }
            }
        });

        this._sendSettings();
    }

    private _sendSettings() {
        const config = vscode.workspace.getConfiguration('kernelide');
        this._view?.webview.postMessage({
            type: 'settings',
            endpoint: config.get('modalEndpoint', ''),
            gpu: config.get('defaultGpu', 'T4'),
            language: config.get('defaultLanguage', 'cuda'),
            timeout: config.get('timeout', 30),
        });
    }

    public async runKernel(code: string) {
        const config = vscode.workspace.getConfiguration('kernelide');
        const gpu = config.get<string>('defaultGpu', 'T4');
        const language = config.get<string>('defaultLanguage', 'cuda');
        const timeout = config.get<number>('timeout', 30);
        await this._executeKernel(code, gpu, language, timeout);
    }

    private async _executeKernel(code: string, gpu: string, language: string, timeout: number) {
        const config = vscode.workspace.getConfiguration('kernelide');
        const endpoint = config.get<string>('modalEndpoint', '');

        if (!endpoint) {
            this._view?.webview.postMessage({
                type: 'output',
                success: false,
                stdout: '',
                stderr: 'Error: Modal endpoint not configured. Please set your Modal.com endpoint URL in settings.',
                executionTime: 0,
            });
            return;
        }

        this._view?.webview.postMessage({ type: 'running' });

        try {
            const response = await fetch(`${endpoint}/execute`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code,
                    language,
                    gpu_type: gpu,
                    timeout,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }

            const result = await response.json() as {
                success: boolean;
                stdout: string;
                stderr: string;
                execution_time: number;
            };

            this._view?.webview.postMessage({
                type: 'output',
                success: result.success,
                stdout: result.stdout,
                stderr: result.stderr,
                executionTime: result.execution_time,
            });
        } catch (error) {
            this._view?.webview.postMessage({
                type: 'output',
                success: false,
                stdout: '',
                stderr: `Execution failed: ${error instanceof Error ? error.message : String(error)}`,
                executionTime: 0,
            });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KernelIDE</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background: var(--vscode-sideBar-background);
            padding: 12px;
        }
        .section {
            margin-bottom: 16px;
        }
        .section-title {
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--vscode-foreground);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .section-title svg {
            width: 16px;
            height: 16px;
        }
        label {
            display: block;
            margin-bottom: 4px;
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
        }
        input, select {
            width: 100%;
            padding: 6px 8px;
            margin-bottom: 8px;
            border: 1px solid var(--vscode-input-border);
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border-radius: 4px;
            font-size: 12px;
        }
        input:focus, select:focus {
            outline: 1px solid var(--vscode-focusBorder);
        }
        button {
            width: 100%;
            padding: 8px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: opacity 0.2s;
        }
        button:hover {
            opacity: 0.9;
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .btn-primary {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
        }
        .btn-secondary {
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
            margin-top: 4px;
        }
        .btn-run {
            background: #22c55e;
            color: white;
            margin-top: 12px;
        }
        .output-container {
            background: var(--vscode-editor-background);
            border: 1px solid var(--vscode-panel-border);
            border-radius: 4px;
            padding: 8px;
            min-height: 150px;
            max-height: 400px;
            overflow-y: auto;
            font-family: var(--vscode-editor-font-family);
            font-size: 11px;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .output-success {
            color: #22c55e;
        }
        .output-error {
            color: #ef4444;
        }
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 0;
            font-size: 10px;
            color: var(--vscode-descriptionForeground);
        }
        .running {
            color: #f59e0b;
        }
        .divider {
            height: 1px;
            background: var(--vscode-panel-border);
            margin: 12px 0;
        }
        .connected {
            color: #22c55e;
        }
        .disconnected {
            color: #ef4444;
        }
        .loader {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid #f59e0b;
            border-radius: 50%;
            border-top-color: transparent;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="section">
        <div class="section-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="3"/>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
            </svg>
            Modal Configuration
        </div>
        <label>Endpoint URL</label>
        <input type="text" id="endpoint" placeholder="https://your-workspace--kernelide-executor-api.modal.run">
        <div class="status-bar">
            <span id="connection-status" class="disconnected">Not connected</span>
        </div>
    </div>

    <div class="divider"></div>

    <div class="section">
        <div class="section-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="4" y="4" width="16" height="16" rx="2" ry="2"/>
                <rect x="9" y="9" width="6" height="6"/>
            </svg>
            Kernel Settings
        </div>
        <label>GPU Type</label>
        <select id="gpu">
            <option value="T4">T4 (Budget)</option>
            <option value="L4">L4</option>
            <option value="A10">A10</option>
            <option value="A100-40GB">A100 40GB</option>
            <option value="A100-80GB">A100 80GB</option>
            <option value="L40S">L40S</option>
            <option value="H100">H100</option>
            <option value="H200">H200</option>
            <option value="B200">B200 (Premium)</option>
        </select>

        <label>Language / DSL</label>
        <select id="language">
            <option value="cuda">CUDA C++</option>
            <option value="triton">Triton (Python)</option>
            <option value="cutlass">CUTLASS</option>
            <option value="cutedsl">CUTE DSL</option>
            <option value="mojo">Mojo</option>
        </select>

        <label>Timeout (seconds)</label>
        <input type="number" id="timeout" value="30" min="5" max="300">
    </div>

    <button class="btn-primary" id="saveBtn">Save Settings</button>
    <button class="btn-run" id="runBtn">
        <span id="runBtnText">▶ Run Kernel (⌘⇧K)</span>
    </button>

    <div class="divider"></div>

    <div class="section">
        <div class="section-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="4 17 10 11 4 5"/>
                <line x1="12" y1="19" x2="20" y2="19"/>
            </svg>
            Output
        </div>
        <div class="status-bar">
            <span id="exec-status"></span>
            <span id="exec-time"></span>
        </div>
        <div class="output-container" id="output">Run a kernel to see output...</div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        const endpointInput = document.getElementById('endpoint');
        const gpuSelect = document.getElementById('gpu');
        const languageSelect = document.getElementById('language');
        const timeoutInput = document.getElementById('timeout');
        const saveBtn = document.getElementById('saveBtn');
        const runBtn = document.getElementById('runBtn');
        const runBtnText = document.getElementById('runBtnText');
        const output = document.getElementById('output');
        const connectionStatus = document.getElementById('connection-status');
        const execStatus = document.getElementById('exec-status');
        const execTime = document.getElementById('exec-time');

        vscode.postMessage({ type: 'getSettings' });

        saveBtn.addEventListener('click', () => {
            vscode.postMessage({
                type: 'saveSettings',
                endpoint: endpointInput.value,
                gpu: gpuSelect.value,
                language: languageSelect.value,
                timeout: parseInt(timeoutInput.value) || 30,
            });
            updateConnectionStatus();
        });

        runBtn.addEventListener('click', () => {
            vscode.postMessage({
                type: 'runKernel',
                gpu: gpuSelect.value,
                language: languageSelect.value,
                timeout: parseInt(timeoutInput.value) || 30,
            });
        });

        function updateConnectionStatus() {
            if (endpointInput.value) {
                connectionStatus.textContent = 'Configured';
                connectionStatus.className = 'connected';
            } else {
                connectionStatus.textContent = 'Not configured';
                connectionStatus.className = 'disconnected';
            }
        }

        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.type) {
                case 'settings':
                    endpointInput.value = message.endpoint || '';
                    gpuSelect.value = message.gpu || 'T4';
                    languageSelect.value = message.language || 'cuda';
                    timeoutInput.value = message.timeout || 30;
                    updateConnectionStatus();
                    break;
                case 'running':
                    runBtn.disabled = true;
                    runBtnText.innerHTML = '<span class="loader"></span> Running...';
                    output.textContent = 'Executing kernel on GPU...';
                    output.className = 'output-container';
                    execStatus.textContent = '';
                    execStatus.className = 'running';
                    execTime.textContent = '';
                    break;
                case 'output':
                    runBtn.disabled = false;
                    runBtnText.textContent = '▶ Run Kernel (⌘⇧K)';
                    if (message.success) {
                        output.textContent = message.stdout || '(No output)';
                        output.className = 'output-container output-success';
                        execStatus.textContent = '✓ Success';
                        execStatus.className = 'output-success';
                    } else {
                        output.textContent = message.stderr || message.stdout || 'Unknown error';
                        output.className = 'output-container output-error';
                        execStatus.textContent = '✗ Failed';
                        execStatus.className = 'output-error';
                    }
                    if (message.executionTime) {
                        execTime.textContent = message.executionTime.toFixed(2) + 's';
                    }
                    break;
                case 'error':
                    runBtn.disabled = false;
                    runBtnText.textContent = '▶ Run Kernel (⌘⇧K)';
                    output.textContent = message.message;
                    output.className = 'output-container output-error';
                    break;
            }
        });
    </script>
</body>
</html>`;
    }
}
