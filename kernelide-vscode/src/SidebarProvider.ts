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

        webviewView.webview.onDidReceiveMessage(async (data: any) => {
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
                stderr: 'Error: Modal endpoint not configured. Paste your endpoint URL above and click Connect.',
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
        .btn-connect {
            background: #3b82f6;
            color: white;
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
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
        }
        .status-connected {
            color: #22c55e;
            background: rgba(34, 197, 94, 0.1);
        }
        .status-disconnected {
            color: #ef4444;
            background: rgba(239, 68, 68, 0.1);
        }
        .status-none {
            color: #f59e0b;
            background: rgba(245, 158, 11, 0.1);
        }
        .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            display: inline-block;
        }
        .dot-green { background: #22c55e; }
        .dot-red { background: #ef4444; }
        .dot-yellow { background: #f59e0b; }
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
                <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
                <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
            </svg>
            Connect to Modal
        </div>
        <label>Endpoint URL</label>
        <input type="text" id="endpoint" placeholder="https://your-workspace--kernelide-executor-api.modal.run">
        <button class="btn-connect" id="connectBtn">Connect</button>
        <div class="status-bar">
            <span id="connection-status" class="status-badge status-none"><span class="dot dot-yellow"></span> Not connected</span>
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
            <option value="cutile">cuTile (Python)</option>
            <option value="mojo">Mojo</option>
        </select>

        <label>Timeout (seconds)</label>
        <input type="number" id="timeout" value="30" min="5" max="300">
    </div>

    <button class="btn-primary" id="saveBtn">Save Settings</button>
    <button class="btn-run" id="runBtn">
        <span id="runBtnText">&#9654; Run Kernel</span>
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
        const connectBtn = document.getElementById('connectBtn');
        const runBtn = document.getElementById('runBtn');
        const runBtnText = document.getElementById('runBtnText');
        const output = document.getElementById('output');
        const connectionStatus = document.getElementById('connection-status');
        const execStatus = document.getElementById('exec-status');
        const execTime = document.getElementById('exec-time');

        vscode.postMessage({ type: 'getSettings' });

        // Connect button - saves endpoint and verifies
        connectBtn.addEventListener('click', () => {
            const url = endpointInput.value.trim().replace(/\\/+$/, '');
            if (!url) {
                connectionStatus.className = 'status-badge status-none';
                connectionStatus.innerHTML = '<span class="dot dot-yellow"></span> Enter a URL';
                return;
            }

            connectionStatus.className = 'status-badge status-none';
            connectionStatus.innerHTML = '<span class="loader"></span> Verifying...';

            // Save immediately
            vscode.postMessage({
                type: 'saveSettings',
                endpoint: url,
                gpu: gpuSelect.value,
                language: languageSelect.value,
                timeout: parseInt(timeoutInput.value) || 30,
            });

            // Verify via health check
            fetch(url + '/health')
                .then(r => r.json())
                .then(data => {
                    if (data.app === 'kernelide-executor') {
                        connectionStatus.className = 'status-badge status-connected';
                        connectionStatus.innerHTML = '<span class="dot dot-green"></span> Connected';
                    } else {
                        connectionStatus.className = 'status-badge status-disconnected';
                        connectionStatus.innerHTML = '<span class="dot dot-red"></span> Invalid endpoint';
                    }
                })
                .catch(() => {
                    connectionStatus.className = 'status-badge status-disconnected';
                    connectionStatus.innerHTML = '<span class="dot dot-red"></span> Could not connect';
                });
        });

        // Enter key to connect
        endpointInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') connectBtn.click();
        });

        saveBtn.addEventListener('click', () => {
            vscode.postMessage({
                type: 'saveSettings',
                endpoint: endpointInput.value.trim().replace(/\\/+$/, ''),
                gpu: gpuSelect.value,
                language: languageSelect.value,
                timeout: parseInt(timeoutInput.value) || 30,
            });
        });

        runBtn.addEventListener('click', () => {
            vscode.postMessage({
                type: 'runKernel',
                gpu: gpuSelect.value,
                language: languageSelect.value,
                timeout: parseInt(timeoutInput.value) || 30,
            });
        });

        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.type) {
                case 'settings':
                    endpointInput.value = message.endpoint || '';
                    gpuSelect.value = message.gpu || 'T4';
                    languageSelect.value = message.language || 'cuda';
                    timeoutInput.value = message.timeout || 30;
                    // Auto-verify if endpoint exists
                    if (message.endpoint) {
                        connectionStatus.className = 'status-badge status-none';
                        connectionStatus.innerHTML = '<span class="loader"></span> Verifying...';
                        fetch(message.endpoint + '/health')
                            .then(r => r.json())
                            .then(data => {
                                if (data.app === 'kernelide-executor') {
                                    connectionStatus.className = 'status-badge status-connected';
                                    connectionStatus.innerHTML = '<span class="dot dot-green"></span> Connected';
                                } else {
                                    connectionStatus.className = 'status-badge status-disconnected';
                                    connectionStatus.innerHTML = '<span class="dot dot-red"></span> Invalid endpoint';
                                }
                            })
                            .catch(() => {
                                connectionStatus.className = 'status-badge status-disconnected';
                                connectionStatus.innerHTML = '<span class="dot dot-red"></span> Could not connect';
                            });
                    }
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
                    runBtnText.innerHTML = '&#9654; Run Kernel';
                    if (message.success) {
                        output.textContent = message.stdout || '(No output)';
                        output.className = 'output-container output-success';
                        execStatus.textContent = 'Success';
                        execStatus.className = 'output-success';
                    } else {
                        output.textContent = message.stderr || message.stdout || 'Unknown error';
                        output.className = 'output-container output-error';
                        execStatus.textContent = 'Failed';
                        execStatus.className = 'output-error';
                    }
                    if (message.executionTime) {
                        execTime.textContent = message.executionTime.toFixed(2) + 's';
                    }
                    break;
                case 'error':
                    runBtn.disabled = false;
                    runBtnText.innerHTML = '&#9654; Run Kernel';
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
