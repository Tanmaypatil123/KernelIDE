# KernelIDE VS Code Extension

GPU kernel development extension for VS Code with Modal.com integration.

## Features

- **Activity Bar Icon**: Quick access to KernelIDE from the sidebar
- **Modal.com Integration**: Configure your Modal endpoint for GPU execution
- **Multi-GPU Support**: T4, L4, A10, A100, L40S, H100, H200, B200
- **Multiple Languages**: CUDA, Triton, CUTLASS, CUTE DSL, Mojo
- **Live Output**: See kernel execution results in the sidebar

## Installation

### Quick Install (Recommended)

Download the latest release and install directly:

```bash
# Download the extension
curl -L -o kernelide.vsix https://github.com/Tanmaypatil123/KernelIDE/releases/download/vscode-extension-v1/kernelide-0.1.0.vsix

# Install the extension
code --install-extension kernelide.vsix
```

Or download manually from the [releases page](https://github.com/Tanmaypatil123/KernelIDE/releases/tag/vscode-extension-v1) and install:

```bash
code --install-extension kernelide-0.1.0.vsix
```

After installation, restart VS Code to activate the extension.

## Setup

1. Deploy `modal_executor.py` to Modal: `modal deploy modal_executor.py`
2. Copy your endpoint URL (e.g., `https://your-workspace--kernelide-executor-api.modal.run`)
3. Open KernelIDE sidebar and paste the endpoint
4. Select GPU and language, then run!

## Usage

- Click the KernelIDE icon in the activity bar
- Configure your Modal endpoint
- Open a kernel file
- Press `Cmd+Shift+K` (Mac) or `Ctrl+Shift+K` (Windows/Linux) to run

## Development

```bash
cd kernelide-vscode
npm install
npm run compile
```

Press F5 in VS Code to launch Extension Development Host.
