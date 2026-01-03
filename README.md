# KernelIDE

A browser-based IDE for practicing GPU kernel development with serverless execution on [Modal.com](https://modal.com). Write CUDA, Triton, CUTLASS, and CUTE DSL code and run it on cloud GPUs - pay only for actual execution time.

## Features

- **Monaco Editor** - VS Code-like editing experience with syntax highlighting and autocompletion
- **Multiple Languages** - CUDA C++, Triton (Python), CUTLASS C++, CUTE DSL (Python), Mojo
- **GPU Selection** - Choose from T4, L4, A10, A100, L40S, H100, H200, B200
- **Configurable Timeout** - Set execution limits (5-300 seconds)
- **Serverless Execution** - No idle charges, pay only for kernel runtime
- **Self-Hostable** - Bring your own Modal API key
- **VSCode Extension** - Use KernelIDE directly within Visual Studio Code

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Deploy the Modal Backend

First, install the Modal CLI and authenticate:

```bash
pip install modal
modal setup  # This will open browser for authentication
```

Then deploy the executor to your Modal account:

```bash
modal deploy modal_executor.py
```

After deployment, you'll see your endpoint URL:
```
https://<your-workspace>--kernelide-executor-api.modal.run
```

### 3. Run the Frontend

```bash
npm run dev
```

Open http://localhost:5173 in your browser.

### 4. Configure the IDE

1. Click **Settings** in the top-right
2. Enter your Modal API credentials (from https://modal.com/settings)
3. Paste your deployed endpoint URL
4. Click **Verify** to confirm the connection

## Usage

### Web IDE

1. Select a language from the dropdown (CUDA, Triton, CUTLASS, CUTE DSL, Mojo)
2. Write your kernel code in the editor
3. Choose your GPU and timeout in Settings
4. Click **Run** or press `Cmd/Ctrl + Enter`
5. View output and execution time in the right panel

### VSCode Extension

The KernelIDE VSCode extension provides the same GPU kernel development experience directly in your editor.

**Installation:**

Download and install the pre-built extension:

```bash
# Download the extension
curl -L -o kernelide.vsix https://github.com/Tanmaypatil123/KernelIDE/releases/download/vscode-extension-v1/kernelide-0.1.0.vsix

# Install the extension
code --install-extension kernelide.vsix
```

After installation, restart VS Code and follow the setup instructions in the [VSCode Extension README](kernelide-vscode/README.md) for complete installation and usage details.

## Supported Languages

| Language | File Type | Description |
|----------|-----------|-------------|
| CUDA C++ | `.cu` | Native CUDA kernels with nvcc |
| Triton | `.py` | OpenAI Triton GPU kernels |
| CUTLASS C++ | `.cu` | NVIDIA CUTLASS templates |
| CUTE DSL | `.py` | nvidia-cutlass-dsl Python package |
| Mojo | `.mojo` | Modular's Mojo language |

## GPU Pricing (Modal.com)

| GPU | VRAM | Approx. Price |
|-----|------|---------------|
| T4 | 16 GB | ~$0.76/hr |
| L4 | 24 GB | ~$0.80/hr |
| A10 | 24 GB | ~$1.10/hr |
| A100-40GB | 40 GB | ~$2.10/hr |
| A100-80GB | 80 GB | ~$2.50/hr |
| L40S | 48 GB | ~$1.70/hr |
| H100 | 80 GB | ~$3.95/hr |
| H200 | 141 GB | ~$4.50/hr |
| B200 | 192 GB | ~$5.50/hr |

*Prices are approximate and per-second billing applies.*

## Tech Stack

**Frontend:**
- React 18 + TypeScript
- Vite
- Monaco Editor (@monaco-editor/react)
- Lucide React (icons)

**Backend (Modal):**
- CUDA 12.9 + Python 3.12
- PyTorch, Triton, cuda-python
- nvidia-cutlass-dsl
- CUTLASS 3.7 (for C++ CUTLASS)
- FastAPI web endpoint

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Self-Hosting

1. Fork/clone this repository
2. Deploy `modal_executor.py` to your Modal account
3. Build the frontend: `npm run build`
4. Serve the `dist/` folder with any static hosting (Vercel, Netlify, GitHub Pages, etc.)
5. Users configure their own Modal API keys in Settings

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
