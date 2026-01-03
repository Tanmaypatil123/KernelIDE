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
after you do `modal setup` go to this link and [create](https://modal.com/settings) a new token by going to  Create proxy auth tokens :

<img width="743" height="367" alt="image" src="https://github.com/user-attachments/assets/b20e1b12-d8e6-42b4-bd53-125a3d8d195d" />


Then deploy the executor to your Modal account:

```bash
modal deploy modal_executor.py
```

After you run above command you'll see someting like this for the first time.

<img width="738" height="376" alt="image" src="https://github.com/user-attachments/assets/0b771fcf-885f-4d28-be1c-e094b45b7776" />

After deployment, you'll see your endpoint URL:
```
https://<your-workspace>--kernelide-executor-api.modal.run
```

### 3. Then , Run the Frontend

```bash
npm run dev
```

Open http://localhost:5173 in your browser, and you will see the KernelIDE like this.

<img width="2930" height="1496" alt="image" src="https://github.com/user-attachments/assets/86d52a9e-ffd8-4386-a126-2a3c964b1431" />

### 4. Configure the IDE

1. Click **Settings** in the top-right
2. Enter your [Modal API credentials](https://modal.com/settings)
3. Click **Deploy Executoir** and paste your deployed endpoint URL it should look like this `https://<your-workspace>--kernelide-executor-api.modal.run` to confirm the connection. after you do this, the screen should look like this in below image.
   <img width="2924" height="1486" alt="image" src="https://github.com/user-attachments/assets/7b3fe28d-ab11-47a4-a839-9db9fb680085" />
4.Click on save settings and close the settings popup.


## Usage

### Web IDE

1. Select a language from the dropdown (CUDA, Triton, CUTLASS, CUTE DSL, Mojo)
2. Write your kernel code in the editor
3. Choose your GPU and timeout in Settings
4. Click **Run** or press `Cmd/Ctrl + Enter`
5. View output and execution time in the right panel
<img width="2922" height="1502" alt="image" src="https://github.com/user-attachments/assets/320ba2d7-4901-4da9-b3e0-51e26ec8525d" />


### VSCode Extension

The KernelIDE VSCode extension provides the same GPU kernel development experience directly in your editor.

**Installation:**

Download and install the pre-built extension:

```bash
# Download the extension
curl -L -o kernelide.vsix https://github.com/Tanmaypatil123/KernelIDE/releases/download/vscode-extension-v1/kernelide-0.1.0.vsix

# Install the extension
code --install-extension kernelide-0.1.0.vsix
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

## Contributing/Development

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
