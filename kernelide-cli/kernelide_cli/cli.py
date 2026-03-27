"""
KernelIDE CLI - Submit GPU kernels from your terminal.

Usage:
    kernelide setup          Configure your Modal endpoint
    kernelide submit <file>  Submit a kernel for execution
    kernelide config         Show current configuration
"""

import json
import os
import sys
import time
from pathlib import Path

import click
import requests

CONFIG_DIR = Path.home() / ".kernelide"
CONFIG_FILE = CONFIG_DIR / "config.json"

SUPPORTED_GPUS = [
    "T4", "L4", "A10", "A100-40GB", "A100-80GB", "L40S", "H100", "H200", "B200",
]

SUPPORTED_LANGUAGES = ["cuda", "triton", "cutlass", "cutedsl", "mojo", "cutile"]

LANGUAGE_INFO = {
    "cuda":    {"name": "CUDA C++",    "ext": ".cu",   "desc": "NVIDIA CUDA C++ kernels compiled with nvcc"},
    "triton":  {"name": "Triton",      "ext": ".py",   "desc": "OpenAI Triton Python DSL for GPU programming"},
    "cutlass": {"name": "CUTLASS C++", "ext": ".cu",   "desc": "NVIDIA CUTLASS templates for GEMM and convolutions"},
    "cutedsl": {"name": "CUTE DSL",    "ext": ".py",   "desc": "CuTe layout DSL for tensor core programming"},
    "mojo":    {"name": "Mojo",        "ext": ".mojo", "desc": "Modular Mojo language for GPU/AI workloads"},
    "cutile":  {"name": "cuTile",      "ext": ".py",   "desc": "NVIDIA cuTile tile-based GPU programming"},
}

EXTENSION_MAP = {
    ".cu": "cuda",
    ".py": "triton",
    ".mojo": "mojo",
    ".🔥": "mojo",
}


def load_config() -> dict:
    """Load config from ~/.kernelide/config.json."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_config(config: dict):
    """Save config to ~/.kernelide/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def verify_endpoint(endpoint: str) -> bool:
    """Check if the endpoint is reachable and is a KernelIDE executor."""
    try:
        resp = requests.get(f"{endpoint}/health", timeout=5)
        if resp.ok:
            data = resp.json()
            return data.get("app") == "kernelide-executor"
    except requests.RequestException:
        pass
    return False


def detect_language(filepath: str):
    """Auto-detect language from file extension."""
    ext = Path(filepath).suffix.lower()
    return EXTENSION_MAP.get(ext)


@click.group()
@click.version_option(version="0.1.0", prog_name="kernelide")
def cli():
    """KernelIDE CLI - Submit GPU kernels from your terminal."""
    pass


@cli.command()
def setup():
    """Configure your Modal endpoint and default settings."""
    config = load_config()

    click.echo()
    click.secho("KernelIDE Setup", fg="cyan", bold=True)
    click.secho("=" * 40, fg="cyan")
    click.echo()

    # Endpoint URL
    default_endpoint = config.get("endpoint", "")
    endpoint = click.prompt(
        "Modal endpoint URL",
        default=default_endpoint or None,
        show_default=bool(default_endpoint),
    )
    endpoint = endpoint.rstrip("/")

    click.echo("Verifying endpoint... ", nl=False)
    if verify_endpoint(endpoint):
        click.secho("Connected!", fg="green", bold=True)
    else:
        click.secho("Failed", fg="red", bold=True)
        click.echo("Could not verify the endpoint. Save anyway?")
        if not click.confirm("Continue?", default=False):
            click.echo("Setup cancelled.")
            return

    # Default GPU
    click.echo()
    click.echo("Available GPUs:")
    for i, gpu in enumerate(SUPPORTED_GPUS, 1):
        click.echo(f"  {i}. {gpu}")

    default_gpu = config.get("gpu", "T4")
    gpu = click.prompt(
        "Default GPU",
        default=default_gpu,
        type=click.Choice(SUPPORTED_GPUS, case_sensitive=False),
        show_choices=False,
    )

    # Default timeout
    default_timeout = config.get("timeout", 30)
    timeout = click.prompt(
        "Default timeout (seconds, 5-300)",
        default=default_timeout,
        type=click.IntRange(5, 300),
    )

    config.update({
        "endpoint": endpoint,
        "gpu": gpu,
        "timeout": timeout,
    })
    save_config(config)

    click.echo()
    click.secho("Configuration saved!", fg="green", bold=True)
    click.echo(f"  Config file: {CONFIG_FILE}")


@cli.command()
def config():
    """Show current configuration."""
    cfg = load_config()

    if not cfg:
        click.secho("No configuration found. Run 'kernelide setup' first.", fg="yellow")
        return

    click.echo()
    click.secho("KernelIDE Configuration", fg="cyan", bold=True)
    click.secho("=" * 40, fg="cyan")
    click.echo(f"  Endpoint : {cfg.get('endpoint', 'not set')}")
    click.echo(f"  GPU      : {cfg.get('gpu', 'T4')}")
    click.echo(f"  Timeout  : {cfg.get('timeout', 30)}s")
    click.echo(f"  Config   : {CONFIG_FILE}")
    click.echo()


@cli.command()
def languages():
    """List all supported kernel languages/DSLs."""
    click.echo()
    click.secho("Supported Languages", fg="cyan", bold=True)
    click.secho("=" * 60, fg="cyan")
    click.echo()
    for key, info in LANGUAGE_INFO.items():
        click.secho(f"  {key:<10}", fg="green", bold=True, nl=False)
        click.echo(f" {info['name']:<14} ({info['ext']})")
        click.echo(f"             {info['desc']}")
        click.echo()
    click.echo(f"  Total: {len(LANGUAGE_INFO)} languages supported")
    click.echo()
    click.echo("  Use --language/-l flag with submit to pick a language:")
    click.secho("    kernelide submit kernel.py --language cutedsl", fg="yellow")
    click.echo()


@cli.command()
def gpus():
    """List all supported GPU types."""
    click.echo()
    click.secho("Supported GPUs", fg="cyan", bold=True)
    click.secho("=" * 40, fg="cyan")
    click.echo()
    for gpu in SUPPORTED_GPUS:
        click.echo(f"  - {gpu}")
    click.echo()
    click.echo(f"  Total: {len(SUPPORTED_GPUS)} GPU types supported")
    click.echo()
    click.echo("  Use --gpu/-g flag with submit to pick a GPU:")
    click.secho("    kernelide submit kernel.cu --gpu H100", fg="yellow")
    click.echo()


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--gpu", "-g",
    type=click.Choice(SUPPORTED_GPUS, case_sensitive=False),
    default=None,
    help="GPU type for execution.",
)
@click.option(
    "--timeout", "-t",
    type=click.IntRange(5, 300),
    default=None,
    help="Execution timeout in seconds (5-300).",
)
@click.option(
    "--language", "-l",
    type=click.Choice(SUPPORTED_LANGUAGES, case_sensitive=False),
    default=None,
    help="Override auto-detected language.",
)
@click.option(
    "--endpoint", "-e",
    default=None,
    help="Override saved endpoint URL.",
)
def submit(file, gpu, timeout, language, endpoint):
    """Submit a kernel file for execution on a remote GPU."""
    config = load_config()

    # Resolve endpoint
    endpoint = endpoint or config.get("endpoint")
    if not endpoint:
        click.secho(
            "Error: No endpoint configured. Run 'kernelide setup' or pass --endpoint.",
            fg="red",
        )
        sys.exit(1)
    endpoint = endpoint.rstrip("/")

    # Resolve GPU and timeout from flags or config
    gpu = gpu or config.get("gpu", "T4")
    timeout = timeout or config.get("timeout", 30)

    # Detect or validate language
    if language:
        lang = language.lower()
    else:
        lang = detect_language(file)
        if not lang:
            click.secho(
                f"Error: Cannot detect language from '{Path(file).suffix}'. "
                f"Use --language to specify one of: {', '.join(SUPPORTED_LANGUAGES)}",
                fg="red",
            )
            sys.exit(1)

    # Read source file
    filepath = Path(file)
    code = filepath.read_text(encoding="utf-8")

    # Display submission info
    click.echo()
    click.secho("Submitting kernel", fg="cyan", bold=True)
    click.echo(f"  File     : {filepath.name}")
    click.echo(f"  Language : {lang}")
    click.echo(f"  GPU      : {gpu}")
    click.echo(f"  Timeout  : {timeout}s")
    click.echo()

    # Send request
    click.echo("Executing... ", nl=False)
    start = time.time()

    try:
        resp = requests.post(
            f"{endpoint}/execute",
            json={
                "code": code,
                "language": lang,
                "gpu_type": gpu,
                "timeout": timeout,
            },
            timeout=timeout + 30,  # HTTP timeout > execution timeout
        )
    except requests.ConnectionError:
        click.secho("Failed", fg="red", bold=True)
        click.secho(f"Error: Could not connect to {endpoint}", fg="red")
        sys.exit(1)
    except requests.Timeout:
        click.secho("Timed out", fg="red", bold=True)
        click.secho("Error: Request timed out waiting for response.", fg="red")
        sys.exit(1)

    elapsed = time.time() - start

    if not resp.ok:
        click.secho("Failed", fg="red", bold=True)
        click.secho(f"Error: HTTP {resp.status_code} - {resp.text}", fg="red")
        sys.exit(1)

    result = resp.json()
    success = result.get("success", False)
    exec_time = result.get("execution_time", 0)

    if success:
        click.secho("Done!", fg="green", bold=True)
    else:
        click.secho("Failed", fg="red", bold=True)

    click.echo()

    # Output
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")

    if stdout:
        click.secho("--- stdout ---", fg="cyan")
        click.echo(stdout)

    if stderr:
        click.secho("--- stderr ---", fg="yellow" if success else "red")
        click.echo(stderr)

    # Summary
    click.echo()
    status_icon = click.style("SUCCESS", fg="green", bold=True) if success else click.style("FAILED", fg="red", bold=True)
    click.echo(f"  Status   : {status_icon}")
    click.echo(f"  GPU Time : {exec_time:.2f}s")
    click.echo(f"  Total    : {elapsed:.2f}s (includes network)")
    click.echo()

    if not success:
        sys.exit(1)
