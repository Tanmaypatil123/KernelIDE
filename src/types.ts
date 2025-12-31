export type Language = 'cuda' | 'triton' | 'mojo' | 'cutlass' | 'cutedsl';

export type GPUType = 'T4' | 'L4' | 'A10' | 'A100-40GB' | 'A100-80GB' | 'L40S' | 'H100' | 'H200' | 'B200';

export interface Settings {
  apiKey: string;
  apiSecret: string;
  gpu: GPUType;
  timeout: number;
}

export interface ExecutionResult {
  success: boolean;
  stdout: string;
  stderr: string;
  executionTime?: number;
  gpuMetrics?: {
    memoryUsed?: string;
    gpuUtilization?: string;
  };
  error?: string;
}

export interface DeploymentStatus {
  deployed: boolean;
  endpoint?: string;
  error?: string;
}

export const GPU_OPTIONS: { value: GPUType; label: string; vram: string; price: string }[] = [
  { value: 'T4', label: 'NVIDIA T4', vram: '16 GB', price: '~$0.76/hr' },
  { value: 'L4', label: 'NVIDIA L4', vram: '24 GB', price: '~$0.80/hr' },
  { value: 'A10', label: 'NVIDIA A10', vram: '24 GB', price: '~$1.10/hr' },
  { value: 'A100-40GB', label: 'NVIDIA A100 40GB', vram: '40 GB', price: '~$2.10/hr' },
  { value: 'A100-80GB', label: 'NVIDIA A100 80GB', vram: '80 GB', price: '~$2.50/hr' },
  { value: 'L40S', label: 'NVIDIA L40S', vram: '48 GB', price: '~$1.70/hr' },
  { value: 'H100', label: 'NVIDIA H100', vram: '80 GB', price: '~$3.95/hr' },
  { value: 'H200', label: 'NVIDIA H200', vram: '141 GB', price: '~$4.50/hr' },
  { value: 'B200', label: 'NVIDIA B200', vram: '192 GB', price: '~$5.50/hr' },
];

export const LANGUAGE_OPTIONS: { value: Language; label: string; extension: string }[] = [
  { value: 'cuda', label: 'CUDA C++', extension: '.cu' },
  { value: 'triton', label: 'Triton (Python)', extension: '.py' },
  { value: 'cutlass', label: 'CUTLASS C++', extension: '.cu' },
  { value: 'cutedsl', label: 'CUTE DSL (Python)', extension: '.py' },
  { value: 'mojo', label: 'Mojo', extension: '.mojo' },
];
