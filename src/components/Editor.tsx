import MonacoEditor from '@monaco-editor/react';
import type { OnMount } from '@monaco-editor/react';
import { useRef, useEffect } from 'react';
import type { editor } from 'monaco-editor';
import type { Language } from '../types';

interface EditorProps {
  language: Language;
  code: string;
  onChange: (code: string) => void;
}

const languageMap: Record<Language, string> = {
  cuda: 'cpp',
  triton: 'python',
  mojo: 'python',
  cutlass: 'cpp',
  cutedsl: 'python',
};

export function Editor({ language, code, onChange }: EditorProps) {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;

    // Configure dark theme
    monaco.editor.defineTheme('kernelide-dark', {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'comment', foreground: '6a737d', fontStyle: 'italic' },
        { token: 'keyword', foreground: 'ff79c6' },
        { token: 'string', foreground: 'f1fa8c' },
        { token: 'number', foreground: 'bd93f9' },
        { token: 'type', foreground: '8be9fd' },
        { token: 'function', foreground: '50fa7b' },
        { token: 'variable', foreground: 'f8f8f2' },
        { token: 'operator', foreground: 'ff79c6' },
      ],
      colors: {
        'editor.background': '#0d0d0d',
        'editor.foreground': '#e4e4e4',
        'editor.lineHighlightBackground': '#1a1a1a',
        'editor.selectionBackground': '#3b82f640',
        'editor.inactiveSelectionBackground': '#3b82f620',
        'editorCursor.foreground': '#3b82f6',
        'editorLineNumber.foreground': '#4a4a4a',
        'editorLineNumber.activeForeground': '#888888',
        'editorIndentGuide.background': '#2a2a2a',
        'editorIndentGuide.activeBackground': '#3a3a3a',
        'editorBracketMatch.background': '#3b82f640',
        'editorBracketMatch.border': '#3b82f6',
      },
    });
    monaco.editor.setTheme('kernelide-dark');

    // Add CUDA/GPU specific completions
    const cudaCompletions = [
      // CUDA keywords
      { label: '__global__', kind: monaco.languages.CompletionItemKind.Keyword, insertText: '__global__', detail: 'CUDA kernel function qualifier' },
      { label: '__device__', kind: monaco.languages.CompletionItemKind.Keyword, insertText: '__device__', detail: 'CUDA device function qualifier' },
      { label: '__shared__', kind: monaco.languages.CompletionItemKind.Keyword, insertText: '__shared__', detail: 'CUDA shared memory qualifier' },
      { label: '__constant__', kind: monaco.languages.CompletionItemKind.Keyword, insertText: '__constant__', detail: 'CUDA constant memory qualifier' },
      { label: '__host__', kind: monaco.languages.CompletionItemKind.Keyword, insertText: '__host__', detail: 'CUDA host function qualifier' },
      { label: '__restrict__', kind: monaco.languages.CompletionItemKind.Keyword, insertText: '__restrict__', detail: 'Pointer aliasing hint' },
      { label: '__launch_bounds__', kind: monaco.languages.CompletionItemKind.Keyword, insertText: '__launch_bounds__(${1:maxThreadsPerBlock}, ${2:minBlocksPerMultiprocessor})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Kernel launch bounds' },
      
      // Thread indexing
      { label: 'threadIdx', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'threadIdx', detail: 'Thread index within block' },
      { label: 'threadIdx.x', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'threadIdx.x', detail: 'Thread X index' },
      { label: 'threadIdx.y', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'threadIdx.y', detail: 'Thread Y index' },
      { label: 'threadIdx.z', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'threadIdx.z', detail: 'Thread Z index' },
      { label: 'blockIdx', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'blockIdx', detail: 'Block index within grid' },
      { label: 'blockIdx.x', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'blockIdx.x', detail: 'Block X index' },
      { label: 'blockIdx.y', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'blockIdx.y', detail: 'Block Y index' },
      { label: 'blockIdx.z', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'blockIdx.z', detail: 'Block Z index' },
      { label: 'blockDim', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'blockDim', detail: 'Block dimensions' },
      { label: 'blockDim.x', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'blockDim.x', detail: 'Block X dimension' },
      { label: 'blockDim.y', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'blockDim.y', detail: 'Block Y dimension' },
      { label: 'blockDim.z', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'blockDim.z', detail: 'Block Z dimension' },
      { label: 'gridDim', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'gridDim', detail: 'Grid dimensions' },
      { label: 'gridDim.x', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'gridDim.x', detail: 'Grid X dimension' },
      { label: 'gridDim.y', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'gridDim.y', detail: 'Grid Y dimension' },
      { label: 'gridDim.z', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'gridDim.z', detail: 'Grid Z dimension' },
      { label: 'warpSize', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'warpSize', detail: 'Warp size (32)' },
      
      // Memory functions
      { label: 'cudaMalloc', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaMalloc(&${1:ptr}, ${2:size})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Allocate device memory' },
      { label: 'cudaMallocManaged', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaMallocManaged(&${1:ptr}, ${2:size})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Allocate unified memory' },
      { label: 'cudaMallocHost', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaMallocHost(&${1:ptr}, ${2:size})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Allocate pinned host memory' },
      { label: 'cudaFree', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaFree(${1:ptr})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Free device memory' },
      { label: 'cudaFreeHost', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaFreeHost(${1:ptr})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Free pinned host memory' },
      { label: 'cudaMemcpy', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaMemcpy(${1:dst}, ${2:src}, ${3:size}, ${4:cudaMemcpyHostToDevice})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Copy memory between host and device' },
      { label: 'cudaMemcpyAsync', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaMemcpyAsync(${1:dst}, ${2:src}, ${3:size}, ${4:cudaMemcpyHostToDevice}, ${5:stream})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Async copy memory' },
      { label: 'cudaMemset', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaMemset(${1:ptr}, ${2:value}, ${3:size})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Set device memory' },
      { label: 'cudaMemsetAsync', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaMemsetAsync(${1:ptr}, ${2:value}, ${3:size}, ${4:stream})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Async set device memory' },
      { label: 'cudaMemcpyHostToDevice', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'cudaMemcpyHostToDevice', detail: 'Host to device copy direction' },
      { label: 'cudaMemcpyDeviceToHost', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'cudaMemcpyDeviceToHost', detail: 'Device to host copy direction' },
      { label: 'cudaMemcpyDeviceToDevice', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'cudaMemcpyDeviceToDevice', detail: 'Device to device copy direction' },
      
      // Synchronization
      { label: '__syncthreads', kind: monaco.languages.CompletionItemKind.Function, insertText: '__syncthreads()', detail: 'Synchronize threads in block' },
      { label: '__syncwarp', kind: monaco.languages.CompletionItemKind.Function, insertText: '__syncwarp(${1:mask})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Synchronize threads in warp' },
      { label: '__threadfence', kind: monaco.languages.CompletionItemKind.Function, insertText: '__threadfence()', detail: 'Memory fence (device scope)' },
      { label: '__threadfence_block', kind: monaco.languages.CompletionItemKind.Function, insertText: '__threadfence_block()', detail: 'Memory fence (block scope)' },
      { label: '__threadfence_system', kind: monaco.languages.CompletionItemKind.Function, insertText: '__threadfence_system()', detail: 'Memory fence (system scope)' },
      { label: 'cudaDeviceSynchronize', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaDeviceSynchronize()', detail: 'Synchronize device' },
      
      // Streams
      { label: 'cudaStream_t', kind: monaco.languages.CompletionItemKind.TypeParameter, insertText: 'cudaStream_t ${1:stream}', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'CUDA stream type' },
      { label: 'cudaStreamCreate', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaStreamCreate(&${1:stream})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Create stream' },
      { label: 'cudaStreamDestroy', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaStreamDestroy(${1:stream})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Destroy stream' },
      { label: 'cudaStreamSynchronize', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaStreamSynchronize(${1:stream})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Synchronize stream' },
      
      // Atomic operations
      { label: 'atomicAdd', kind: monaco.languages.CompletionItemKind.Function, insertText: 'atomicAdd(&${1:address}, ${2:val})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Atomic addition' },
      { label: 'atomicSub', kind: monaco.languages.CompletionItemKind.Function, insertText: 'atomicSub(&${1:address}, ${2:val})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Atomic subtraction' },
      { label: 'atomicMax', kind: monaco.languages.CompletionItemKind.Function, insertText: 'atomicMax(&${1:address}, ${2:val})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Atomic maximum' },
      { label: 'atomicMin', kind: monaco.languages.CompletionItemKind.Function, insertText: 'atomicMin(&${1:address}, ${2:val})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Atomic minimum' },
      { label: 'atomicExch', kind: monaco.languages.CompletionItemKind.Function, insertText: 'atomicExch(&${1:address}, ${2:val})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Atomic exchange' },
      { label: 'atomicCAS', kind: monaco.languages.CompletionItemKind.Function, insertText: 'atomicCAS(&${1:address}, ${2:compare}, ${3:val})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Atomic compare and swap' },
      { label: 'atomicAnd', kind: monaco.languages.CompletionItemKind.Function, insertText: 'atomicAnd(&${1:address}, ${2:val})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Atomic AND' },
      { label: 'atomicOr', kind: monaco.languages.CompletionItemKind.Function, insertText: 'atomicOr(&${1:address}, ${2:val})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Atomic OR' },
      { label: 'atomicXor', kind: monaco.languages.CompletionItemKind.Function, insertText: 'atomicXor(&${1:address}, ${2:val})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Atomic XOR' },
      
      // Warp primitives
      { label: '__shfl_sync', kind: monaco.languages.CompletionItemKind.Function, insertText: '__shfl_sync(${1:mask}, ${2:var}, ${3:srcLane})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Warp shuffle' },
      { label: '__shfl_up_sync', kind: monaco.languages.CompletionItemKind.Function, insertText: '__shfl_up_sync(${1:mask}, ${2:var}, ${3:delta})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Warp shuffle up' },
      { label: '__shfl_down_sync', kind: monaco.languages.CompletionItemKind.Function, insertText: '__shfl_down_sync(${1:mask}, ${2:var}, ${3:delta})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Warp shuffle down' },
      { label: '__shfl_xor_sync', kind: monaco.languages.CompletionItemKind.Function, insertText: '__shfl_xor_sync(${1:mask}, ${2:var}, ${3:laneMask})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Warp shuffle XOR' },
      { label: '__ballot_sync', kind: monaco.languages.CompletionItemKind.Function, insertText: '__ballot_sync(${1:mask}, ${2:predicate})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Warp ballot' },
      { label: '__all_sync', kind: monaco.languages.CompletionItemKind.Function, insertText: '__all_sync(${1:mask}, ${2:predicate})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Warp all' },
      { label: '__any_sync', kind: monaco.languages.CompletionItemKind.Function, insertText: '__any_sync(${1:mask}, ${2:predicate})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Warp any' },
      { label: '__activemask', kind: monaco.languages.CompletionItemKind.Function, insertText: '__activemask()', detail: 'Get active thread mask' },
      { label: '__popc', kind: monaco.languages.CompletionItemKind.Function, insertText: '__popc(${1:x})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Population count' },
      { label: '__clz', kind: monaco.languages.CompletionItemKind.Function, insertText: '__clz(${1:x})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Count leading zeros' },
      { label: '__ffs', kind: monaco.languages.CompletionItemKind.Function, insertText: '__ffs(${1:x})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Find first set bit' },
      
      // Math functions
      { label: '__expf', kind: monaco.languages.CompletionItemKind.Function, insertText: '__expf(${1:x})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Fast exponential' },
      { label: '__logf', kind: monaco.languages.CompletionItemKind.Function, insertText: '__logf(${1:x})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Fast log' },
      { label: '__sinf', kind: monaco.languages.CompletionItemKind.Function, insertText: '__sinf(${1:x})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Fast sine' },
      { label: '__cosf', kind: monaco.languages.CompletionItemKind.Function, insertText: '__cosf(${1:x})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Fast cosine' },
      { label: '__powf', kind: monaco.languages.CompletionItemKind.Function, insertText: '__powf(${1:x}, ${2:y})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Fast power' },
      { label: 'rsqrtf', kind: monaco.languages.CompletionItemKind.Function, insertText: 'rsqrtf(${1:x})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Reciprocal square root' },
      { label: 'fmaf', kind: monaco.languages.CompletionItemKind.Function, insertText: 'fmaf(${1:x}, ${2:y}, ${3:z})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Fused multiply-add' },
      
      // Device info
      { label: 'cudaGetDeviceProperties', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaGetDeviceProperties(&${1:prop}, ${2:device})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Get device properties' },
      { label: 'cudaGetDevice', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaGetDevice(&${1:device})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Get current device' },
      { label: 'cudaSetDevice', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaSetDevice(${1:device})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Set current device' },
      
      // Error handling
      { label: 'cudaGetLastError', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaGetLastError()', detail: 'Get last error' },
      { label: 'cudaGetErrorString', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cudaGetErrorString(${1:error})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Get error string' },
      { label: 'cudaSuccess', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'cudaSuccess', detail: 'Success return code' },
      
      // Kernel launch
      { label: 'kernel<<<blocks, threads>>>', kind: monaco.languages.CompletionItemKind.Snippet, insertText: '${1:kernel}<<<${2:numBlocks}, ${3:threadsPerBlock}>>>(${4:args})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Launch CUDA kernel' },
      { label: 'kernel<<<blocks, threads, shared>>>', kind: monaco.languages.CompletionItemKind.Snippet, insertText: '${1:kernel}<<<${2:numBlocks}, ${3:threadsPerBlock}, ${4:sharedMem}>>>(${5:args})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Launch kernel with shared memory' },
      { label: 'kernel<<<blocks, threads, shared, stream>>>', kind: monaco.languages.CompletionItemKind.Snippet, insertText: '${1:kernel}<<<${2:numBlocks}, ${3:threadsPerBlock}, ${4:sharedMem}, ${5:stream}>>>(${6:args})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Launch kernel with stream' },
      { label: 'dim3', kind: monaco.languages.CompletionItemKind.TypeParameter, insertText: 'dim3(${1:x}, ${2:y}, ${3:z})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: '3D dimension type' },
      
      // Cooperative groups
      { label: 'cooperative_groups', kind: monaco.languages.CompletionItemKind.Module, insertText: '#include <cooperative_groups.h>\nnamespace cg = cooperative_groups;', detail: 'Cooperative groups header' },
      { label: 'cg::this_thread_block', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cg::this_thread_block()', detail: 'Get thread block group' },
      { label: 'cg::tiled_partition', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cg::tiled_partition<${1:32}>(${2:block})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Partition into tiles' },
    ];

    const tritonCompletions = [
      // Triton basics
      { label: '@triton.jit', kind: monaco.languages.CompletionItemKind.Snippet, insertText: '@triton.jit\ndef ${1:kernel_name}(${2:args}):\n    ${3:pass}', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Triton JIT kernel decorator' },
      { label: 'tl.constexpr', kind: monaco.languages.CompletionItemKind.TypeParameter, insertText: 'tl.constexpr', detail: 'Compile-time constant' },
      
      // Program IDs
      { label: 'tl.program_id', kind: monaco.languages.CompletionItemKind.Function, insertText: 'tl.program_id(axis=${1:0})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Get program ID for axis' },
      { label: 'tl.num_programs', kind: monaco.languages.CompletionItemKind.Function, insertText: 'tl.num_programs(axis=${1:0})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Get number of programs for axis' },
      
      // Range and indexing
      { label: 'tl.arange', kind: monaco.languages.CompletionItemKind.Function, insertText: 'tl.arange(${1:0}, ${2:BLOCK_SIZE})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Create range of indices' },
      { label: 'tl.zeros', kind: monaco.languages.CompletionItemKind.Function, insertText: 'tl.zeros((${1:BLOCK_SIZE},), dtype=${2:tl.float32})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Create zero tensor' },
      
      // Memory operations
      { label: 'tl.load', kind: monaco.languages.CompletionItemKind.Function, insertText: 'tl.load(${1:ptr} + ${2:offsets}, mask=${3:mask})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Load from memory' },
      { label: 'tl.store', kind: monaco.languages.CompletionItemKind.Function, insertText: 'tl.store(${1:ptr} + ${2:offsets}, ${3:value}, mask=${4:mask})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Store to memory' },
      { label: 'tl.atomic_add', kind: monaco.languages.CompletionItemKind.Function, insertText: 'tl.atomic_add(${1:ptr}, ${2:val})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Atomic add' },
      
      // Math operations
      { label: 'tl.dot', kind: monaco.languages.CompletionItemKind.Function, insertText: 'tl.dot(${1:a}, ${2:b})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Matrix multiplication' },
      { label: 'tl.sum', kind: monaco.languages.CompletionItemKind.Function, insertText: 'tl.sum(${1:x}, axis=${2:0})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Sum reduction' },
      { label: 'tl.max', kind: monaco.languages.CompletionItemKind.Function, insertText: 'tl.max(${1:x}, axis=${2:0})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Max reduction' },
      { label: 'tl.exp', kind: monaco.languages.CompletionItemKind.Function, insertText: 'tl.exp(${1:x})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Exponential' },
      { label: 'tl.log', kind: monaco.languages.CompletionItemKind.Function, insertText: 'tl.log(${1:x})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Natural logarithm' },
      { label: 'tl.sqrt', kind: monaco.languages.CompletionItemKind.Function, insertText: 'tl.sqrt(${1:x})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Square root' },
      
      // Types
      { label: 'tl.float16', kind: monaco.languages.CompletionItemKind.TypeParameter, insertText: 'tl.float16', detail: 'Float16 type' },
      { label: 'tl.float32', kind: monaco.languages.CompletionItemKind.TypeParameter, insertText: 'tl.float32', detail: 'Float32 type' },
      { label: 'tl.int32', kind: monaco.languages.CompletionItemKind.TypeParameter, insertText: 'tl.int32', detail: 'Int32 type' },
      
      // Triton utilities
      { label: 'triton.cdiv', kind: monaco.languages.CompletionItemKind.Function, insertText: 'triton.cdiv(${1:x}, ${2:y})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Ceiling division' },
      
      // PyTorch integration
      { label: 'torch.empty_like', kind: monaco.languages.CompletionItemKind.Function, insertText: 'torch.empty_like(${1:x})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Create empty tensor like x' },
      { label: 'torch.rand', kind: monaco.languages.CompletionItemKind.Function, insertText: "torch.rand(${1:size}, device='cuda')", insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Random tensor' },
      { label: 'torch.zeros', kind: monaco.languages.CompletionItemKind.Function, insertText: "torch.zeros(${1:size}, device='cuda')", insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Zero tensor' },
      { label: 'torch.randn', kind: monaco.languages.CompletionItemKind.Function, insertText: "torch.randn(${1:size}, device='cuda', dtype=${2:torch.float32})", insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Random normal tensor' },
      { label: 'torch.matmul', kind: monaco.languages.CompletionItemKind.Function, insertText: 'torch.matmul(${1:a}, ${2:b})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Matrix multiplication' },
      { label: 'torch.cuda.synchronize', kind: monaco.languages.CompletionItemKind.Function, insertText: 'torch.cuda.synchronize()', detail: 'Synchronize CUDA device' },
      { label: 'torch.allclose', kind: monaco.languages.CompletionItemKind.Function, insertText: 'torch.allclose(${1:a}, ${2:b}, rtol=${3:1e-3}, atol=${4:1e-3})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Check if tensors are close' },
    ];

    // CUTE DSL Python completions (tiled GEMM patterns)
    const cuteDslCompletions = [
      // CUTE-style concepts in Python/Triton
      { label: 'BLOCK_M', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'BLOCK_M = ${1:128}', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Tile size M dimension' },
      { label: 'BLOCK_N', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'BLOCK_N = ${1:128}', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Tile size N dimension' },
      { label: 'BLOCK_K', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'BLOCK_K = ${1:32}', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Tile size K dimension' },
      
      // Layout/stride patterns
      { label: 'stride pattern', kind: monaco.languages.CompletionItemKind.Snippet, insertText: 'stride_${1:a}m, stride_${1:a}k = ${1:a}.stride(0), ${1:a}.stride(1)', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Get tensor strides (CUTE Layout)' },
      
      // Tile offset patterns
      { label: 'tile offsets', kind: monaco.languages.CompletionItemKind.Snippet, insertText: 'offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\noffs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'CUTE-style tile offset calculation' },
      
      // Pointer arithmetic (CUTE make_tensor equivalent)
      { label: 'ptr arithmetic 2D', kind: monaco.languages.CompletionItemKind.Snippet, insertText: '${1:a}_ptrs = ${1:a}_ptr + offs_m[:, None] * stride_${1:a}m + offs_k[None, :] * stride_${1:a}k', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'CUTE-style pointer calculation' },
      
      // Predicated load (CUTE copy with mask)
      { label: 'predicated load', kind: monaco.languages.CompletionItemKind.Snippet, insertText: 'mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)\n${1:tile} = tl.load(${1:tile}_ptrs, mask=mask, other=0.0)', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'CUTE-style predicated memory load' },
      
      // Accumulator init (CUTE make_fragment_like)
      { label: 'accumulator', kind: monaco.languages.CompletionItemKind.Snippet, insertText: 'acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'CUTE-style accumulator fragment' },
      
      // K-loop pattern
      { label: 'k-loop', kind: monaco.languages.CompletionItemKind.Snippet, insertText: 'for k_start in range(0, K, BLOCK_K):\n    k_offs = k_start + tl.arange(0, BLOCK_K)\n    # Load A and B tiles\n    # acc += tl.dot(a_tile, b_tile)', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'CUTE-style K-dimension loop' },
      
      // Grid launch pattern
      { label: 'grid config', kind: monaco.languages.CompletionItemKind.Snippet, insertText: 'grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'CUTE-style grid configuration' },
      
      // GEMM kernel template
      { label: 'cute gemm kernel', kind: monaco.languages.CompletionItemKind.Snippet, insertText: `@triton.jit
def cute_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        # Load and compute
        pass
    
    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)`, insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Full CUTE-style GEMM kernel template' },
    ];

    const cutlassCompletions = [
      // CUTLASS includes
      { label: '#include <cutlass/cutlass.h>', kind: monaco.languages.CompletionItemKind.Snippet, insertText: '#include <cutlass/cutlass.h>', detail: 'CUTLASS main header' },
      { label: '#include <cutlass/gemm/device/gemm.h>', kind: monaco.languages.CompletionItemKind.Snippet, insertText: '#include <cutlass/gemm/device/gemm.h>', detail: 'CUTLASS GEMM header' },
      
      // CUTE DSL
      { label: 'cute::make_tensor', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cute::make_tensor(${1:ptr}, ${2:layout})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Create CUTE tensor' },
      { label: 'cute::make_layout', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cute::make_layout(cute::make_shape(${1:M}, ${2:N}), cute::make_stride(${3:1}, ${4:M}))', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Create CUTE layout' },
      { label: 'cute::copy', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cute::copy(${1:src}, ${2:dst})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'CUTE copy operation' },
      { label: 'cute::gemm', kind: monaco.languages.CompletionItemKind.Function, insertText: 'cute::gemm(${1:tiled_mma}, ${2:tCrA}, ${3:tCrB}, ${4:tCrC})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'CUTE GEMM operation' },
      
      // Layouts
      { label: 'cutlass::layout::RowMajor', kind: monaco.languages.CompletionItemKind.TypeParameter, insertText: 'cutlass::layout::RowMajor', detail: 'Row-major layout' },
      { label: 'cutlass::layout::ColumnMajor', kind: monaco.languages.CompletionItemKind.TypeParameter, insertText: 'cutlass::layout::ColumnMajor', detail: 'Column-major layout' },
      
      // Architectures
      { label: 'cutlass::arch::Sm70', kind: monaco.languages.CompletionItemKind.TypeParameter, insertText: 'cutlass::arch::Sm70', detail: 'Volta architecture (V100)' },
      { label: 'cutlass::arch::Sm80', kind: monaco.languages.CompletionItemKind.TypeParameter, insertText: 'cutlass::arch::Sm80', detail: 'Ampere architecture (A100)' },
      { label: 'cutlass::arch::Sm90', kind: monaco.languages.CompletionItemKind.TypeParameter, insertText: 'cutlass::arch::Sm90', detail: 'Hopper architecture (H100)' },
      
      // HostTensor
      { label: 'cutlass::HostTensor', kind: monaco.languages.CompletionItemKind.Class, insertText: 'cutlass::HostTensor<${1:float}, ${2:cutlass::layout::RowMajor}> ${3:tensor}({${4:M}, ${5:N}})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, detail: 'Host tensor with device sync' },
    ];

    // Register completion providers
    monaco.languages.registerCompletionItemProvider('cpp', {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      provideCompletionItems: (model: any, position: any) => {
        const word = model.getWordUntilPosition(position);
        const range = {
          startLineNumber: position.lineNumber,
          endLineNumber: position.lineNumber,
          startColumn: word.startColumn,
          endColumn: word.endColumn,
        };
        
        const suggestions = [...cudaCompletions, ...cutlassCompletions].map(item => ({
          ...item,
          range,
        }));
        
        return { suggestions };
      },
    });

    monaco.languages.registerCompletionItemProvider('python', {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      provideCompletionItems: (model: any, position: any) => {
        const word = model.getWordUntilPosition(position);
        const range = {
          startLineNumber: position.lineNumber,
          endLineNumber: position.lineNumber,
          startColumn: word.startColumn,
          endColumn: word.endColumn,
        };
        
        const suggestions = [...tritonCompletions, ...cuteDslCompletions].map(item => ({
          ...item,
          range,
        }));
        
        return { suggestions };
      },
    });

    // Focus editor
    editor.focus();
  };

  useEffect(() => {
    if (editorRef.current) {
      const model = editorRef.current.getModel();
      if (model) {
        const monacoLang = languageMap[language];
        // @ts-ignore
        window.monaco?.editor.setModelLanguage(model, monacoLang);
      }
    }
  }, [language]);

  return (
    <MonacoEditor
      height="100%"
      language={languageMap[language]}
      value={code}
      onChange={(value) => onChange(value || '')}
      onMount={handleMount}
      options={{
        fontSize: 14,
        fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace",
        fontLigatures: true,
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        lineNumbers: 'on',
        renderLineHighlight: 'line',
        cursorBlinking: 'smooth',
        cursorSmoothCaretAnimation: 'on',
        smoothScrolling: true,
        padding: { top: 16, bottom: 16 },
        automaticLayout: true,
        tabSize: 4,
        insertSpaces: true,
        wordWrap: 'off',
        bracketPairColorization: { enabled: true },
        guides: {
          bracketPairs: true,
          indentation: true,
        },
        suggest: {
          showKeywords: true,
          showSnippets: true,
          showFunctions: true,
          showVariables: true,
        },
        quickSuggestions: {
          other: true,
          comments: false,
          strings: false,
        },
      }}
    />
  );
}
