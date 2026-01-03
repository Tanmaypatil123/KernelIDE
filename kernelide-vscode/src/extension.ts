import * as vscode from 'vscode';
import { SidebarProvider } from './SidebarProvider';

export function activate(context: vscode.ExtensionContext) {
    const sidebarProvider = new SidebarProvider(context.extensionUri, context);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'kernelide.sidebarView',
            sidebarProvider
        )
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('kernelide.runKernel', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showErrorMessage('No active editor found');
                return;
            }
            const code = editor.document.getText();
            sidebarProvider.runKernel(code);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('kernelide.configure', () => {
            vscode.commands.executeCommand('workbench.action.openSettings', 'kernelide');
        })
    );
}

export function deactivate() {}
