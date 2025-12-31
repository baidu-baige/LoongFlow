/* eslint-disable no-lonely-if,complexity,max-len,max-statements,max-lines,no-use-before-define */
function initVisualizer() {

    // ===== Theme Management =====
    let isDarkTheme = true; // Default to dark theme

    // ===== Monaco Editor Instances =====
    let codeEditorInstance = null;
    let diffEditorInstance = null;
    let monacoEditorLoaded = false;

    // Initialize Monaco Editor
    function initMonacoEditor() {
        if (monacoEditorLoaded && typeof monaco !== 'undefined') {
            return Promise.resolve();
        }

        return new Promise((resolve, reject) => {
            if (typeof require === 'undefined') {
                reject(new Error('Monaco Editor loader not found. Please ensure loader.js is loaded.'));
                return;
            }

            // Configure Monaco Editor paths
            require.config({
                paths: {
                    vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs',
                },
            });

            // Load Monaco Editor
            require(['vs/editor/editor.main'], () => {
                monacoEditorLoaded = true;
                resolve();
            }, err => {
                console.error('Failed to load Monaco Editor:', err);
                reject(err);
            });
        });
    }

    function initTheme() {
    // Load theme from localStorage or default to dark
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            isDarkTheme = false;
            document.body.classList.remove('dark-theme');
            updateThemeIcon('☀️');
            updatePrismTheme('light');
        } else {
            isDarkTheme = true;
            document.body.classList.add('dark-theme');
            updateThemeIcon('🌙');
            updatePrismTheme('dark');
        }
    }

    function toggleTheme() {
        isDarkTheme = !isDarkTheme;
        if (isDarkTheme) {
            document.body.classList.add('dark-theme');
            localStorage.setItem('theme', 'dark');
            updateThemeIcon('🌙');
            updatePrismTheme('dark');
        } else {
            document.body.classList.remove('dark-theme');
            localStorage.setItem('theme', 'light');
            updateThemeIcon('☀️');
            updatePrismTheme('light');
        }
        // Update node text colors
        updateNodeTextColors();
        // Update selected node marker color
        if (selectedNodeId) {
            updateSelectedNodeMarker();
        }
        // Update Monaco Editor theme
        updateMonacoTheme();
        // Update score chart colors
        updateScoreChart();
    }

    let selectedNodeData = null;
    let selectedNodeId = null; // Track the ID of the currently selected node

    // Save current zoom transform
    let currentTransform = d3.zoomIdentity;

    const tooltip = d3.select('#tooltip');
    const checkpointSelect = document.getElementById('checkpointSelect');
    const checkpointSlider = document.getElementById('checkpointSlider');
    const currentCheckpointSpan = document.getElementById('currentCheckpoint');
    const totalCheckpointsSpan = document.getElementById('totalCheckpoints');
    const playPauseBtn = document.getElementById('playPauseBtn');
    const playPauseIcon = document.getElementById('playPauseIcon');
    const playbackSpeedSelect = document.getElementById('playbackSpeed');
    const loadingState = document.getElementById('loadingState');
    const treeContainer = document.getElementById('tree-container');
    const scoreChartCard = document.getElementById('scoreChartCard');
    const scoreChartContainer = document.getElementById('scoreChartContainer');
    const scoreChartEmpty = document.getElementById('scoreChartEmpty');
    const scoreChartSummary = document.getElementById('scoreChartSummary');
    const scoreChartSvg = d3.select('#scoreChart');
    const scoreChartExpandBtn = document.getElementById('scoreChartExpandBtn');
    const scoreChartExpandedSection = document.getElementById('scoreChartExpandedSection');
    const scoreChartExpandedContainer = document.getElementById('scoreChartExpandedContainer');
    const scoreChartExpandedEmpty = document.getElementById('scoreChartExpandedEmpty');
    const scoreChartExpandedSummary = document.getElementById('scoreChartExpandedSummary');
    const scoreChartExpandedSvg = d3.select('#scoreChartExpanded');
    const scoreChartCollapseBtn = document.getElementById('scoreChartCollapse');
    let isScoreChartExpanded = false;

    function updatePrismTheme(theme) {
        const lightTheme = document.getElementById('prism-theme-light');
        const darkTheme = document.getElementById('prism-theme-dark');
        if (theme === 'dark') {
            if (lightTheme) {
                lightTheme.disabled = true;
            }
            if (darkTheme) {
                darkTheme.disabled = false;
            }
        } else {
            if (lightTheme) {
                lightTheme.disabled = false;
            }
            if (darkTheme) {
                darkTheme.disabled = true;
            }
        }
        // Re-highlight code after theme change
        if (selectedNodeData) {
            updateCodeTab(selectedNodeData);
        }
    }

    function updateThemeIcon(icon) {
        const themeIcon = document.querySelector('.theme-icon');
        if (themeIcon) {
            themeIcon.textContent = icon;
        }
    }

    function updateNodeTextColors() {
        const textColor = isDarkTheme ? '#e0e0e0' : '#1a1a1a';
        d3.selectAll('.node text').style('fill', textColor);
    }

    // Update the visual marker of the selected node
    function updateSelectedNodeMarker() {
        // First, remove all existing selection markers
        removeSelectedNodeMarker();

        if (!selectedNodeId) {
            return;}

        // Set circle color based on theme: white for dark mode, gray for light mode
        const markerColor = isDarkTheme ? '#ffffff' : '#666666';

        // Find the corresponding node in the tree view and island view
        const nodeGroups = d3.selectAll('.node');
        nodeGroups.each(function (d) {
            const nodeData = d.data || d;
            const nodeId = nodeData.solution_id || nodeData.id;

            if (nodeId === selectedNodeId) {
                const nodeGroup = d3.select(this);
                const circle = nodeGroup.select('circle');
                const nodeSize = parseFloat(circle.attr('r')) || getNodeSize(d);

                // Add a circle as the selection marker (add it after the node circle, to be displayed above)
                nodeGroup.append('circle')
                    .attr('class', 'selected-marker')
                    .attr('r', nodeSize + 3)
                    .attr('fill', 'none')
                    .attr('stroke', markerColor)
                    .attr('stroke-width', 2)
                    .attr('opacity', 1);
            }
        });
    }

    // Remove the visual marker of the selected node
    function removeSelectedNodeMarker() {
        d3.selectAll('.selected-marker').remove();
    }

    // Initialize theme on load
    initTheme();

    // Theme toggle button
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }

    // ===== Node Details Panel =====


    function showNodeDetails(nodeData) {
        selectedNodeData = nodeData;
        selectedNodeId = nodeData.solution_id || nodeData.id;
        const panel = document.getElementById('nodeDetailsPanel');
        if (!panel) {
            return;
        }

        // Show the right area
        const rightArea = document.querySelector('.app-right');
        if (rightArea) {
            rightArea.style.display = 'block';
        }

        panel.classList.remove('hidden');

        // Update the title with the node ID
        const nodeTitle = document.getElementById('nodeTitle');
        if (nodeTitle && nodeData && nodeData.id) {
            nodeTitle.textContent = nodeData.id;
        }

        // Update tab content
        updateCodeTab(nodeData);
        updateDiffTab(nodeData);
        updateDetailTab(nodeData);

        // Update selected node visual marker
        updateSelectedNodeMarker();
    }

    function hideNodeDetails() {
        const panel = document.getElementById('nodeDetailsPanel');
        if (panel) {
            panel.classList.add('hidden');
        }

        // Hide the right area
        const rightArea = document.querySelector('.app-right');
        if (rightArea) {
            rightArea.style.display = 'none';
        }

        selectedNodeData = null;
        selectedNodeId = null;

        // Remove the visual marker of the selected node
        removeSelectedNodeMarker();

        // Dispose Monaco Editor instances to free memory
        if (codeEditorInstance) {
            try {
                codeEditorInstance.dispose();
            } catch (e) {
                console.warn('Error disposing code editor:', e);
            }
            codeEditorInstance = null;
        }

        if (diffEditorInstance) {
            try {
                diffEditorInstance.dispose();
            } catch (e) {
                console.warn('Error disposing diff editor:', e);
            }
            diffEditorInstance = null;
        }
    }

    // Detect code language from content
    function detectLanguage(code) {
        if (!code || typeof code !== 'string') {
            return 'text';
        }

        const codeLower = code.trim().toLowerCase();

        // Python detection
        if (codeLower.includes('def ') || codeLower.includes('import ')
        || codeLower.includes('from ') || codeLower.includes('class ')
        || codeLower.startsWith('#!/usr/bin/env python') || codeLower.startsWith('#!') && codeLower.includes('python')) {
            return 'python';
        }

        // JavaScript/TypeScript detection
        if (codeLower.includes('function ') || codeLower.includes('const ')
        || codeLower.includes('let ') || codeLower.includes('var ')
        || codeLower.includes('=>') || codeLower.includes('export ')
        || codeLower.includes('interface ') || codeLower.includes('type ')) {
            return 'javascript';
        }

        // JSON detection
        if ((codeLower.startsWith('{') && codeLower.endsWith('}'))
        || (codeLower.startsWith('[') && codeLower.endsWith(']'))) {
            try {
                JSON.parse(code);
                return 'json';
            } catch (e) {
            // Not valid JSON, continue
            }
        }

        // Shell/Bash detection
        if (codeLower.startsWith('#!/bin/bash') || codeLower.startsWith('#!/bin/sh')
        || codeLower.includes('$(') || codeLower.includes('${')) {
            return 'bash';
        }

        // YAML detection
        if (codeLower.includes('---') || (codeLower.includes(':') && codeLower.includes('\n'))) {
            return 'yaml';
        }

        // Markdown detection
        if (codeLower.includes('# ') || codeLower.includes('## ')
        || codeLower.includes('**') || codeLower.includes('```')) {
            return 'markdown';
        }

        // Default to text
        return 'text';
    }

    async function updateCodeTab(nodeData) {
        const codeContent = document.getElementById('codeContent');
        if (!codeContent) {
            return;}

        // Get code from node data
        const code = nodeData.solution || '';

        // Detect language
        const language = detectLanguage(code);

        // Map language to Monaco language ID
        const monacoLanguage = mapLanguageToMonaco(language);

        try {
        // Initialize Monaco Editor if not already loaded
            await initMonacoEditor();

            // Destroy existing editor if it exists
            if (codeEditorInstance) {
                codeEditorInstance.dispose();
                codeEditorInstance = null;
            }

            // Create new Monaco Editor instance
            codeEditorInstance = monaco.editor.create(codeContent, {
                value: code,
                language: monacoLanguage,
                theme: isDarkTheme ? 'vs-dark' : 'vs',
                readOnly: true,
                automaticLayout: true,
                minimap: {enabled: false},
                scrollBeyondLastLine: false,
                fontSize: 13,
                lineNumbers: 'on',
                wordWrap: 'on',
                folding: true,
                renderWhitespace: 'selection',
            });

            // Update theme
            updateMonacoTheme();

        } catch (error) {
            console.error('Failed to initialize Monaco Editor:', error);
            codeContent.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">编辑器加载失败: ${error.message || '未知错误'}</div>`;
        }
    }

    // Map language names to Monaco Editor language IDs
    function mapLanguageToMonaco(language) {
        const languageMap = {
            'python': 'python',
            'javascript': 'javascript',
            'typescript': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'csharp': 'csharp',
            'go': 'go',
            'rust': 'rust',
            'php': 'php',
            'ruby': 'ruby',
            'swift': 'swift',
            'kotlin': 'kotlin',
            'html': 'html',
            'css': 'css',
            'json': 'json',
            'xml': 'xml',
            'yaml': 'yaml',
            'markdown': 'markdown',
            'sql': 'sql',
            'shell': 'shell',
            'bash': 'shell',
            'sh': 'shell',
        };
        return languageMap[language.toLowerCase()] || 'plaintext';
    }

    // Copy code to clipboard
    function copyCodeToClipboard() {
        let code = '';

        // Get code from Monaco Editor if available
        if (codeEditorInstance) {
            code = codeEditorInstance.getValue();
        } else {
        // Fallback: try to get from diff editor
            if (diffEditorInstance && diffEditorInstance.getModifiedEditor) {
                code = diffEditorInstance.getModifiedEditor().getValue();
            } else {
                showCopyFeedback('未找到代码', false);
                return;
            }
        }

        if (!code || !code.trim()) {
            showCopyFeedback('代码为空', false);
            return;
        }

        // Use Clipboard API if available
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(code).then(() => {
                showCopyFeedback('已复制到剪贴板', true);
            }).catch(err => {
                console.error('复制失败:', err);
                // Fallback to old method
                fallbackCopyTextToClipboard(code);
            });
        } else {
        // Fallback for older browsers
            fallbackCopyTextToClipboard(code);
        }
    }

    // Show copy feedback
    function showCopyFeedback(message, success) {
        const copyBtn = document.getElementById('copyCodeBtn');
        if (!copyBtn) {
            return;}

        const originalText = copyBtn.querySelector('.copy-text').textContent;
        const originalIcon = copyBtn.querySelector('.copy-icon').textContent;

        // Update button text and icon
        copyBtn.querySelector('.copy-text').textContent = message;
        copyBtn.querySelector('.copy-icon').textContent = success ? '✓' : '✗';

        // Add success/error class
        if (success) {
            copyBtn.classList.add('copy-success');
        } else {
            copyBtn.classList.add('copy-error');
        }

        // Reset after 2 seconds
        setTimeout(() => {
            copyBtn.querySelector('.copy-text').textContent = originalText;
            copyBtn.querySelector('.copy-icon').textContent = originalIcon;
            copyBtn.classList.remove('copy-success', 'copy-error');
        }, 2000);
    }


    // Fallback copy method for older browsers
    function fallbackCopyTextToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();

        try {
            const successful = document.execCommand('copy');
            if (successful) {
                showCopyFeedback('已复制到剪贴板', true);
            } else {
                showCopyFeedback('复制失败', false);
            }
        } catch (err) {
            console.error('复制失败:', err);
            showCopyFeedback('复制失败', false);
        } finally {
            document.body.removeChild(textArea);
        }
    }


    // Update the theme of Monaco Editor
    function updateMonacoTheme() {
        const theme = isDarkTheme ? 'vs-dark' : 'vs';

        if (codeEditorInstance) {
            monaco.editor.setTheme(theme);
        }

        if (diffEditorInstance) {
            monaco.editor.setTheme(theme);
        }
    }

    async function updateDiffTab(nodeData) {
        const diffContent = document.getElementById('diffContent');
        if (!diffContent) {
            return;}

        // Show loading status
        diffContent.innerHTML = '<div style="padding: 20px; text-align: center;">正在加载差异...</div>';

        const currentId = nodeData.solution_id || nodeData.id;
        const parentId = nodeData.parent_id;

        // If there is no parent, display a prompt message
        if (!parentId) {
            diffContent.innerHTML = '<div style="padding: 20px; text-align: center;">无父节点，无法显示差异</div>';
            return;
        }

        // If there is no checkpoint ID，display error
        if (!currentCheckpointId) {
            diffContent.innerHTML = '<div style="padding: 20px; text-align: center;">无法获取checkpoint信息</div>';
            return;
        }

        try {
        // Call the backend API to get the diff
            const url = `/api/checkpoints/${currentCheckpointId}/diff?current_node_id=${encodeURIComponent(currentId)}&parent_node_id=${encodeURIComponent(parentId)}`;
            const diffData = await fetchJSON(url);

            if (!diffData.has_changes) {
                diffContent.innerHTML = `<div style="padding: 20px; text-align: center;">${diffData.message || '代码无变化'}</div>`;
                return;
            }

            // Get the original code and current code
            const parentCode = diffData.parent_code || '';
            const currentCode = diffData.current_code || '';

            // Initialize Monaco Editor if not already loaded
            await initMonacoEditor();

            // Clear the container
            diffContent.innerHTML = '';

            // If an editor instance already exists, destroy it first
            if (diffEditorInstance) {
                try {
                    diffEditorInstance.dispose();
                } catch (e) {
                    console.warn('Error destroying Monaco diff editor:', e);
                }
                diffEditorInstance = null;
            }

            // Create a Monaco Editor diff view
            diffEditorInstance = monaco.editor.createDiffEditor(diffContent, {
                theme: isDarkTheme ? 'vs-dark' : 'vs',
                readOnly: true,
                automaticLayout: true,
                minimap: {enabled: false},
                scrollBeyondLastLine: false,
                fontSize: 13,
                lineNumbers: 'on',
                wordWrap: 'on',
                renderSideBySide: true,
                enableSplitViewResizing: true,
            });

            // Set the original code and modified code
            const originalModel = monaco.editor.createModel(parentCode, 'python');
            const modifiedModel = monaco.editor.createModel(currentCode, 'python');
            diffEditorInstance.setModel({
                original: originalModel,
                modified: modifiedModel,
            });

            // Update the theme
            updateMonacoTheme();

        } catch (error) {
            console.error('获取diff失败:', error);
            diffContent.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">获取差异失败: ${error.message || '未知错误'}</div>`;
        }
    }

    // HTML escape function
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function updateDetailTab(nodeData) {
    // remove the children field
        const omittedNodeData = JSON.parse(JSON.stringify(nodeData));
        delete omittedNodeData.children;
        delete omittedNodeData.solution;
        delete omittedNodeData.id;
        delete omittedNodeData.name;

        const formattedContent = document.getElementById('detailFormattedContent');
        const rawContent = document.getElementById('rawJsonContent');

        if (!formattedContent || !rawContent) {
            return;
        }

        // Update Raw JSON
        // Create a copy without the bulky solution code for cleaner JSON view if desired,
        // but user asked for raw details. The solution code might be large.
        // Let's keep it as is for "Raw".
        rawContent.textContent = JSON.stringify(omittedNodeData, null, 2);
        if (window.Prism) {
            rawContent.className = 'code-content language-json';
            Prism.highlightElement(rawContent);
        }

        // Update Formatted View
        let html = '<div class="detail-list" style="display: flex; flex-direction: column; gap: 8px;">';

        // Helper to format value
        function formatValue(key, value) {
            if (key === 'timestamp') {
                let ts = Number(value);
                if (!isNaN(ts)) {
                    if (ts < 1e11) {
                        ts *= 1000;
                    } // Assuming seconds if < 100 billion
                    const date = new Date(ts);
                    const year = date.getFullYear();
                    const month = String(date.getMonth() + 1).padStart(2, '0');
                    const day = String(date.getDate()).padStart(2, '0');
                    const hours = String(date.getHours()).padStart(2, '0');
                    const minutes = String(date.getMinutes()).padStart(2, '0');
                    const seconds = String(date.getSeconds()).padStart(2, '0');
                    return `<span style="color: #ce9178;">"${year}-${month}-${day} ${hours}:${minutes}:${seconds}"</span>`;
                }
            }

            if (key === 'evaluation') {
            // JSON formatter render
                let jsonVal = null;
                try {
                    const obj = typeof value === 'string' ? JSON.parse(value) : value;
                    jsonVal = JSON.stringify(obj, null, 2);
                } catch (e) {
                    jsonVal = String(value);
                }
                return `<pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word; color: #ce9178; max-height: 300px; overflow-y: auto; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px;">${escapeHtml(jsonVal)}</pre>`;
            }

            if (key === 'generate_plan' || key === 'summary') {
                return `<div style="max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px;">${escapeHtml(value)}</div>`;
            }

            if (value === null) {
                return '<span style="color: #888;">null</span>';
            }
            if (value === undefined) {
                return '<span style="color: #888;">undefined</span>';
            }
            if (typeof value === 'boolean') {
                return `<span style="color: ${value ? '#4caf50' : '#f44336'};">${value}</span>`;
            }
            if (typeof value === 'number') {
                return `<span style="color: #4fc1ff;">${value}</span>`;
            }
            if (typeof value === 'string') {
                if (key === 'solution') {
                    return '<span style="color: #888;">(See Code Tab)</span>';
                }
                if (value.length > 200) {
                    return `<span style="color: #ce9178;" title="${escapeHtml(value)}">"${escapeHtml(value.substring(0, 200))}..."</span>`;
                }
                return `<span style="color: #ce9178;">"${escapeHtml(value)}"</span>`;
            }
            if (Array.isArray(value)) {
                return `[<span style="color: #9cdcfe;">Array(${value.length})</span>] ${JSON.stringify(value)}`;
            }
            if (typeof value === 'object') {
                return `{<span style="color: #9cdcfe;">Object</span>} ${JSON.stringify(value)}`;
            }
            return String(value);
        }

        const priorityKeys = ['solution_id', 'score', 'generation', 'iteration', 'island_id', 'parent_id', 'is_best', 'is_elite', 'is_island_best'];
        const allKeys = Object.keys(omittedNodeData).sort();
        const otherKeys = allKeys.filter(k => !priorityKeys.includes(k));
        const orderedKeys = [...priorityKeys, ...otherKeys];

        orderedKeys.forEach(key => {
            if (omittedNodeData.hasOwnProperty(key)) {
                const value = nodeData[key];
                html += `
                <div class="detail-item" style="display: flex; padding: 6px 0; border-bottom: 1px solid #333; align-items: flex-start;">
                    <div class="detail-key" style="width: 140px; font-weight: bold; color: #9cdcfe; flex-shrink: 0;">${key}</div>
                    <div class="detail-value" style="flex-grow: 1; word-break: break-all; line-height: 1.4;">${formatValue(key, value)}</div>
                </div>
            `;
            }
        });

        html += '</div>';
        formattedContent.innerHTML = html;
    }

    // Initialize Detail View Mode Switching
    function initDetailViewModes() {
        const modeBtns = document.querySelectorAll('.detail-mode-btn');
        const contents = document.querySelectorAll('.detail-content-mode');

        modeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.dataset.mode;

                // Update buttons
                modeBtns.forEach(b => {
                    b.classList.remove('active');
                    b.removeAttribute('style'); // Clean up any potential inline styles
                });
                btn.classList.add('active');

                // Update content
                contents.forEach(c => {
                    if (c.id === (mode === 'formatted' ? 'detailFormattedContent' : 'detailRawContent')) {
                        c.classList.remove('hidden');
                        c.classList.add('active');
                    } else {
                        c.classList.add('hidden');
                        c.classList.remove('active');
                    }
                });
            });
        });

        // Set initial state style
        const activeBtn = document.querySelector('.detail-mode-btn.active');
        if (activeBtn) {
            activeBtn.removeAttribute('style'); // Clean up any potential inline styles
        }
    }

    // Tab switching
    document.addEventListener('DOMContentLoaded', () => {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(btn => {
            btn.addEventListener('click', function () {
                const tabName = this.dataset.tab;

                // Update button states
                tabButtons.forEach(b => b.classList.remove('active'));
                this.classList.add('active');

                // Update content visibility
                tabContents.forEach(content => {
                    content.classList.remove('active');
                });

                const targetTab = document.getElementById(tabName + 'Tab');
                if (targetTab) {
                    targetTab.classList.add('active');

                    // Re-load code when switching to code tab
                    if (tabName === 'code' && selectedNodeData) {
                    // Re-invoke updateCodeTab to ensure the data is up-to-date
                        updateCodeTab(selectedNodeData).catch(err => {
                            console.error('Failed to update code tab:', err);
                        });
                    }

                    // Re-load diff when switching to diff tab
                    if (tabName === 'diff' && selectedNodeData) {
                    // Re-invoke updateDiffTab to ensure the data is up-to-date
                        updateDiffTab(selectedNodeData).catch(err => {
                            console.error('Failed to update diff tab:', err);
                        });
                    }
                }
            });
        });

        // Close button
        const closeBtn = document.getElementById('closeNodePanel');
        if (closeBtn) {
            closeBtn.addEventListener('click', hideNodeDetails);
        }

        // Copy code button
        const copyCodeBtn = document.getElementById('copyCodeBtn');
        if (copyCodeBtn) {
            copyCodeBtn.addEventListener('click', copyCodeToClipboard);
        }

        initDetailViewModes();
    });

    const config = {
        orientation: 'vertical',
        nodeSize: 6,
        showLabels: false,
        width: 1200,
        height: 600,
        viewMode: 'tree', // 'tree' or 'island'
    };

    // Function to get the actual size of the container
    function getContainerSize() {
        if (!treeContainer) {
            return {width: config.width, height: config.height};
        }
        const rect = treeContainer.getBoundingClientRect();
        return {
            width: Math.max(rect.width || config.width, 400),
            height: Math.max(rect.height || config.height, 300),
        };
    }

    // Function to update the SVG size
    function updateSVGSize() {
        const size = getContainerSize();
        svg.attr('width', size.width)
            .attr('height', size.height);
        config.width = size.width;
        config.height = size.height;
    }

    const svg = d3.select('#tree-container')
        .append('svg')
        .attr('width', config.width)
        .attr('height', config.height);

    // Initialize SVG size
    updateSVGSize();

    const g = svg.append('g')
        .attr('transform', 'translate(40,20)');

    // Create zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.1, 4]) // Zoom range: 10% to 400%
        .on('zoom', event => {
        // Apply the zoom transform directly
            g.attr('transform', event.transform);
        });

    // Apply the zoom behavior to the SVG
    svg.call(zoom);

    // Helper function to calculate tooltip position
    function calculateTooltipPosition(nodeX, nodeY) {
    // Get current zoom transform
        const transform = d3.zoomTransform(svg.node());

        // Node coordinates are relative to the g group, and the g group's transform is the zoom transform
        // So apply the transform directly to convert node coordinates to screen coordinates
        const screenX = transform.applyX(nodeX);
        const screenY = transform.applyY(nodeY);

        // Get the position of the SVG and visualization-area
        const svgRect = svg.node().getBoundingClientRect();
        const visualizationArea = document.querySelector('.visualization-area');
        if (!visualizationArea) {
            return {x: screenX, y: screenY};
        }
        const areaRect = visualizationArea.getBoundingClientRect();

        // Calculate coordinates relative to the visualization-area
        const relativeX = screenX + (svgRect.left - areaRect.left);
        const relativeY = screenY + (svgRect.top - areaRect.top);

        return {x: relativeX, y: relativeY};
    }

    let treeData = null;
    let islandsData = null;
    let checkpointsList = [];
    let scoreHistory = [];
    let currentCheckpointIndex = 0;
    let currentCheckpointId = null; // Currently selected checkpoint ID
    let isPlaying = false;
    let playbackInterval = null;
    let playbackSpeed = 1000;

    const islandColors = [
        '#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a',
    ];

    function showExpandedScoreChart() {
        if (!scoreChartExpandedSection) {
            return;
        }
        isScoreChartExpanded = true;
        if (scoreChartCard) {
            scoreChartCard.classList.add('hidden');
        }
        scoreChartExpandedSection.classList.remove('hidden');
        updateScoreChart();
        // Smooth scroll to the zoomed area for easy viewing
        scoreChartExpandedSection.scrollIntoView({behavior: 'smooth', block: 'start'});
    }

    function hideExpandedScoreChart() {
        if (!scoreChartExpandedSection) {
            return;
        }
        isScoreChartExpanded = false;
        scoreChartExpandedSection.classList.add('hidden');
        if (scoreChartCard) {
            scoreChartCard.classList.remove('hidden');
        }
        updateScoreChart();
    }

    function setLoading(isLoading, message) {
        if (!loadingState) {
            return;
        }
        if (isLoading) {
            loadingState.textContent = message || '';
            loadingState.classList.remove('hidden');
        } else {
            loadingState.classList.add('hidden');
        }
    }

    function setTreeContainerVisible(visible) {
        if (!treeContainer) {
            return;
        }
        if (visible) {
            treeContainer.style.display = '';
        } else {
            treeContainer.style.display = 'none';
        }
    }

    function updateStats(stats = {}) {
        document.getElementById('totalSolutions').textContent =
        stats.total_solutions ?? '--';
        document.getElementById('validSolutions').textContent =
        stats.total_valid_solutions ?? '--';
        const bestScore = stats.best_score;
        document.getElementById('bestScore').textContent =
        typeof bestScore === 'number' ? bestScore.toFixed(4) : '--';
        document.getElementById('iterations').textContent =
        stats.last_iteration ?? '--';
    }

    function renderScoreChart(history, {
        container = scoreChartContainer,
        svg = scoreChartSvg,
        emptyEl = scoreChartEmpty,
        summaryEl = scoreChartSummary,
        heightOverride = null,
    } = {}) {
        if (!container || !svg || !svg.node()) {
            return;
        }
        if (container.classList && container.classList.contains('hidden')) {
            return; // Do not render when the container is hidden to avoid flickering
        }

        const width = container.clientWidth || 320;
        const height = heightOverride || container.clientHeight || 220;
        svg.attr('width', width).attr('height', height);
        svg.selectAll('*').remove();

        if (!history || history.length === 0) {
            if (emptyEl) {
                emptyEl.classList.remove('hidden');
            }
            if (summaryEl) {
                summaryEl.textContent = '';
            }
            return;
        }

        if (emptyEl) {
            emptyEl.classList.add('hidden');
        }

        const margin = {top: 16, right: 14, bottom: 36, left: 52};
        const innerWidth = Math.max(20, width - margin.left - margin.right);
        const innerHeight = Math.max(20, height - margin.top - margin.bottom);

        const sortedHistory = [...history].sort((a, b) => {
            if (a.iteration === b.iteration) {
                return a.score - b.score;
            }
            return a.iteration - b.iteration;
        });
        const uniqueIterations = Array.from(new Set(sortedHistory.map(d => d.iteration)));
        const maxXTicks = 10;
        const tickValues = (() => {
            if (uniqueIterations.length <= maxXTicks) {
                return uniqueIterations;
            }
            const step = Math.ceil(uniqueIterations.length / maxXTicks);
            const picked = uniqueIterations.filter((_, idx) => idx % step === 0);
            if (picked[0] !== uniqueIterations[0]) {
                picked.unshift(uniqueIterations[0]);
            }
            const lastVal = uniqueIterations[uniqueIterations.length - 1];
            if (picked[picked.length - 1] !== lastVal) {
                picked.push(lastVal);
            }
            return picked;
        })();

        const xExtent = d3.extent(sortedHistory, d => d.iteration);
        const yExtent = d3.extent(sortedHistory, d => d.score);

        const xScale = d3.scaleLinear()
            .domain(xExtent)
            .nice()
            .range([0, innerWidth]);

        const yScale = d3.scaleLinear()
            .domain(yExtent)
            .nice()
            .range([innerHeight, 0]);

        const accentColor = getComputedStyle(document.documentElement).getPropertyValue('--accent-color')?.trim() || '#67b7ff';
        const bestColor = getComputedStyle(document.documentElement).getPropertyValue('--success-color')?.trim() || '#43e97b';
        const axisColor = getComputedStyle(document.documentElement).getPropertyValue('--text-secondary')?.trim() || '#8a8a8a';
        const gridColor = getComputedStyle(document.documentElement).getPropertyValue('--border-color')?.trim() || '#2e2e2e';

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // grid lines
        g.append('g')
            .attr('class', 'y-grid')
            .call(d3.axisLeft(yScale)
                .ticks(5)
                .tickSize(-innerWidth)
                .tickFormat(''))
            .selectAll('line')
            .attr('stroke', gridColor)
            .attr('stroke-dasharray', '4 4')
            .attr('opacity', 0.6);

        const line = d3.line()
            .x(d => xScale(d.iteration))
            .y(d => yScale(d.score))
            .curve(d3.curveMonotoneX);

        g.append('path')
            .datum(sortedHistory)
            .attr('fill', 'none')
            .attr('stroke', accentColor)
            .attr('stroke-width', 1.6)
            .attr('d', line);

        g.selectAll('.score-point')
            .data(sortedHistory)
            .enter()
            .append('circle')
            .attr('class', 'score-point')
            .attr('cx', d => xScale(d.iteration))
            .attr('cy', d => yScale(d.score))
            .attr('r', d => (d.is_best ? 4 : 3))
            .attr('fill', d => (d.is_best ? bestColor : accentColor))
            .attr('opacity', 0.9);

        g.append('g')
            .attr('transform', `translate(0,${innerHeight})`)
            .call(d3.axisBottom(xScale)
                // When there is too much data, display at intervals
                .tickValues(tickValues)
                .tickFormat(d3.format('d')))
            .call(gAxis => gAxis.selectAll('text').attr('fill', axisColor))
            .call(gAxis => gAxis.selectAll('line').attr('stroke', axisColor))
            .call(gAxis => gAxis.selectAll('path').attr('stroke', axisColor));

        g.append('g')
            .call(d3.axisLeft(yScale).ticks(6))
            .call(gAxis => gAxis.selectAll('text').attr('fill', axisColor))
            .call(gAxis => gAxis.selectAll('line').attr('stroke', axisColor))
            .call(gAxis => gAxis.selectAll('path').attr('stroke', axisColor));

        const latest = sortedHistory[sortedHistory.length - 1];
        const best = sortedHistory.reduce((max, cur) => Math.max(max, cur.score), Number.NEGATIVE_INFINITY);
        if (summaryEl) {
            summaryEl.textContent = `最新迭代 ${latest.iteration}：${latest.score.toFixed(4)} ｜ 最高分 ${best.toFixed(4)}`;
        }
    }

    function updateScoreChart(history = scoreHistory) {
        renderScoreChart(history);
        if (isScoreChartExpanded) {
            renderScoreChart(history, {
                container: scoreChartExpandedContainer,
                svg: scoreChartExpandedSvg,
                emptyEl: scoreChartExpandedEmpty,
                summaryEl: scoreChartExpandedSummary,
                heightOverride: 420,
            });
        }
    }

    function getNodeColor(d) {
        const islandId = d.data.island_id || 0;
        return islandColors[islandId % islandColors.length];
    }

    function getNodeSize(d) {
        let size = config.nodeSize;
        if (d.data.is_best) {
            size = Math.round(size * 1.5);
        }
        if (d.data.is_elite) {
            size = Math.round(size * 1.3);
        }
        if (d.data.is_island_best) {
            size = Math.round(size * 1.2);
        }
        return size;
    }

    function updateIslandLegend() {
        const islandLegendItems = document.getElementById('islandLegendItems');
        if (!islandLegendItems) {
            return;
        }

        islandLegendItems.innerHTML = '';

        if (!islandsData || islandsData.length === 0) {
            return;
        }

        islandsData.forEach((islandSolutions, islandIndex) => {
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            legendItem.innerHTML = `
            <div class="legend-color island-${islandIndex % 5}"></div>
            <span class="legend-text">Island ${islandIndex}</span>
        `;
            islandLegendItems.appendChild(legendItem);
        });
    }

    function updateVisualization() {
        g.selectAll('*').remove();

        // Update SVG size to fit the container
        updateSVGSize();

        if (!treeData) {
        // Reset zoom
            currentTransform = d3.zoomIdentity;
            svg.call(zoom.transform, currentTransform);
            g.attr('transform', 'translate(40,20)');
            updateIslandLegend();
            return;
        }

        // Update island legend
        updateIslandLegend();

        // Select the render function based on the view mode
        if (config.viewMode === 'island') {
            updateIslandVisualization();
        } else {
            updateTreeVisualization();
        }
    }

    // Tree visualization render function
    function updateTreeVisualization() {
        const isVertical = config.orientation === 'vertical' || config.orientation === 'vertical-bottom-up';
        const isBottomUp = config.orientation === 'vertical-bottom-up';
        const treeHeight = config.height - 100;
        const treeWidth = config.width - 200;

        // Build a map from solution_id to island index for sorting
        const solutionToIsland = new Map();
        if (islandsData && islandsData.length > 0) {
            islandsData.forEach((islandSolutions, islandIndex) => {
                islandSolutions.forEach(solutionId => {
                    solutionToIsland.set(solutionId, islandIndex);
                });
            });
        }

        // Custom sort function: sort by island order first, then by iteration
        const sortNodes = (a, b) => {
            const aIsland = solutionToIsland.get(a.data.solution_id) ?? (islandsData ? islandsData.length + 1000 : 0);
            const bIsland = solutionToIsland.get(b.data.solution_id) ?? (islandsData ? islandsData.length + 1000 : 0);
            if (aIsland !== bIsland) {
                return aIsland - bIsland;
            }
            // If same island, sort by iteration
            const aIter = a.data.iteration || 0;
            const bIter = b.data.iteration || 0;
            return aIter - bIter;
        };

        const root = d3.hierarchy(treeData)
            .sort(sortNodes);

        // Calculate node spacing: dynamically adjust based on node size
        const getNodeSpacing = (a, b) => {
            const nodeSizeA = getNodeSize(a);
            const nodeSizeB = getNodeSize(b);
            // Minimum spacing is the sum of the radii of two nodes plus extra padding
            let minSpacing = (nodeSizeA + nodeSizeB) * 1.5 + 20;
            // If labels are displayed, add extra spacing to accommodate them (label width: ~80-100px)
            if (config.showLabels) {
                minSpacing += 100;
            }
            return minSpacing;
        };

        const tree = d3.tree()
            .size([treeWidth, treeHeight])
            .separation(getNodeSpacing);

        tree(root);

        // Calculate actual tree dimensions after layout
        let actualTreeHeight = 0;
        let actualTreeWidth = 0;
        root.each(d => {
            if (isVertical) {
                actualTreeHeight = Math.max(actualTreeHeight, d.y);
                actualTreeWidth = Math.max(actualTreeWidth, d.x);
            } else {
                actualTreeHeight = Math.max(actualTreeHeight, d.x);
                actualTreeWidth = Math.max(actualTreeWidth, d.y);
            }
        });

        // Use actual tree dimensions for bottom-up transformation
        const effectiveTreeHeight = Math.max(treeHeight, actualTreeHeight);

        // Helper function to transform coordinates based on orientation
        const transformX = d => {
            if (isVertical) {
                return d.x;
            }
            return d.y;
        };

        const transformY = d => {
            if (isVertical) {
                return isBottomUp ? effectiveTreeHeight - d.y : d.y + 20;
            }
            return d.x;
        };

        // Create a map of solution_id to node for quick lookup
        const nodeMap = new Map();
        root.each(d => {
            nodeMap.set(d.data.solution_id, d);
        });

        g.selectAll('.link')
            .data(root.links())
            .enter()
            .append('path')
            .attr('class', 'link')
            .attr('d', isVertical
                ? d3.linkVertical().x(d => transformX(d)).y(d => transformY(d))
                : d3.linkHorizontal().x(d => transformX(d)).y(d => transformY(d)));

        const nodeGroups = g.selectAll('.node')
            .data(root.descendants())
            .enter()
            .append('g')
            .attr('class', 'node')
            .attr('transform', d => {
                const x = transformX(d);
                const y = transformY(d);
                return `translate(${x},${y})`;
            });

        nodeGroups.append('circle')
            .attr('r', d => getNodeSize(d))
            .attr('fill', d => getNodeColor(d))
            .on('mouseover', function (event, d) {
                d3.select(this).attr('r', getNodeSize(d) * 1.3);

                const badges = [];
                if (d.data.is_best) {
                    badges.push('<span class="inline-block px-2 py-1 rounded text-xs font-bold bg-red-500 text-white ml-1">Best</span>');
                }
                if (d.data.is_elite) {
                    badges.push('<span class="inline-block px-2 py-1 rounded text-xs font-bold bg-yellow-400 text-gray-800 ml-1">Elite</span>');
                }
                if (d.data.is_island_best) {
                    badges.push('<span class="inline-block px-2 py-1 rounded text-xs font-bold bg-green-400 text-gray-800 ml-1">Island Best</span>');
                }

                const tooltipHtml = `
                <div class="font-bold mb-2 text-indigo-400 text-sm">Solution: ${d.data.solution_id}</div>
                <div class="my-1 leading-relaxed">分数: <strong>${d.data.score?.toFixed(4)}</strong> ${badges.join('')}</div>
                <div class="my-1 leading-relaxed">代数: ${d.data.generation}</div>
                <div class="my-1 leading-relaxed">迭代: ${d.data.iteration}</div>
                <div class="my-1 leading-relaxed">岛屿: ${d.data.island_id}</div>
                <div class="my-1 leading-relaxed">子节点数: ${d.children ? d.children.length : 0}</div>
            `;

                // Calculate node coordinates in the SVG
                const nodeX = transformX(d);
                const nodeY = transformY(d);
                const pos = calculateTooltipPosition(nodeX, nodeY);

                tooltip
                    .html(tooltipHtml)
                    .style('left', (pos.x + 15) + 'px')
                    .style('top', (pos.y - 10) + 'px')
                    .classed('opacity-0', false)
                    .classed('opacity-100', true);
            })
            .on('mouseout', function (event, d) {
                d3.select(this).attr('r', getNodeSize(d));
                tooltip.classed('opacity-0', true)
                    .classed('opacity-100', false);
            })
            .on('click', (event, d) => {
                event.stopPropagation();
                showNodeDetails(d.data);
            });

        if (config.showLabels) {
            const textColor = isDarkTheme ? '#e0e0e0' : '#1a1a1a';
            nodeGroups.append('text')
                .attr('dy', '.35em')
                .attr('x', d => {
                    const nodeSize = getNodeSize(d);
                    // Ensure labels are outside nodes with sufficient spacing
                    return d.children ? -nodeSize - 8 : nodeSize + 8;
                })
                .attr('text-anchor', d => (d.children ? 'end' : 'start'))
                .text(d => d.data.name)
                .style('font-size', '11px')
                .style('fill', textColor)
                .style('pointer-events', 'none'); // Prevent labels from blocking mouse events
        }

        // After rendering is complete, add a selection marker if there is a selected node
        if (selectedNodeId) {
            updateSelectedNodeMarker();
        }

        // Calculate actual bounds of all transformed nodes
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;

        root.descendants().forEach(d => {
            const x = transformX(d);
            const y = transformY(d);
            const nodeSize = getNodeSize(d);
            minX = Math.min(minX, x - nodeSize);
            maxX = Math.max(maxX, x + nodeSize);
            minY = Math.min(minY, y - nodeSize);
            maxY = Math.max(maxY, y + nodeSize);
        });

        // Also include links in bounds calculation
        root.links().forEach(link => {
            const sourceX = transformX(link.source);
            const sourceY = transformY(link.source);
            const targetX = transformX(link.target);
            const targetY = transformY(link.target);
            minX = Math.min(minX, sourceX, targetX);
            maxX = Math.max(maxX, sourceX, targetX);
            minY = Math.min(minY, sourceY, targetY);
            maxY = Math.max(maxY, sourceY, targetY);
        });

        // Add padding
        const padding = 40;
        const contentWidth = maxX - minX + padding * 2;
        const contentHeight = maxY - minY + padding * 2;

        // Get the current size of container
        const containerSize = getContainerSize();
        const viewWidth = containerSize.width;
        const viewHeight = containerSize.height;

        // Set SVG size to match container size
        svg.attr('width', viewWidth)
            .attr('height', viewHeight);

        // Calculate initial scale and translation to center the content within the view
        const initialScale = Math.min(
            (viewWidth - 80) / contentWidth,
            (viewHeight - 80) / contentHeight,
            1 // Only zoom out, not zoom in
        );

        // Calculate the center position
        const contentCenterX = (minX + maxX) / 2;
        const contentCenterY = (minY + maxY) / 2;
        const viewCenterX = viewWidth / 2;
        const viewCenterY = viewHeight / 2;


        // Create initial transform: first translate to align content center with view center, then scale
        const initialTransform = d3.zoomIdentity
            .translate(viewCenterX - contentCenterX * initialScale, viewCenterY - contentCenterY * initialScale)
            .scale(initialScale);
        // Apply the transform
        currentTransform = initialTransform;
        svg.call(zoom.transform, initialTransform);
    }

    // Build solution mapping for island view
    // treeData is raw JSON data; node properties are directly on nodes, not under data
    function buildSolutionMap(node) {
        const map = new Map();
        function traverse(n) {
            // In the original tree data, solution_id is directly on the node
            if (n && n.solution_id) {
                map.set(n.solution_id, n);
            }
            // If nodes are wrapped by d3.hierarchy, get the actual node from n.data
            if (n.data && n.data.solution_id) {
                map.set(n.data.solution_id, n.data);
            }
            // Traverse child nodes
            if (n.children) {
                n.children.forEach(traverse);
            }
        }
        traverse(node);
        return map;
    }

    function updateIslandVisualization() {
        g.selectAll('*').remove();

        if (!treeData || !islandsData || islandsData.length === 0) {
            updateSVGSize();
            currentTransform = d3.zoomIdentity;
            svg.call(zoom.transform, currentTransform);
            g.attr('transform', 'translate(40,20)');
            return;
        }
        const solutionMap = buildSolutionMap(treeData);
        // Build parent-child relationship mapping (only for relationships within the same island)
        const parentChildMap = new Map(); // parent_id -> [child_ids]
        const childParentMap = new Map(); // child_id -> parent_id
        const allSolutionIds = new Set();

        solutionMap.forEach((nodeData, solutionId) => {
            allSolutionIds.add(solutionId);
            const parentId = nodeData.parent_id;
            if (parentId && allSolutionIds.has(parentId)) {
                if (!parentChildMap.has(parentId)) {
                    parentChildMap.set(parentId, []);
                }
                parentChildMap.get(parentId).push(solutionId);
                childParentMap.set(solutionId, parentId);
            }
        });

        // Prepare node data for each island and check for parent-child structure
        const islandGroups = [];
        islandsData.forEach((islandSolutions, islandIndex) => {
            if (islandSolutions.length === 0) {
                return;
            }

            // Get all node data for the island
            const nodes = [];
            const nodeIdMap = new Map(); // solution_id -> node object

            islandSolutions.forEach(solutionId => {
                const nodeData = solutionMap.get(solutionId);
                if (nodeData) {
                    const node = {
                        id: solutionId,
                        ...nodeData,
                        x: 10,
                        y: 10,
                    };
                    nodes.push(node);
                    nodeIdMap.set(solutionId, node);
                }
            });

            // Build parent-child relationships within this island
            const islandParentChildMap = new Map();
            const islandRoots = [];

            nodes.forEach(node => {
                const parentId = node.parent_id;
                // Only consider parent-child relationships within the same island
                if (parentId && nodeIdMap.has(parentId)) {
                    if (!islandParentChildMap.has(parentId)) {
                        islandParentChildMap.set(parentId, []);
                    }
                    islandParentChildMap.get(parentId).push(node.id);
                } else {
                    islandRoots.push(node.id);
                }
            });

            // Build tree structure
            const hasTreeStructure = islandParentChildMap.size > 0;
            if (hasTreeStructure) {
            // Set the children property for each node
                nodes.forEach(node => {
                    const children = islandParentChildMap.get(node.id) || [];
                    node.children = children.map(childId => nodeIdMap.get(childId)).filter(Boolean);
                });
            }

            if (nodes.length > 0) {
                islandGroups.push({
                    islandIndex,
                    nodes,
                    solutionIds: islandSolutions,
                    hasTreeStructure,
                    roots: islandRoots,
                });
            }
        });

        if (islandGroups.length === 0) {
            return;
        }

        // Get the size of the current container
        const containerSize = getContainerSize();

        // Calculate the position of each island (grid layout)
        const cols = Math.ceil(Math.sqrt(islandGroups.length));
        const rows = Math.ceil(islandGroups.length / cols);
        const islandWidth = (containerSize.width - 200) / cols;
        const islandHeight = (containerSize.height - 100) / rows;

        // Step 1: Complete layout for all islands, no rendering
        const islandBounds = [];

        islandGroups.forEach((islandData, idx) => {
            const col = idx % cols;
            const row = Math.floor(idx / cols);
            const centerX = col * islandWidth + islandWidth / 2;
            const centerY = row * islandHeight + islandHeight / 2;
            if (islandData.hasTreeStructure && islandData.roots.length > 0) {
            // Use tree layout
            // Build tree structure data
                const treeNodes = [];
                const nodeMap = new Map();

                // Create all nodes
                islandData.nodes.forEach(node => {
                    const treeNode = {
                        data: node,
                        children: [],
                    };
                    treeNodes.push(treeNode);
                    nodeMap.set(node.id, treeNode);
                });

                // Build tree relationships
                const rootNodes = [];
                islandData.nodes.forEach(node => {
                    const treeNode = nodeMap.get(node.id);
                    if (node.children && node.children.length > 0) {
                        treeNode.children = node.children.map(child => nodeMap.get(child.id)).filter(Boolean);
                    }
                    if (islandData.roots.includes(node.id)) {
                        rootNodes.push(treeNode);
                    }
                });

                // Create virtual root node (if there's more than one root)
                let treeRoot;
                if (rootNodes.length === 1) {
                    treeRoot = rootNodes[0];
                } else {
                    treeRoot = {
                        data: {id: 'root', 'solution_id': 'root', isRoot: true},
                        children: rootNodes,
                    };
                }

                // Use d3.hierarchy and d3.tree layout
                const hierarchy = d3.hierarchy(treeRoot, d => d.children);

                // Calculate node spacing: dynamically adjust based on node size
                const getIslandNodeSpacing = (a, b) => {
                    const nodeSizeA = a.data && a.data.data ? getNodeSize({data: a.data.data})
                        : a.data && a.data.id ? getNodeSize({data: a.data}) : config.nodeSize;
                    const nodeSizeB = b.data && b.data.data ? getNodeSize({data: b.data.data})
                        : b.data && b.data.id ? getNodeSize({data: b.data}) : config.nodeSize;
                    // Minimum spacing is sum of two nodes' radii plus extra padding
                    let minSpacing = (nodeSizeA + nodeSizeB) * 1.5 + 20;
                    // If labels are displayed, add extra spacing to accommodate them (label width: ~80-100px)
                    if (config.showLabels) {
                        minSpacing += 100;
                    }
                    return minSpacing;
                };

                const treeLayout = d3.tree()
                    .size([islandWidth * 0.8, islandHeight * 0.8]) // Vertical layout: first parameter is horizontal width, second is vertical height
                    .separation(getIslandNodeSpacing);

                treeLayout(hierarchy);

                // Convert tree layout coordinates to island center coordinate system
                hierarchy.descendants().forEach(d => {
                // d.data may be a tree node object (with data property) or a data object directly
                    let nodeData = null;
                    if (d.data && d.data.data) {
                    // Tree node object: get the actual solution node
                        nodeData = d.data.data;
                    } else if (d.data && d.data.id) {
                    // Directly a data object (case of virtual root node)
                        nodeData = d.data;
                    }

                    // Only process actual solution nodes, skip virtual root node
                    if (nodeData && !nodeData.isRoot && nodeData.id !== 'root') {
                        const node = nodeData;
                        // Vertical layout: d.x is horizontal direction, d.y is vertical direction
                        node.x = centerX + (d.x - islandWidth * 0.4);
                        node.y = centerY + (d.y - islandHeight * 0.4);
                    }
                });

                // Store link info for rendering (save actual node data of source and target)
                islandData.links = hierarchy.links()
                    .map(link => {
                    // Extract the actual solution node from the hierarchy node
                        let sourceData = null;
                        let targetData = null;

                        if (link.source.data && link.source.data.data) {
                            sourceData = link.source.data.data;
                        } else if (link.source.data && link.source.data.id) {
                            sourceData = link.source.data;
                        }

                        if (link.target.data && link.target.data.data) {
                            targetData = link.target.data.data;
                        } else if (link.target.data && link.target.data.id) {
                            targetData = link.target.data;
                        }

                        return {sourceData, targetData};
                    })
                    .filter(({sourceData, targetData}) => {
                        return sourceData && !sourceData.isRoot && sourceData.id !== 'root'
                           && targetData && !targetData.isRoot && targetData.id !== 'root';
                    })
                    .map(({sourceData, targetData}) => ({
                        source: sourceData,
                        target: targetData,
                    }));
            } else {
            // Use force-directed graph layout
                islandData.nodes.forEach(node => {
                    const angle = Math.random() * 2 * Math.PI;
                    const radius = Math.random() * 30;
                    node.x = centerX + Math.cos(angle) * radius;
                    node.y = centerY + Math.sin(angle) * radius;
                });

                // Calculate collision radius, increase radius if labels are displayed
                const getCollisionRadius = d => {
                    const baseRadius = getNodeSize({data: d}) * 2 + 15;
                    // If labels are displayed, add extra radius to accommodate them
                    if (config.showLabels) {
                        return baseRadius + 50;
                    }
                    return baseRadius;
                };

                const simulation = d3.forceSimulation(islandData.nodes)
                    .force('charge', d3.forceManyBody().strength(-150))
                    .force('center', d3.forceCenter(centerX, centerY).strength(0.5))
                    .force('collision', d3.forceCollide().radius(getCollisionRadius))
                    .stop();

                for (let i = 0; i < 300; ++i) {
                    simulation.tick();
                }
                islandData.links = [];
            }

            // Calculate the bounding box of island nodes
            let minX = Infinity;
            let maxX = -Infinity;
            let minY = Infinity;
            let maxY = -Infinity;
            islandData.nodes.forEach(node => {
                const nodeSize = getNodeSize({data: node});
                minX = Math.min(minX, node.x - nodeSize);
                maxX = Math.max(maxX, node.x + nodeSize);
                minY = Math.min(minY, node.y - nodeSize);
                maxY = Math.max(maxY, node.y + nodeSize);
            });

            islandBounds.push({
                islandIndex: islandData.islandIndex,
                minX, maxX, minY, maxY,
                centerX, centerY,
                width: maxX - minX,
                height: maxY - minY,
            });
        });

        // Step 2: Detect and resolve collisions between islands, adjust island positions
        const minIslandDistance = 50; // Minimum distance between islands
        const basePadding = 15;
        const maxPadding = basePadding;

        // Use force-directed graph to separate colliding islands
        const islandPositions = islandBounds.map(bounds => ({
            x: bounds.centerX,
            y: bounds.centerY,
            bounds: bounds,
        }));

        // Detect collisions and adjust positions
        let hasCollision = true;
        let iterations = 0;
        const maxIterations = 100;

        while (hasCollision && iterations < maxIterations) {
            hasCollision = false;
            iterations++;

            islandPositions.forEach(
                (pos1, i) => {
                    islandPositions.forEach((pos2, j) => {
                        if (i >= j) {
                            return;
                        }

                        const bounds1 = pos1.bounds;
                        const bounds2 = pos2.bounds;

                        // Calculate the distance between the two islands' bounding boxes
                        const horizontalOverlap = Math.max(0,
                            Math.min(bounds1.maxX, bounds2.maxX) - Math.max(bounds1.minX, bounds2.minX)
                        );
                        const verticalOverlap = Math.max(0,
                            Math.min(bounds1.maxY, bounds2.maxY) - Math.max(bounds1.minY, bounds2.minY)
                        );

                        // If overlap exists, separation is required
                        if (horizontalOverlap > 0 || verticalOverlap > 0) {
                            hasCollision = true;
                            // Calculate the distance between center points
                            const dx = pos2.x - pos1.x;
                            const dy = pos2.y - pos1.y;
                            const distance = Math.sqrt(dx * dx + dy * dy);

                            // Calculate the required separation distance
                            const neededDistance = Math.max(
                                (bounds1.width + bounds2.width) / 2 + minIslandDistance,
                                (bounds1.height + bounds2.height) / 2 + minIslandDistance
                            );

                            if (distance < neededDistance && distance > 0) {
                                // Calculate the separation direction
                                const separationX = (dx / distance) * (neededDistance - distance) / 2;
                                const separationY = (dy / distance) * (neededDistance - distance) / 2;

                                // Adjust positions
                                pos1.x -= separationX;
                                pos1.y -= separationY;
                                pos2.x += separationX;
                                pos2.y += separationY;
                            } else if (distance === 0) {
                                // If completely overlapped, separate randomly
                                const angle = Math.random() * 2 * Math.PI;
                                const separation = neededDistance / 2;
                                pos1.x -= Math.cos(angle) * separation;
                                pos1.y -= Math.sin(angle) * separation;
                                pos2.x += Math.cos(angle) * separation;
                                pos2.y += Math.sin(angle) * separation;
                            }
                        }
                    });
                });
        }

        // Update island node positions (relative to the new center point)
        islandGroups.forEach((islandData, idx) => {
            const newCenterX = islandPositions[idx].x;
            const newCenterY = islandPositions[idx].y;
            const oldCenterX = islandPositions[idx].bounds.centerX;
            const oldCenterY = islandPositions[idx].bounds.centerY;

            const offsetX = newCenterX - oldCenterX;
            const offsetY = newCenterY - oldCenterY;

            // Move all nodes
            islandData.nodes.forEach(node => {
                node.x += offsetX;
                node.y += offsetY;
            });

            // Update the bounding box
            let minX = Infinity;
            let maxX = -Infinity;
            let minY = Infinity;
            let maxY = -Infinity;
            islandData.nodes.forEach(node => {
                const nodeSize = getNodeSize({data: node});
                minX = Math.min(minX, node.x - nodeSize);
                maxX = Math.max(maxX, node.x + nodeSize);
                minY = Math.min(minY, node.y - nodeSize);
                maxY = Math.max(maxY, node.y + nodeSize);
            });

            islandBounds[idx] = {
                islandIndex: islandData.islandIndex,
                minX, maxX, minY, maxY,
                centerX: newCenterX,
                centerY: newCenterY,
                width: maxX - minX,
                height: maxY - minY,
            };
        });

        // Step 3: Draw all islands
        islandGroups.forEach(islandData => {
            const islandG = g.append('g')
                .attr('class', 'island-group')
                .attr('data-island-index', islandData.islandIndex);

            // Draw tree links (if any)
            if (islandData.hasTreeStructure && islandData.links && islandData.links.length > 0) {
            // Use vertically oriented curved links
                const linkGenerator = d3.linkVertical()
                    .x(d => d.x)
                    .y(d => d.y);

                islandG.selectAll('.link')
                    .data(islandData.links)
                    .enter()
                    .append('path')
                    .attr('class', 'link')
                    .attr('d', d => linkGenerator({
                        source: {x: d.source.x, y: d.source.y},
                        target: {x: d.target.x, y: d.target.y},
                    }))
                    .attr('stroke', '#999')
                    .attr('stroke-width', 2)
                    .attr('stroke-opacity', 0.6)
                    .attr('fill', 'none')
                    .lower();
            }

            // Calculate the bounds of all nodes for drawing circular enclosures
            let minX = Infinity;
            let maxX = -Infinity;
            let minY = Infinity;
            let maxY = -Infinity;
            islandData.nodes.forEach(node => {
                const nodeSize = getNodeSize({data: node});
                minX = Math.min(minX, node.x - nodeSize);
                maxX = Math.max(maxX, node.x + nodeSize);
                minY = Math.min(minY, node.y - nodeSize);
                maxY = Math.max(maxY, node.y + nodeSize);
            });

            // Calculate center and radius
            const circleCenterX = (minX + maxX) / 2;
            const circleCenterY = (minY + maxY) / 2;
            const width = maxX - minX;
            const height = maxY - minY;
            // Radius = half of the diagonal + padding
            const radius = Math.sqrt(width * width + height * height) / 2 + maxPadding;

            // Draw the circular enclosure
            islandG.append('circle')
                .attr('cx', circleCenterX)
                .attr('cy', circleCenterY)
                .attr('r', radius)
                .attr('fill', islandColors[islandData.islandIndex % islandColors.length])
                .attr('fill-opacity', 0.1)
                .attr('stroke', islandColors[islandData.islandIndex % islandColors.length])
                .attr('stroke-width', 2)
                .attr('stroke-opacity', 0.6)
                .lower(); // Place at the bottom layer

            // Draw nodes
            const nodeGroups = islandG.selectAll(`.node-island-${islandData.islandIndex}`)
                .data(islandData.nodes)
                .enter()
                .append('g')
                .attr('class', 'node')
                .attr('transform', d => `translate(${d.x},${d.y})`);

            nodeGroups.append('circle')
                .attr('r', d => getNodeSize({data: d}))
                .attr('fill', islandColors[islandData.islandIndex % islandColors.length])
            // .attr('stroke', d => {
            //     if (d.is_best) return '#ff4444';
            //     if (d.is_elite) return '#ffd700';
            //     if (d.is_island_best) return '#44ff44';
            //     return '#fff';
            // })
            // .attr('stroke-width', d => {
            //     if (d.is_best) return 4;
            //     if (d.is_elite) return 3;
            //     if (d.is_island_best) return 3;
            //     return 2;
            // })
                .on('mouseover', function (event, d) {
                    d3.select(this).attr('r', getNodeSize({data: d}) * 1.3);

                    const badges = [];
                    if (d.is_best) {
                        badges.push('<span class="inline-block px-2 py-1 rounded text-xs font-bold bg-red-500 text-white ml-1">Best</span>');
                    }
                    if (d.is_elite) {
                        badges.push('<span class="inline-block px-2 py-1 rounded text-xs font-bold bg-yellow-400 text-gray-800 ml-1">Elite</span>');
                    }
                    if (d.is_island_best) {
                        badges.push('<span class="inline-block px-2 py-1 rounded text-xs font-bold bg-green-400 text-gray-800 ml-1">Island Best</span>');
                    }

                    const tooltipHtml = `
                    <div class="font-bold mb-2 text-indigo-400 text-sm">Solution: ${d.solution_id}</div>
                    <div class="my-1 leading-relaxed">分数: <strong>${d.score.toFixed(4)}</strong> ${badges.join('')}</div>
                    <div class="my-1 leading-relaxed">代数: ${d.generation}</div>
                    <div class="my-1 leading-relaxed">迭代: ${d.iteration}</div>
                    <div class="my-1 leading-relaxed">岛屿: ${d.island_id}</div>
                `;

                    // Calculate the coordinates of nodes in the SVG
                    const nodeX = d.x;
                    const nodeY = d.y;
                    const pos = calculateTooltipPosition(nodeX, nodeY);

                    tooltip
                        .html(tooltipHtml)
                        .style('left', (pos.x + 15) + 'px')
                        .style('top', (pos.y - 10) + 'px')
                        .classed('opacity-0', false)
                        .classed('opacity-100', true);
                })
                .on('mouseout', function (event, d) {
                    d3.select(this).attr('r', getNodeSize({data: d}));
                    tooltip.classed('opacity-0', true)
                        .classed('opacity-100', false);
                })
                .on('click', (event, d) => {
                    event.stopPropagation();
                    showNodeDetails(d);
                });

            if (config.showLabels) {
                const textColor = isDarkTheme ? '#e0e0e0' : '#1a1a1a';
                nodeGroups.append('text')
                    .attr('dy', '.35em')
                    .attr('x', d => {
                        const nodeSize = getNodeSize({data: d});
                        //  Ensure labels are outside nodes with sufficient spacing
                        return nodeSize + 8;
                    })
                    .attr('text-anchor', 'start')
                    .text(d => d.name || d.solution_id.substring(0, 8))
                    .style('font-size', '10px')
                    .style('fill', textColor)
                    .style('pointer-events', 'none'); // Prevent labels from blocking mouse events
            }
        });

        // After rendering is complete, add selection markers if there are selected nodes
        if (selectedNodeId) {
            updateSelectedNodeMarker();
        }

        // Calculate the bounds of the actual content
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;

        islandGroups.forEach(islandData => {
            islandData.nodes.forEach(node => {
                const nodeSize = getNodeSize({data: node});
                minX = Math.min(minX, node.x - nodeSize);
                maxX = Math.max(maxX, node.x + nodeSize);
                minY = Math.min(minY, node.y - nodeSize);
                maxY = Math.max(maxY, node.y + nodeSize);
            });
        });

        // Adjust SVG size and initial zoom
        const padding = 40;
        const contentWidth = maxX - minX + padding * 2;
        const contentHeight = maxY - minY + padding * 2;

        // Use the previously obtained container size
        const viewWidth = containerSize.width;
        const viewHeight = containerSize.height;

        // Set SVG size to match container size
        svg.attr('width', viewWidth)
            .attr('height', viewHeight);

        const initialScale = Math.min(
            (viewWidth - 80) / contentWidth,
            (viewHeight - 80) / contentHeight,
            1
        );

        const contentCenterX = (minX + maxX) / 2;
        const contentCenterY = (minY + maxY) / 2;
        const viewCenterX = viewWidth / 2;
        const viewCenterY = viewHeight / 2;

        const initialTransform = d3.zoomIdentity
            .translate(viewCenterX - contentCenterX * initialScale, viewCenterY - contentCenterY * initialScale)
            .scale(initialScale);

        currentTransform = initialTransform;
        svg.call(zoom.transform, initialTransform);
    }

    async function fetchJSON(url) {
        const response = await fetch(url);
        if (!response.ok) {
            const text = await response.text();
            throw new Error(text || response.statusText);
        }
        return response.json();
    }

    async function loadCheckpoints() {
        try {
            setLoading(false);
            const data = await fetchJSON('/api/checkpoints');
            checkpointSelect.innerHTML = '';

            if (!data.checkpoints || data.checkpoints.length === 0) {
                setLoading(true, '暂无 checkpoint 数据');
                setTreeContainerVisible(false);
                updateStats();
                treeData = null;
                updateVisualization();
                checkpointsList = [];
                updateTimelineUI();
                return;
            }

            // Sort checkpoints by numeric order, sort on the backend
            const sortedCheckpoints = data.checkpoints;

            checkpointsList = sortedCheckpoints;

            sortedCheckpoints.forEach(name => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                checkpointSelect.appendChild(option);
            });

            const defaultId = data.default || sortedCheckpoints[0];
            checkpointSelect.value = defaultId;
            currentCheckpointIndex = sortedCheckpoints.indexOf(defaultId);
            updateTimelineUI();
            await loadCheckpointData(defaultId);
        } catch (error) {
            setLoading(true, '加载 checkpoint 失败');
            setTreeContainerVisible(false);
            checkpointsList = [];
            updateTimelineUI();
        }
    }

    function updateTimelineUI() {
        const total = checkpointsList.length;
        totalCheckpointsSpan.textContent = total;

        if (checkpointSlider) {
            checkpointSlider.max = Math.max(0, total - 1);
            checkpointSlider.value = currentCheckpointIndex;
        }

        if (currentCheckpointIndex >= 0 && currentCheckpointIndex < total) {
            currentCheckpointSpan.textContent = checkpointsList[currentCheckpointIndex];
        } else {
            currentCheckpointSpan.textContent = '--';
        }

        // Update checkpoint labels on slider
        updateSliderLabels();
    }

    function updateSliderLabels() {
        const labelsContainer = document.getElementById('checkpointLabels');
        if (!labelsContainer || checkpointsList.length === 0) {
            return;
        }

        labelsContainer.innerHTML = '';
        const total = checkpointsList.length;

        // Show first, middle, and last labels
        if (total > 0) {
            const firstLabel = document.createElement('span');
            firstLabel.textContent = checkpointsList[0];
            labelsContainer.appendChild(firstLabel);

            if (total > 2) {
                const middleLabel = document.createElement('span');
                middleLabel.textContent = checkpointsList[Math.floor(total / 2)];
                labelsContainer.appendChild(middleLabel);
            }

            if (total > 1) {
                const lastLabel = document.createElement('span');
                lastLabel.textContent = checkpointsList[total - 1];
                labelsContainer.appendChild(lastLabel);
            }
        }
    }

    let isLoadingCheckpoint = false;
    async function loadCheckpointData(checkpointId) {
        if (!checkpointId) {
            setLoading(true, '请选择 checkpoint');
            setTreeContainerVisible(false);
            return;
        }

        // Prevent concurrent loads
        if (isLoadingCheckpoint) {
            return;
        }

        try {
            isLoadingCheckpoint = true;
            setLoading(true, '');
            currentCheckpointId = checkpointId; // save current checkpoint ID
            const data = await fetchJSON(`/api/checkpoints/${checkpointId}/tree`);
            treeData = data.tree;
            islandsData = data.islands || null;
            scoreHistory = data.score_history || [];
            updateStats(data.stats);
            updateScoreChart();
            setLoading(false);
            setTreeContainerVisible(true);
            updateIslandLegend();
            updateVisualization();
        } catch (error) {
            setLoading(true, `加载 ${checkpointId} 失败`);
            setTreeContainerVisible(false);
        } finally {
            isLoadingCheckpoint = false;
        }
    }

    document.getElementById('orientation').addEventListener('change', e => {
        config.orientation = e.target.value;
        updateVisualization();
    });

    document.getElementById('nodeSize').addEventListener('input', e => {
        config.nodeSize = parseInt(e.target.value, 10);
        document.getElementById('nodeSizeValue').textContent = config.nodeSize;
        updateVisualization();
    });

    document.getElementById('showLabels').addEventListener('change', e => {
        config.showLabels = e.target.checked;
        updateVisualization();
    });

    checkpointSelect.addEventListener('change', e => {
        const checkpointId = e.target.value;
        const index = checkpointsList.indexOf(checkpointId);
        if (index >= 0) {
            currentCheckpointIndex = index;
            updateTimelineUI();
        }
        loadCheckpointData(checkpointId);
    });

    // Timeline slider event handlers
    if (checkpointSlider) {
        checkpointSlider.addEventListener('input', e => {
            const index = parseInt(e.target.value, 10);
            if (index >= 0 && index < checkpointsList.length) {
            // Stop playback when user manually drags
                if (isPlaying) {
                    isPlaying = false;
                    playPauseIcon.textContent = '▶';
                    stopPlayback();
                }

                currentCheckpointIndex = index;
                const checkpointId = checkpointsList[index];
                checkpointSelect.value = checkpointId;
                currentCheckpointSpan.textContent = checkpointId;
                loadCheckpointData(checkpointId);
            }
        });
    }

    // Play/Pause functionality
    function togglePlayback() {
        if (checkpointsList.length === 0) {
            return;}

        isPlaying = !isPlaying;

        if (isPlaying) {
            playPauseIcon.textContent = '⏸';
            startPlayback();
        } else {
            playPauseIcon.textContent = '▶';
            stopPlayback();
        }
    }

    function startPlayback() {
        if (playbackInterval) {
            clearInterval(playbackInterval);
        }

        playbackInterval = setInterval(() => {
        // Don't advance if currently loading
            if (isLoadingCheckpoint) {
                return;
            }

            if (currentCheckpointIndex < checkpointsList.length - 1) {
                currentCheckpointIndex++;
                const checkpointId = checkpointsList[currentCheckpointIndex];
                checkpointSelect.value = checkpointId;
                checkpointSlider.value = currentCheckpointIndex;
                currentCheckpointSpan.textContent = checkpointId;
                loadCheckpointData(checkpointId).catch(() => {
                // Stop playback on error
                    isPlaying = false;
                    playPauseIcon.textContent = '▶';
                    stopPlayback();
                });
            } else {
            // Reached the end, check if still playing
            // eslint-disable-next-line no-lonely-if
                if (isPlaying) {
                // Loop back to the first checkpoint
                    currentCheckpointIndex = 0;
                    const checkpointId = checkpointsList[currentCheckpointIndex];
                    checkpointSelect.value = checkpointId;
                    checkpointSlider.value = currentCheckpointIndex;
                    currentCheckpointSpan.textContent = checkpointId;
                    loadCheckpointData(checkpointId).catch(() => {
                    // Stop playback on error
                        isPlaying = false;
                        playPauseIcon.textContent = '▶';
                        stopPlayback();
                    });
                } else {
                // User manually stopped, stop playback
                    stopPlayback();
                }
            }
        }, playbackSpeed);
    }

    function stopPlayback() {
        if (playbackInterval) {
            clearInterval(playbackInterval);
            playbackInterval = null;
        }
    }

    if (playPauseBtn) {
        playPauseBtn.addEventListener('click', togglePlayback);
    }

    // Function to navigate to a specific checkpoint index
    function navigateToCheckpoint(index) {
        if (index < 0 || index >= checkpointsList.length) {
            return; // Out of bounds
        }

        // Stop playback when manually navigating
        if (isPlaying) {
            isPlaying = false;
            playPauseIcon.textContent = '▶';
            stopPlayback();
        }

        currentCheckpointIndex = index;
        const checkpointId = checkpointsList[index];
        checkpointSelect.value = checkpointId;
        if (checkpointSlider) {
            checkpointSlider.value = index;
        }
        currentCheckpointSpan.textContent = checkpointId;
        loadCheckpointData(checkpointId);
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', event => {
    // Space key: toggle playback
        if (event.code === 'Space' || event.key === ' ') {
            event.preventDefault(); // Prevent page scroll
            if (checkpointsList.length > 0) {
                togglePlayback();
            }
        }
        // Left arrow: previous checkpoint
        else if (event.code === 'ArrowLeft' || event.key === 'ArrowLeft') {
            event.preventDefault(); // Prevent page scroll
            if (checkpointsList.length > 0 && currentCheckpointIndex > 0) {
                navigateToCheckpoint(currentCheckpointIndex - 1);
            }
        }
        // Right arrow: next checkpoint
        else if (event.code === 'ArrowRight' || event.key === 'ArrowRight') {
            event.preventDefault(); // Prevent page scroll
            if (checkpointsList.length > 0 && currentCheckpointIndex < checkpointsList.length - 1) {
                navigateToCheckpoint(currentCheckpointIndex + 1);
            }
        }
    });

    // Playback speed control
    if (playbackSpeedSelect) {
        playbackSpeedSelect.addEventListener('change', e => {
            playbackSpeed = parseInt(e.target.value, 10);
            if (isPlaying) {
                stopPlayback();
                startPlayback();
            }
        });
    }

    if (scoreChartExpandBtn) {
        scoreChartExpandBtn.addEventListener('click', showExpandedScoreChart);
    }
    if (scoreChartCollapseBtn) {
        scoreChartCollapseBtn.addEventListener('click', hideExpandedScoreChart);
    }

    window.addEventListener('resize', () => updateScoreChart());

    // Zoom control function
    function zoomIn() {
        svg.transition().call(zoom.scaleBy, 1.5);
    }

    function zoomOut() {
        svg.transition().call(zoom.scaleBy, 1 / 1.5);
    }

    function resetZoom() {
    // Reset to the initial state, recalculate the initial layout
        if (treeData) {
            updateVisualization();
        } else {
            svg.transition().call(zoom.transform, d3.zoomIdentity);
        }
    }

    // Bind events to zoom control buttons
    document.addEventListener('DOMContentLoaded', () => {
        const zoomInBtn = document.getElementById('zoomIn');
        const zoomOutBtn = document.getElementById('zoomOut');
        const resetZoomBtn = document.getElementById('resetZoom');

        if (zoomInBtn) {
            zoomInBtn.addEventListener('click', zoomIn);
        }
        if (zoomOutBtn) {
            zoomOutBtn.addEventListener('click', zoomOut);
        }
        if (resetZoomBtn) {
            resetZoomBtn.addEventListener('click', resetZoom);
        }
    });

    // View mode switch function
    function switchViewMode(mode) {
        config.viewMode = mode;

        // Update button states
        document.querySelectorAll('.view-mode-btn').forEach(btn => {
            if (btn.dataset.view === mode) {
                btn.classList.add('active');
                btn.classList.remove('bg-gray-300', 'text-gray-700');
                btn.classList.add('bg-indigo-500', 'text-white');
            } else {
                btn.classList.remove('active');
                btn.classList.remove('bg-indigo-500', 'text-white');
            }
        });

        // Show/hide layout direction controls based on view mode
        const orientationContainer = document.getElementById('orientationContainer');
        if (orientationContainer) {
            if (mode === 'island') {
                orientationContainer.style.display = 'none';
            } else {
                orientationContainer.style.display = 'flex';
            }
        }

        // Update visualization
        updateVisualization();
    }

    // Bind events to view mode buttons
    document.addEventListener('DOMContentLoaded', () => {
        const treeViewBtn = document.getElementById('treeViewBtn');
        const islandViewBtn = document.getElementById('islandViewBtn');

        if (treeViewBtn) {
            treeViewBtn.addEventListener('click', () => switchViewMode('tree'));
        }
        if (islandViewBtn) {
            islandViewBtn.addEventListener('click', () => switchViewMode('island'));
        }

        // Set the display state of layout direction controls based on the current view mode during initialization
        const orientationContainer = document.getElementById('orientationContainer');
        if (orientationContainer && config.viewMode === 'island') {
            orientationContainer.style.display = 'none';
        }
    });

    // ===== Legend Drag Functionality =====
    function initLegendDrag() {
        const legends = document.querySelectorAll('.legend');
        const visualizationArea = document.querySelector('.visualization-area');

        if (!visualizationArea) {
            return;
        }

        legends.forEach(legend => {
            let isDragging = false;
            let currentX = 0;
            let currentY = 0;
            let initialX = 0;
            let initialY = 0;
            let xOffset = 0;
            let yOffset = 0;

            // Restore position from localStorage
            const legendId =
            legend.classList.contains('zoom-hint') ? 'zoom-hint'
                : legend.classList.contains('island-legend') ? 'island-legend'
                    : legend.classList.contains('solution-legend') ? 'solution-legend' : null;

            if (legendId) {
                const savedPos = localStorage.getItem(`legend-${legendId}-position`);
                if (savedPos) {
                    try {
                        const pos = JSON.parse(savedPos);
                        // Restore position, prioritizing the use of left and top
                        if (pos.left) {
                            legend.style.left = pos.left;
                            legend.style.right = 'auto';
                        }
                        if (pos.top) {
                            legend.style.top = pos.top;
                            legend.style.bottom = 'auto';
                        }
                        if (pos.right && !pos.left) {
                            legend.style.right = pos.right;
                            legend.style.left = 'auto';
                        }
                        if (pos.bottom && !pos.top) {
                            legend.style.bottom = pos.bottom;
                            legend.style.top = 'auto';
                        }
                    } catch (e) {
                        console.warn('Failed to restore legend position:', e);
                    }
                }
            }

            // Mouse down event
            legend.addEventListener('mousedown', dragStart);

            function dragStart(e) {
                if (e.button !== 0) {
                    return;
                } // Only handle left mouse button

                initialX = e.clientX;
                initialY = e.clientY;

                // Get the current calculated position
                const rect = legend.getBoundingClientRect();
                const areaRect = visualizationArea.getBoundingClientRect();

                // Calculate the offset relative to the visualization-area
                xOffset = rect.left - areaRect.left;
                yOffset = rect.top - areaRect.top;

                isDragging = true;
                legend.classList.add('dragging');

                // Clear right and bottom to use left and top
                legend.style.right = 'auto';
                legend.style.bottom = 'auto';

                document.addEventListener('mousemove', drag);
                document.addEventListener('mouseup', dragEnd);

                e.preventDefault();
            }

            function drag(e) {
                if (!isDragging) {
                    return;
                }

                e.preventDefault();

                currentX = e.clientX - initialX;
                currentY = e.clientY - initialY;

                const areaRect = visualizationArea.getBoundingClientRect();
                const newX = xOffset + currentX;
                const newY = yOffset + currentY;

                // Restrict within the visible area
                const maxX = areaRect.width - legend.offsetWidth;
                const maxY = areaRect.height - legend.offsetHeight;

                const constrainedX = Math.max(0, Math.min(newX, maxX));
                const constrainedY = Math.max(0, Math.min(newY, maxY));

                legend.style.left = constrainedX + 'px';
                legend.style.top = constrainedY + 'px';
            }

            function dragEnd() {
                if (!isDragging) {
                    return;}

                isDragging = false;
                legend.classList.remove('dragging');

                // Save position to localStorage
                if (legendId) {

                    const pos = {
                        left: legend.style.left,
                        top: legend.style.top,
                    };
                    localStorage.setItem(`legend-${legendId}-position`, JSON.stringify(pos));
                }

                document.removeEventListener('mousemove', drag);
                document.removeEventListener('mouseup', dragEnd);
            }
        });
    }

    // Initialize: hide loading state
    setLoading(false);

    // Initialize legend drag functionality
    function initLegendDragOnReady() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initLegendDrag);
        } else {
        // DOM already loaded
            initLegendDrag();
        }
    }

    initLegendDragOnReady();
    loadCheckpoints();

    // Recalculate container size and update visualization when window size changes
    let resizeTimeout;
    window.addEventListener('resize', () => {
    // Use debouncing to avoid frequent triggering
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            if (treeData) {
                updateVisualization();
            }
        }, 250);
    });

    // Initialize drag and drop for the right panel
    function initResizeHandle() {
        const resizer = document.getElementById('resizeHandle');
        const rightPanel = document.querySelector('.app-right');

        if (!resizer || !rightPanel) {
            return;}

        resizer.addEventListener('mousedown', e => {
            e.preventDefault();
            document.addEventListener('mousemove', resize);
            document.addEventListener('mouseup', stopResize);
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';

            resizer.classList.add('resizing');
        });

        function resize(e) {
        // Calculate the new width of the right panel
        // Get the right boundary position of app-right
            const rightPanelRect = rightPanel.getBoundingClientRect();
            const rightEdge = rightPanelRect.right;

            // New width = right boundary - current mouse X coordinate
            let newWidth = rightEdge - e.clientX;

            // Restrict width range
            const maxWidth = window.innerWidth * 0.5;
            const minWidth = 300;

            if (newWidth > maxWidth) {
                newWidth = maxWidth;
            }
            if (newWidth < minWidth) {
                newWidth = minWidth;
            }

            rightPanel.style.width = newWidth + 'px';

            // Trigger layout update for code editor and diff editor
            if (codeEditorInstance) {
                codeEditorInstance.layout();
            }
            if (diffEditorInstance) {
                diffEditorInstance.layout();
            }
        }

        function stopResize() {
            document.removeEventListener('mousemove', resize);
            document.removeEventListener('mouseup', stopResize);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            resizer.classList.remove('resizing');
        }
    }

    // Initialize after the DOM is fully loaded
    document.addEventListener('DOMContentLoaded', initResizeHandle);

}

initVisualizer();

