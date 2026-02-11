// General Agent Visualizer - Main Application Logic

// ==================== State Management ====================
const state = {
    tasks: [],
    currentTask: null,
    currentIteration: null,
    scoreChart: null,
};

// ==================== API Functions ====================
async function fetchTasks() {
    try {
        const response = await fetch('/api/tasks');
        const data = await response.json();
        if (data.success) {
            state.tasks = data.tasks;
            renderTaskList();
        } else {
            showError('Failed to load tasks: ' + data.error);
        }
    } catch (error) {
        showError('Error fetching tasks: ' + error.message);
    }
}

async function fetchTaskDetails(taskId) {
    try {
        const response = await fetch(`/api/tasks/${taskId}`);
        const data = await response.json();
        if (data.success) {
            state.currentTask = data.task;
            renderTaskOverview();
        } else {
            showError('Failed to load task: ' + data.error);
        }
    } catch (error) {
        showError('Error fetching task details: ' + error.message);
    }
}

async function fetchIterationDetails(taskId, iteration) {
    try {
        const response = await fetch(`/api/tasks/${taskId}/iterations/${iteration}`);
        const data = await response.json();
        if (data.success) {
            state.currentIteration = data.iteration;
            renderIterationDetails();
        } else {
            showError('Failed to load iteration: ' + data.error);
        }
    } catch (error) {
        showError('Error fetching iteration details: ' + error.message);
    }
}

async function fetchFileContent(taskId, iteration, filepath) {
    try {
        const response = await fetch(`/api/tasks/${taskId}/iterations/${iteration}/files/${encodeURIComponent(filepath)}`);
        const data = await response.json();
        if (data.success) {
            showCodeViewer(data.file);
        } else {
            showError('Failed to load file: ' + data.error);
        }
    } catch (error) {
        showError('Error fetching file: ' + error.message);
    }
}

// ==================== Rendering Functions ====================
function renderTaskList() {
    const taskList = document.getElementById('taskList');

    if (state.tasks.length === 0) {
        taskList.innerHTML = '<div class="loading">No tasks found</div>';
        return;
    }

    taskList.innerHTML = state.tasks.map(task => `
        <div class="task-card ${state.currentTask && state.currentTask.task_id === task.task_id ? 'active' : ''}"
             onclick="selectTask('${task.task_id}')">
            <div class="task-name">${task.task_name}</div>
            <div class="task-info">
                <span>${task.num_iterations} iterations</span>
                <span>Score: ${task.latest_score !== null ? task.latest_score.toFixed(2) : 'N/A'}</span>
            </div>
        </div>
    `).join('');
}

function renderTaskOverview() {
    if (!state.currentTask) return;

    // Hide other sections
    document.getElementById('welcomeScreen').style.display = 'none';
    document.getElementById('iterationDetails').style.display = 'none';
    document.getElementById('taskOverview').style.display = 'block';

    // Update header
    document.getElementById('taskTitle').textContent = state.currentTask.task_name;
    document.getElementById('taskIterations').textContent = `${state.currentTask.current_iteration} iterations`;

    const statusBadge = document.getElementById('taskStatus');
    statusBadge.textContent = 'Completed';
    statusBadge.className = 'badge status-completed';

    // Render score chart
    renderScoreChart();

    // Render iterations
    renderIterations();
}

function renderScoreChart() {
    const ctx = document.getElementById('scoreChart');
    const history = state.currentTask.score_history || [];

    // Destroy previous chart if exists
    if (state.scoreChart) {
        state.scoreChart.destroy();
    }

    // Calculate max score for Y-axis
    const scores = history.map(h => h.score);
    const maxScore = scores.length > 0 ? Math.max(...scores) : 1.0;
    // Add 10% padding to the max score for better visualization
    const yAxisMax = Math.ceil(maxScore * 1.1 * 10) / 10;

    state.scoreChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: history.map(h => `Iter ${h.iteration}`),
            datasets: [{
                label: 'Score',
                data: history.map(h => h.score),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.3,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: (context) => `Score: ${context.parsed.y.toFixed(3)}`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: yAxisMax,
                    ticks: {
                        callback: (value) => value.toFixed(1)
                    }
                }
            }
        }
    });
}

function renderIterations() {
    const iterationsList = document.getElementById('iterationsList');
    const iterations = state.currentTask.iterations || [];

    if (iterations.length === 0) {
        iterationsList.innerHTML = '<div class="empty">No iterations yet</div>';
        return;
    }

    iterationsList.innerHTML = iterations.map(iter => `
        <div class="iteration-card" onclick="selectIteration(${iter.iteration})">
            <div class="iteration-number">Iteration ${iter.iteration}</div>
            <div class="iteration-score">
                Score: <span class="score-value">${iter.score !== null ? iter.score.toFixed(3) : 'N/A'}</span>
            </div>
            <div class="iteration-files">${iter.num_files} files</div>
        </div>
    `).join('');
}

function renderIterationDetails() {
    if (!state.currentIteration) return;

    // Show iteration details section
    document.getElementById('taskOverview').style.display = 'none';
    document.getElementById('iterationDetails').style.display = 'block';

    // Update header
    document.getElementById('iterationTitle').textContent = `Iteration ${state.currentIteration.iteration} Details`;

    // Render plan with Markdown
    const planContent = document.getElementById('planContent');
    if (state.currentIteration.plan) {
        planContent.innerHTML = marked.parse(state.currentIteration.plan);
        planContent.classList.add('markdown-content');
    } else {
        planContent.innerHTML = '<p class="empty">No plan available</p>';
        planContent.classList.remove('markdown-content');
    }

    // Render files as a tree structure
    const filesList = document.getElementById('filesList');
    const filesCount = document.getElementById('filesCount');
    const files = state.currentIteration.files || [];

    filesCount.textContent = files.length;

    if (files.length === 0) {
        filesList.innerHTML = '<p class="empty">No files generated</p>';
    } else {
        filesList.innerHTML = renderFileTree(files);
    }

    // Render evaluation with Markdown for logs
    const evaluationContent = document.getElementById('evaluationContent');
    const evaluation = state.currentIteration.evaluation;

    if (evaluation && evaluation.score !== null) {
        let html = `<div style="font-size: 1.5rem; font-weight: bold; color: #3b82f6; margin-bottom: 1rem;">
            Score: ${evaluation.score.toFixed(3)}
        </div>`;

        if (evaluation.logs) {
            html += `<div class="markdown-content">${marked.parse(evaluation.logs)}</div>`;
        }

        evaluationContent.innerHTML = html;
    } else {
        evaluationContent.innerHTML = '<p class="empty">No evaluation data</p>';
    }

    // Render summary with Markdown
    const summaryContent = document.getElementById('summaryContent');
    if (state.currentIteration.summary) {
        summaryContent.innerHTML = marked.parse(state.currentIteration.summary);
        summaryContent.classList.add('markdown-content');
    } else {
        summaryContent.innerHTML = '<p class="empty">No summary available</p>';
        summaryContent.classList.remove('markdown-content');
    }
}

function showCodeViewer(fileData) {
    const modal = document.getElementById('codeViewer');
    const title = document.getElementById('codeViewerTitle');
    const content = document.getElementById('codeViewerContent').querySelector('code');

    title.textContent = fileData.filename;
    content.textContent = fileData.content;

    modal.style.display = 'flex';
}

function hideCodeViewer() {
    document.getElementById('codeViewer').style.display = 'none';
}

// ==================== Event Handlers ====================
function selectTask(taskId) {
    fetchTaskDetails(taskId);
    renderTaskList(); // Re-render to update active state
}

function selectIteration(iteration) {
    if (!state.currentTask) return;
    fetchIterationDetails(state.currentTask.task_id, iteration);
}

function viewFile(filepath) {
    if (!state.currentTask || !state.currentIteration) return;
    fetchFileContent(
        state.currentTask.task_id,
        state.currentIteration.iteration,
        filepath
    );
}

function goBack() {
    state.currentIteration = null;
    document.getElementById('iterationDetails').style.display = 'none';
    document.getElementById('taskOverview').style.display = 'block';
}

function refresh() {
    fetchTasks();
    if (state.currentTask) {
        fetchTaskDetails(state.currentTask.task_id);
    }
}

// ==================== Utility Functions ====================
function buildFileTree(files) {
    const tree = {};

    files.forEach(file => {
        const parts = file.path.split('/');
        let current = tree;

        parts.forEach((part, index) => {
            if (index === parts.length - 1) {
                // This is a file
                if (!current._files) current._files = [];
                current._files.push({
                    name: part,
                    path: file.path,
                    size: file.size,
                    extension: file.extension
                });
            } else {
                // This is a directory
                if (!current[part]) {
                    current[part] = {};
                }
                current = current[part];
            }
        });
    });

    return tree;
}

function renderFileTree(files) {
    const tree = buildFileTree(files);
    return renderTreeNode(tree, '', true);
}

function renderTreeNode(node, path, isRoot = false) {
    let html = '';

    // Render directories
    const dirs = Object.keys(node).filter(k => k !== '_files').sort();
    dirs.forEach(dir => {
        const dirPath = path ? `${path}/${dir}` : dir;
        const hasFiles = node[dir]._files || Object.keys(node[dir]).filter(k => k !== '_files').length > 0;

        html += `
            <div class="file-tree-item folder">
                <div class="folder-header" onclick="toggleFolder(event)">
                    <span class="folder-toggle">📁</span>
                    <span class="folder-name">${dir}</span>
                </div>
                <div class="folder-contents" style="display: none;">
                    ${renderTreeNode(node[dir], dirPath)}
                </div>
            </div>
        `;
    });

    // Render files
    if (node._files) {
        node._files.sort((a, b) => a.name.localeCompare(b.name)).forEach(file => {
            const icon = getFileIcon(file.extension);
            html += `
                <div class="file-tree-item file" onclick="viewFile('${file.path}')">
                    <span class="file-icon">${icon}</span>
                    <span class="file-name">${file.name}</span>
                    <span class="file-size">${formatFileSize(file.size)}</span>
                </div>
            `;
        });
    }

    return html;
}

function getFileIcon(extension) {
    const icons = {
        '.py': '🐍',
        '.js': '📜',
        '.json': '📋',
        '.md': '📝',
        '.txt': '📄',
        '.yaml': '⚙️',
        '.yml': '⚙️',
        '.sh': '🔧',
        '.html': '🌐',
        '.css': '🎨'
    };
    return icons[extension] || '📄';
}

function toggleFolder(event) {
    event.stopPropagation();
    const header = event.currentTarget;
    const folderItem = header.parentElement;
    const toggle = header.querySelector('.folder-toggle');
    const contents = folderItem.querySelector('.folder-contents');

    if (contents.style.display === 'none') {
        contents.style.display = 'block';
        toggle.textContent = '📂';
    } else {
        contents.style.display = 'none';
        toggle.textContent = '📁';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function showError(message) {
    console.error(message);
    // You could add a toast notification here
    alert(message);
}

// ==================== Initialization ====================
document.addEventListener('DOMContentLoaded', () => {
    // Set up event listeners
    document.getElementById('refreshBtn').addEventListener('click', refresh);
    document.getElementById('backBtn').addEventListener('click', goBack);
    document.getElementById('closeCodeViewer').addEventListener('click', hideCodeViewer);

    // Close modal when clicking outside
    document.getElementById('codeViewer').addEventListener('click', (e) => {
        if (e.target.id === 'codeViewer') {
            hideCodeViewer();
        }
    });

    // Initial load
    fetchTasks();

    // Auto-refresh every 10 seconds
    setInterval(() => {
        if (state.currentTask && !state.currentIteration) {
            fetchTaskDetails(state.currentTask.task_id);
        }
    }, 10000);
});
