// ==================== DETECTIONS HISTORY JAVASCRIPT ====================

// DOM Elements
const refreshBtn = document.getElementById('refreshBtn');
const detectionsTableBody = document.getElementById('detectionsTableBody');
const detectionCount = document.getElementById('detectionCount');

// Event Listeners
refreshBtn.addEventListener('click', loadDetections);

/**
 * Load detection history
 */
async function loadDetections() {
    try {
        const data = await apiRequest('/api/detections?limit=100');

        if (data.success) {
            displayDetections(data.detections);
        }
    } catch (error) {
        showToast(`Failed to load detections: ${error.message}`, 'error');
    }
}

/**
 * Display detections in table
 */
function displayDetections(detections) {
    detectionsTableBody.innerHTML = '';

    if (detections.length === 0) {
        detectionsTableBody.innerHTML = `
            <tr>
                <td colspan="4" style="text-align: center; padding: 2rem; color: var(--secondary-color);">
                    No detections found
                </td>
            </tr>
        `;
        detectionCount.textContent = 'Total: 0 detection(s)';
        return;
    }

    detections.forEach(detection => {
        const row = document.createElement('tr');

        const confidence = (detection.confidence * 100).toFixed(0);
        const status = detection.status || 'unknown';
        const badgeClass = status === 'registered' ? 'success' : 'warning';

        row.innerHTML = `
            <td><strong>${detection.plate_number}</strong></td>
            <td>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%"></div>
                    <span class="confidence-text">${confidence}%</span>
                </div>
            </td>
            <td>
                <span class="badge badge-${badgeClass}">
                    ${status.toUpperCase()}
                </span>
            </td>
            <td>${formatDate(detection.detected_at)}</td>
        `;

        detectionsTableBody.appendChild(row);
    });

    detectionCount.textContent = `Total: ${detections.length} detection(s)`;
}

// Initial load
loadDetections();

// Auto-refresh every 30 seconds
setInterval(loadDetections, 30000);
