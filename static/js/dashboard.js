// ==================== DASHBOARD JAVASCRIPT ====================

let selectedFile = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const previewImage = document.getElementById('previewImage');
const uploadBtn = document.getElementById('uploadBtn');
const detectBtn = document.getElementById('detectBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const noResults = document.getElementById('noResults');
const detectionResults = document.getElementById('detectionResults');
const registrationForm = document.getElementById('registrationForm');
const quickRegisterForm = document.getElementById('quickRegisterForm');
const cancelRegisterBtn = document.getElementById('cancelRegisterBtn');

// Event Listeners
uploadBtn.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('click', () => imageInput.click());
detectBtn.addEventListener('click', detectPlate);
clearBtn.addEventListener('click', clearImage);
quickRegisterForm.addEventListener('submit', handleQuickRegister);

if (cancelRegisterBtn) {
    cancelRegisterBtn.addEventListener('click', cancelRegistration);
}

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary-color)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--secondary-color)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--secondary-color)';

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

/**
 * Handle file selection
 */
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

/**
 * Handle file processing
 */
function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showToast('Please select a valid image file (JPG, PNG, BMP)', 'error');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showToast('File size must be less than 16MB', 'error');
        return;
    }

    selectedFile = file;

    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        uploadPlaceholder.style.display = 'none';
        detectBtn.disabled = false;
        clearBtn.disabled = false;
    };
    reader.readAsDataURL(file);

    // Clear previous results
    clearResults();
}

/**
 * Detect license plate in image
 */
async function detectPlate() {
    if (!selectedFile) {
        showToast('Please select an image first', 'error');
        return;
    }

    // Show loading
    loadingSpinner.style.display = 'block';
    noResults.style.display = 'none';
    detectionResults.style.display = 'none';
    detectBtn.disabled = true;

    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', selectedFile);

        // Send request
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Detection failed');
        }

        // Hide loading
        loadingSpinner.style.display = 'none';

        if (data.success && data.detections && data.detections.length > 0) {
            displayResults(data);
        } else {
            noResults.style.display = 'block';
            noResults.innerHTML = '<p>⚠️ No license plates detected in the image</p>';
        }

        // Update stats
        updateStats();

    } catch (error) {
        loadingSpinner.style.display = 'none';
        noResults.style.display = 'block';
        noResults.innerHTML = `<p style="color: var(--danger-color);">Error: ${error.message}</p>`;
        showToast(`Detection failed: ${error.message}`, 'error');
    } finally {
        detectBtn.disabled = false;
    }
}

/**
 * Display detection results
 */
function displayResults(data) {
    detectionResults.innerHTML = '';
    detectionResults.style.display = 'block';

    // Update preview with annotated image
    if (data.image_url) {
        previewImage.src = data.image_url + '?t=' + Date.now();
    }

    // Display each detection
    data.detections.forEach((detection, index) => {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'detection-result';

        const plateNumber = detection.plate_number || 'Unknown';
        const isRegistered = detection.registered;
        const detectionConf = (detection.detection_confidence * 100).toFixed(1);
        const ocrConf = (detection.ocr_confidence * 100).toFixed(1);

        resultDiv.innerHTML = `
            <h4>Detection ${index + 1}: ${plateNumber}</h4>
            <div class="result-item">
                <span class="result-label">Plate Number:</span>
                <span class="result-value"><strong>${plateNumber}</strong></span>
            </div>
            <div class="result-item">
                <span class="result-label">Detection Confidence:</span>
                <span class="result-value">${detectionConf}%</span>
            </div>
            <div class="result-item">
                <span class="result-label">OCR Confidence:</span>
                <span class="result-value">${ocrConf}%</span>
            </div>
            <div class="result-item">
                <span class="result-label">Status:</span>
                <span class="badge badge-${isRegistered ? 'success' : 'warning'}">
                    ${isRegistered ? 'REGISTERED' : 'UNREGISTERED'}
                </span>
            </div>
            ${isRegistered && detection.vehicle_info ? `
                <div class="result-item">
                    <span class="result-label">Owner:</span>
                    <span class="result-value">${detection.vehicle_info.owner_name}</span>
                </div>
                ${detection.vehicle_info.vehicle_type ? `
                <div class="result-item">
                    <span class="result-label">Type:</span>
                    <span class="result-value">${detection.vehicle_info.vehicle_type}</span>
                </div>
                ` : ''}
            ` : ''}
            
            ${!isRegistered && plateNumber !== 'Unknown' ? `
                <button class="btn btn-sm btn-outline-primary" onclick="populateRegistration('${plateNumber}')" style="margin-top: 10px; width: 100%;">
                    Register This Vehicle
                </button>
            ` : ''}
        `;

        detectionResults.appendChild(resultDiv);
    });
}

/**
 * Clear image and results
 */
function clearImage() {
    selectedFile = null;
    imageInput.value = '';
    previewImage.src = '';
    previewImage.style.display = 'none';
    uploadPlaceholder.style.display = 'block';
    detectBtn.disabled = true;
    clearBtn.disabled = true;
    clearResults();
}

/**
 * Clear results
 */
function clearResults() {
    detectionResults.innerHTML = '';
    detectionResults.style.display = 'none';
    noResults.style.display = 'block';
    noResults.innerHTML = '<p>Upload an image and click "Detect Plate" to see results</p>';
    registrationForm.style.display = 'none';
    quickRegisterForm.reset();
}

/**
 * Handle quick registration
 */
async function handleQuickRegister(e) {
    e.preventDefault();

    const formData = new FormData(quickRegisterForm);
    const data = Object.fromEntries(formData.entries());

    try {
        const result = await apiRequest('/api/vehicles', {
            method: 'POST',
            body: JSON.stringify(data)
        });

        if (result.success) {
            showToast('Vehicle registered successfully!', 'success');
            quickRegisterForm.reset();
            registrationForm.style.display = 'none';
            updateStats();

            // Re-detect to update status
            if (selectedFile) {
                setTimeout(() => detectPlate(), 500);
            }
        }
    } catch (error) {
        showToast(`Registration failed: ${error.message} `, 'error');
    }
}

/**
 * Update statistics
 */
async function updateStats() {
    try {
        const data = await apiRequest('/api/stats');
        if (data.success) {
            document.getElementById('total-vehicles').textContent = data.stats.total_vehicles;
            document.getElementById('recent-detections').textContent = data.stats.recent_detections;
        }
    } catch (error) {
        console.error('Failed to update stats:', error);
    }
}

// Initial load
updateStats();

/**
 * Populate registration form with specific plate
 * Called from manual "Register" buttons
 */
function populateRegistration(plateNumber) {
    registrationForm.style.display = 'block';
    document.getElementById('plateNumber').value = plateNumber;
    registrationForm.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Cancel registration and hide form
 */
function cancelRegistration() {
    registrationForm.style.display = 'none';
    quickRegisterForm.reset();
}
