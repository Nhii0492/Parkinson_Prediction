/**
 * Frontend JavaScript for Parkinson's Disease Detection
 * Handles image upload, API calls, and result display
 */

// API endpoint (auto-detect from current location)
const API_URL = window.location.origin;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const selectBtn = document.getElementById('selectBtn');
const uploadBox = document.getElementById('uploadBox');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('resultSection');
const resultCard = document.getElementById('resultCard');
const resultLabel = document.getElementById('resultLabel');
const resultConfidence = document.getElementById('resultConfidence');
const resultMessage = document.getElementById('resultMessage');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

// State
let selectedFile = null;

// ==========================================
// Event Listeners
// ==========================================

// Click button để chọn file
selectBtn.addEventListener('click', () => {
    fileInput.click();
});

// Click upload box to select file (only if no file is selected)
uploadBox.addEventListener('click', (e) => {
    // Only trigger if preview is not showing
    if (previewContainer.style.display === 'none') {
        fileInput.click();
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileSelect(file);
    }
});

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
    } else {
        showError('Please select a valid image file');
    }
});

// Remove image
removeBtn.addEventListener('click', () => {
    resetUpload();
});

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    await analyzeImage(selectedFile);
});

// ==========================================
// Functions
// ==========================================

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file (JPG, PNG, etc.)');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File too large. Please select a file smaller than 10MB');
        return;
    }

    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        uploadBox.style.display = 'none';
        analyzeBtn.disabled = false;
        hideError();
        hideResult();
    };
    reader.readAsDataURL(file);
}

/**
 * Reset upload state
 */
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewContainer.style.display = 'none';
    uploadBox.style.display = 'block';
    analyzeBtn.disabled = true;
    hideError();
    hideResult();
}

/**
 * Analyze image using API
 */
async function analyzeImage(file) {
    // Show loading, hide result and error
    showLoading();
    hideResult();
    hideError();
    analyzeBtn.disabled = true;

    try {
        // Create FormData
        const formData = new FormData();
        formData.append('file', file);

        // Call API
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        // Check response
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        // Parse result
        const result = await response.json();
        
        // Display result
        displayResult(result);

    } catch (error) {
        console.error('Error:', error);
        showError(`Error during analysis: ${error.message}`);
    } finally {
        hideLoading();
        analyzeBtn.disabled = false;
    }
}

/**
 * Display result
 */
function displayResult(result) {
    const { label, confidence } = result;
    
    // Set label
    resultLabel.textContent = label;
    
    // Hide confidence display
    resultConfidence.style.display = 'none';
    
    // Set message
    if (label === 'Healthy') {
        resultMessage.textContent = 
            'The spiral drawing shows characteristics of a healthy individual. ' +
            'However, this is a support tool and does not replace professional medical diagnosis.';
    } else {
        resultMessage.textContent = 
            'The spiral drawing shows characteristics that may be related to Parkinson\'s disease. ' +
            'Please consult a medical specialist for accurate diagnosis.';
    }
    
    // Set card class and style
    resultCard.className = 'result-card';
    resultCard.classList.add(label.toLowerCase());
    
    // Show result section
    resultSection.style.display = 'block';
    
    // Scroll to result
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Show/hide loading
 */
function showLoading() {
    loading.style.display = 'block';
}

function hideLoading() {
    loading.style.display = 'none';
}

/**
 * Show/hide error
 */
function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
    errorSection.style.display = 'none';
}

/**
 * Show/hide result
 */
function hideResult() {
    resultSection.style.display = 'none';
}

// ==========================================
// Health Check on Load
// ==========================================
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (!response.ok) {
            console.warn('API not responding. Please check backend.');
        }
    } catch (error) {
        console.warn('Cannot connect to API. Ensure backend is running at', API_URL);
    }
});

