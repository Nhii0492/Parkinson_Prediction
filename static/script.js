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

// Tab Elements
const uploadTab = document.getElementById('uploadTab');
const drawTab = document.getElementById('drawTab');
const uploadSection = document.getElementById('uploadSection');
const drawSection = document.getElementById('drawSection');

// Canvas Elements
const drawCanvas = document.getElementById('drawCanvas');
const clearCanvasBtn = document.getElementById('clearCanvasBtn');
const useDrawingBtn = document.getElementById('useDrawingBtn');

// State
let selectedFile = null;
let isDrawing = false;
let currentMode = 'upload';

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

// Tab switching
uploadTab.addEventListener('click', () => switchTab('upload'));
drawTab.addEventListener('click', () => switchTab('draw'));

// Canvas drawing
let ctx = drawCanvas.getContext('2d');

// Setup canvas with background similar to real paper
function setupCanvas() {
    // Fill with light gray background (simulating paper)
    ctx.fillStyle = '#f5f5f5';
    ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    
    // Use darker color for drawing (blue/black instead of pure black)
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

setupCanvas();

// Mouse events
drawCanvas.addEventListener('mousedown', startDrawing);
drawCanvas.addEventListener('mousemove', draw);
drawCanvas.addEventListener('mouseup', stopDrawing);
drawCanvas.addEventListener('mouseout', stopDrawing);

// Touch events
drawCanvas.addEventListener('touchstart', handleTouch);
drawCanvas.addEventListener('touchmove', handleTouch);
drawCanvas.addEventListener('touchend', stopDrawing);

// Clear canvas
clearCanvasBtn.addEventListener('click', clearCanvas);

// Use drawing
useDrawingBtn.addEventListener('click', useDrawing);

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
// Tab Functions
// ==========================================
function switchTab(mode) {
    currentMode = mode;
    
    if (mode === 'upload') {
        uploadTab.classList.add('active');
        drawTab.classList.remove('active');
        uploadSection.style.display = 'block';
        drawSection.style.display = 'none';
    } else {
        drawTab.classList.add('active');
        uploadTab.classList.remove('active');
        uploadSection.style.display = 'none';
        drawSection.style.display = 'block';
    }
    
    hideError();
    hideResult();
}

// ==========================================
// Canvas Drawing Functions
// ==========================================
function getCanvasCoordinates(e) {
    const rect = drawCanvas.getBoundingClientRect();
    const scaleX = drawCanvas.width / rect.width;
    const scaleY = drawCanvas.height / rect.height;
    
    if (e.touches) {
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY
        };
    }
    
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    isDrawing = true;
    const coords = getCanvasCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
    useDrawingBtn.disabled = false;
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    
    const coords = getCanvasCoordinates(e);
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
}

function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        ctx.beginPath();
    }
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                       e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    drawCanvas.dispatchEvent(mouseEvent);
}

function clearCanvas() {
    setupCanvas();
    useDrawingBtn.disabled = true;
    hideError();
    hideResult();
}

function resetDrawing() {
    clearCanvas();
    selectedFile = null;
    analyzeBtn.disabled = true;
}

function useDrawing() {
    drawCanvas.toBlob((blob) => {
        if (!blob) {
            showError('Failed to create image from drawing');
            return;
        }
        
        const file = new File([blob], 'drawing.png', { type: 'image/png' });
        selectedFile = file;
        
        previewImage.src = drawCanvas.toDataURL();
        previewContainer.style.display = 'block';
        uploadBox.style.display = 'none';
        analyzeBtn.disabled = false;
        
        switchTab('upload');
        hideError();
        hideResult();
    }, 'image/png');
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

