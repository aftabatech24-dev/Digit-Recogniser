const canvas = document.querySelector('#drawing-canvas');
const context = canvas.getContext('2d', { willReadFrequently: true });
const clearButton = document.querySelector('#clear-button');
const uploadButton = document.querySelector('#upload-button');
const imageUpload = document.querySelector('#image-upload');
const predictButton = document.querySelector('#predict-button');
const expectedDigit = document.querySelector('#expected-digit');
const resultCard = document.querySelector('#result-card');
const emptyState = document.querySelector('#empty-state');
const resultState = document.querySelector('#result-state');

let isDrawing = false;
let lastPoint = null;

function prepareCanvas() {
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.strokeStyle = '#132238';
  context.lineWidth = 34;
}

function pointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function beginDrawing(event) {
  isDrawing = true;
  lastPoint = pointFromEvent(event);
  canvas.setPointerCapture(event.pointerId);
}

function draw(event) {
  if (!isDrawing) return;
  const currentPoint = pointFromEvent(event);
  context.beginPath();
  context.moveTo(lastPoint.x, lastPoint.y);
  context.lineTo(currentPoint.x, currentPoint.y);
  context.stroke();
  lastPoint = currentPoint;
}

function finishDrawing() {
  isDrawing = false;
  lastPoint = null;
}

canvas.addEventListener('pointerdown', beginDrawing);
canvas.addEventListener('pointermove', draw);
canvas.addEventListener('pointerup', finishDrawing);
canvas.addEventListener('pointerleave', finishDrawing);
canvas.addEventListener('pointercancel', finishDrawing);

clearButton.addEventListener('click', () => {
  prepareCanvas();
  resetResult();
});

uploadButton.addEventListener('click', () => imageUpload.click());

imageUpload.addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    const image = new Image();
    image.onload = () => {
      prepareCanvas();
      const scale = Math.min(canvas.width / image.width, canvas.height / image.height) * 0.78;
      const width = image.width * scale;
      const height = image.height * scale;
      context.drawImage(image, (canvas.width - width) / 2, (canvas.height - height) / 2, width, height);
      resetResult();
    };
    image.src = reader.result;
  };
  reader.readAsDataURL(file);
});

predictButton.addEventListener('click', async () => {
  if (expectedDigit.value && !/^[0-9]$/.test(expectedDigit.value)) {
    showError('Expected digit must be a single number from 0 to 9.');
    return;
  }

  predictButton.disabled = true;
  predictButton.innerHTML = 'Recognizing <span class="dot">•</span>';

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: canvas.toDataURL('image/png'),
        expected_digit: expectedDigit.value,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Prediction could not be completed.');
    showResult(data);
  } catch (error) {
    showError(error.message);
  } finally {
    predictButton.disabled = false;
    predictButton.innerHTML = 'Recognize digit <span>→</span>';
  }
});

function resetResult() {
  resultCard.className = 'result-card';
  emptyState.hidden = false;
  resultState.hidden = true;
}

function showError(message) {
  resultCard.className = 'result-card error';
  emptyState.hidden = false;
  resultState.hidden = true;
  emptyState.querySelector('h2').textContent = 'Try again';
  emptyState.querySelector('p').textContent = message;
}

function showResult(data) {
  const hasComparison = data.is_match !== null;
  resultCard.className = `result-card ${hasComparison ? (data.is_match ? 'match' : 'mismatch') : ''}`;
  emptyState.hidden = true;
  resultState.hidden = false;

  document.querySelector('#prediction-number').textContent = data.prediction;
  document.querySelector('#confidence-number').textContent = `${data.confidence}%`;
  requestAnimationFrame(() => {
    document.querySelector('#confidence-fill').style.width = `${data.confidence}%`;
  });

  document.querySelector('#model-accuracy').textContent = `${data.accuracy}%`;

  const comparison = document.querySelector('#comparison');
  comparison.hidden = !hasComparison;
  if (hasComparison) {
    comparison.className = `comparison ${data.is_match ? 'correct' : 'different'}`;
    comparison.textContent = data.is_match
      ? `Correct — this matches your expected digit ${data.expected_digit}.`
      : `Different — you entered ${data.expected_digit}, while the model detected ${data.prediction}.`;
  }

  const predictionList = document.querySelector('#prediction-list');
  predictionList.innerHTML = data.top_predictions.map((item, index) => `
    <div class="prediction-option ${index === 0 ? 'best' : ''}">
      <span>${item.digit}</span><strong>${item.confidence}%</strong>
    </div>
  `).join('');
}

prepareCanvas();
