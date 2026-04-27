document.addEventListener('DOMContentLoaded', () => {
    const currentSignEl = document.getElementById('current-sign');
    const confidenceFillEl = document.getElementById('confidence-fill');
    const confidenceTextEl = document.getElementById('confidence-text');
    const historyTextArea = document.getElementById('sentence-history');
    const toggleBtn = document.getElementById('camera-toggle-btn');
    const toggleBtnText = document.getElementById('toggle-btn-text');
    const videoStreamEl = document.getElementById('video-stream');
    const cameraOfflineEl = document.getElementById('camera-offline');
    const liveIndicator = document.getElementById('live-indicator');

    const btnCopy = document.getElementById('btn-copy');
    const btnClear = document.getElementById('btn-clear');
    const btnEdit = document.getElementById('btn-edit');

    let cameraActive = false;
    let isEditing = false;
    let startTime;
    let timerInterval;

    function updateTimer() {
        const delta = Date.now() - startTime;
        const h = Math.floor(delta / 3600000).toString().padStart(2, '0');
        const m = Math.floor((delta % 3600000) / 60000).toString().padStart(2, '0');
        const s = Math.floor((delta % 60000) / 1000).toString().padStart(2, '0');
        document.getElementById('session-timer').innerText = `${h}:${m}:${s}`;
    }

    btnEdit.addEventListener('click', async () => {
        isEditing = !isEditing;
        if (isEditing) {
            historyTextArea.readOnly = false;
            btnEdit.innerHTML = '<i class="fa-solid fa-unlock"></i>';
            btnEdit.classList.add('editing-active');
            historyTextArea.focus();
        } else {
            historyTextArea.readOnly = true;
            btnEdit.innerHTML = '<i class="fa-solid fa-lock"></i>';
            btnEdit.classList.remove('editing-active');
            await fetch('/update_history', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ new_history: historyTextArea.value })
            });
        }
    });

    btnClear.addEventListener('click', async () => {
        if(confirm("Full history delete karein?")) {
            historyTextArea.value = "";
            await fetch('/clear_history', { method: 'POST' });
        }
    });

    btnCopy.addEventListener('click', () => {
        navigator.clipboard.writeText(historyTextArea.value);
        btnCopy.innerHTML = '<i class="fa-solid fa-check" style="color: #10b981"></i>';
        setTimeout(() => btnCopy.innerHTML = '<i class="fa-solid fa-copy"></i>', 1500);
    });

    toggleBtn.addEventListener('click', async () => {
        const action = cameraActive ? 'stop' : 'start';
        const response = await fetch('/toggle_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action })
        });
        const result = await response.json();
        if (result.status === 'running') {
            cameraActive = true;
            toggleBtn.className = 'main-cta stop-mode';
            toggleBtnText.innerText = 'Terminate';
            videoStreamEl.src = "/video_feed?" + new Date().getTime();
            videoStreamEl.style.display = 'block';
            cameraOfflineEl.style.display = 'none';
            liveIndicator.style.display = 'flex';
            startTime = Date.now();
            timerInterval = setInterval(updateTimer, 1000);
        } else {
            cameraActive = false;
            toggleBtn.className = 'main-cta start-mode';
            toggleBtnText.innerText = 'Start Engine';
            videoStreamEl.style.display = 'none';
            cameraOfflineEl.style.display = 'flex';
            liveIndicator.style.display = 'none';
            clearInterval(timerInterval);
            document.getElementById('session-timer').innerText = "00:00:00";
        }
    });

    setInterval(async () => {
        if (!cameraActive) return;
        try {
            const response = await fetch('/stats');
            const data = await response.json();
            currentSignEl.innerText = data.sign;
            const conf = (data.confidence * 100).toFixed(0);
            confidenceTextEl.innerText = conf + "%";
            confidenceFillEl.style.width = conf + "%";
            document.getElementById('fps-val').innerText = data.fps || "0";
            if (!isEditing) {
                historyTextArea.value = data.history;
                historyTextArea.scrollTop = historyTextArea.scrollHeight;
            }
        } catch (e) {}
    }, 300);
});