/**
 * DriveSafe - Camera Client and Interactive Driver Simulator
 * Manages Server Stream, Client Browser WebRTC Camera, and Interactive Simulation presets.
 */
class CameraController {
  constructor(onTelemetryCallback, onStateChangeCallback) {
    this.onTelemetry = onTelemetryCallback;
    this.onStateChange = onStateChangeCallback;
    this.mode = 'server'; // 'server', 'browser', 'simulation'
    this.isCameraOn = true;
    this.mediaStream = null;
    this.videoEl = document.getElementById('client-webcam-video');
    this.canvasEl = document.getElementById('client-webcam-canvas');
    this.serverImgEl = document.getElementById('webcam-feed');
    this.captureInterval = null;
    this.isProcessing = false;

    // UI Elements for Camera On/Off
    this.btnToggle = document.getElementById('btn-toggle-camera');
    this.btnOverlayOn = document.getElementById('btn-overlay-camera-on');
    this.offOverlay = document.getElementById('camera-off-overlay');
    this.btnText = document.getElementById('camera-btn-text');

    // Simulation State
    this.simPreset = 'alert';
    this.simInterval = null;
    this.simEyeState = 'open';
    this.simTimer = 0;

    this._bindEvents();
    this.syncInitialState();
  }

  _bindEvents() {
    if (this.btnToggle) {
      this.btnToggle.addEventListener('click', () => {
        this.toggleCamera();
      });
    }
    if (this.btnOverlayOn) {
      this.btnOverlayOn.addEventListener('click', () => {
        this.toggleCamera(true);
      });
    }
  }

  syncInitialState() {
    // Check initial backend camera status
    fetch('/api/camera/state')
      .then(res => res.json())
      .then(data => {
        if (typeof data.camera_enabled === 'boolean') {
          this.isCameraOn = data.camera_enabled;
          this.updateCameraUI();
        }
      })
      .catch(() => {});
  }

  updateCameraUI() {
    if (this.btnToggle) {
      this.btnToggle.classList.toggle('active', this.isCameraOn);
      this.btnToggle.classList.toggle('off', !this.isCameraOn);

      const iconOn = this.btnToggle.querySelector('.cam-icon-on');
      const iconOff = this.btnToggle.querySelector('.cam-icon-off');
      if (iconOn && iconOff) {
        iconOn.style.display = this.isCameraOn ? 'inline-block' : 'none';
        iconOff.style.display = this.isCameraOn ? 'none' : 'inline-block';
      }

      if (this.btnText) {
        this.btnText.textContent = this.isCameraOn ? 'Camera ON' : 'Camera OFF';
      }
    }

    if (this.offOverlay) {
      this.offOverlay.classList.toggle('active', !this.isCameraOn);
    }

    if (this.onStateChange) {
      this.onStateChange(this.isCameraOn);
    }
  }

  async toggleCamera(forceState) {
    const nextState = (typeof forceState === 'boolean') ? forceState : !this.isCameraOn;
    if (this.isCameraOn === nextState) return;

    this.isCameraOn = nextState;
    this.updateCameraUI();

    if (!this.isCameraOn) {
      // TURN OFF CAMERA
      if (this.mode === 'server') {
        if (this.serverImgEl) {
          this.serverImgEl.src = '';
        }
        try {
          await fetch('/api/camera/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: false })
          });
        } catch (err) {
          console.warn('Error turning off server camera:', err);
        }
      } else if (this.mode === 'browser') {
        this._stopBrowserStream();
      } else if (this.mode === 'simulation') {
        if (this.simInterval) {
          clearInterval(this.simInterval);
          this.simInterval = null;
        }
      }
    } else {
      // TURN ON CAMERA
      if (this.mode === 'server') {
        try {
          await fetch('/api/camera/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: true })
          });
        } catch (err) {
          console.warn('Error turning on server camera:', err);
        }
        if (this.serverImgEl) {
          this.serverImgEl.style.display = 'block';
          this.canvasEl.style.display = 'none';
          this.serverImgEl.src = '/video_feed?' + Date.now();
        }
      } else if (this.mode === 'browser') {
        this.startBrowserWebcam();
      } else if (this.mode === 'simulation') {
        this.startSimulation();
      }
    }
  }

  setMode(mode) {
    if (this.mode === mode) return;

    this.stopCurrentMode();
    this.mode = mode;

    if (!this.isCameraOn) {
      this.updateCameraUI();
      return;
    }

    if (mode === 'server') {
      this.serverImgEl.style.display = 'block';
      this.canvasEl.style.display = 'none';
      fetch('/api/camera/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: true })
      }).finally(() => {
        this.serverImgEl.src = '/video_feed?' + Date.now();
      });
    } else if (mode === 'browser') {
      this.serverImgEl.style.display = 'none';
      this.canvasEl.style.display = 'block';
      this.startBrowserWebcam();
    } else if (mode === 'simulation') {
      this.serverImgEl.style.display = 'none';
      this.canvasEl.style.display = 'block';
      this.startSimulation();
    }
  }

  _stopBrowserStream() {
    if (this.captureInterval) {
      clearInterval(this.captureInterval);
      this.captureInterval = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    if (this.videoEl) {
      this.videoEl.srcObject = null;
    }
  }

  stopCurrentMode() {
    this._stopBrowserStream();
    if (this.simInterval) {
      clearInterval(this.simInterval);
      this.simInterval = null;
    }
    if (this.serverImgEl) {
      this.serverImgEl.src = '';
    }
  }

  async startBrowserWebcam() {
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { max: 30 } },
        audio: false
      });
      this.videoEl.srcObject = this.mediaStream;
      await this.videoEl.play();

      const ctx = this.canvasEl.getContext('2d');
      this.canvasEl.width = 640;
      this.canvasEl.height = 480;

      // Frame dispatch loop (~18 FPS)
      this.captureInterval = setInterval(() => {
        if (this.isProcessing || !this.isCameraOn) return;
        this.isProcessing = true;

        // Draw mirror video frame to canvas
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(this.videoEl, -640, 0, 640, 480);
        ctx.restore();

        const base64Data = this.canvasEl.toDataURL('image/jpeg', 0.65);

        fetch('/api/process_frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: base64Data, return_annotated: true })
        })
          .then(res => res.json())
          .then(data => {
            if (data.success && data.annotated_frame) {
              const img = new Image();
              img.onload = () => {
                ctx.drawImage(img, 0, 0, 640, 480);
              };
              img.src = data.annotated_frame;
            }
            if (data.telemetry && this.onTelemetry) {
              this.onTelemetry(data.telemetry);
            }
          })
          .catch(err => console.warn('Frame processing error:', err))
          .finally(() => {
            this.isProcessing = false;
          });
      }, 55);
    } catch (err) {
      alert('Unable to access browser camera: ' + err.message + '\nSwitching back to server feed.');
      this.setMode('server');
    }
  }

  startSimulation() {
    const ctx = this.canvasEl.getContext('2d');
    this.canvasEl.width = 640;
    this.canvasEl.height = 480;
    this.simTimer = 0;

    if (this.simInterval) clearInterval(this.simInterval);

    this.simInterval = setInterval(() => {
      if (!this.isCameraOn) return;
      this.simTimer += 0.1;
      this._drawSimulationCanvas(ctx, this.simPreset, this.simTimer);
    }, 100);
  }

  setSimPreset(presetName) {
    this.simPreset = presetName;
  }

  _drawSimulationCanvas(ctx, preset, timer) {
    // Futuristic Simulation Canvas Art
    ctx.fillStyle = '#080d16';
    ctx.fillRect(0, 0, 640, 480);

    // Grid lines
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.08)';
    ctx.lineWidth = 1;
    for (let x = 0; x < 640; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 480); ctx.stroke();
    }
    for (let y = 0; y < 480; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(640, y); ctx.stroke();
    }

    const cx = 320, cy = 230;

    // Head Contour
    let pitchOffset = 0;
    if (preset === 'nodding') {
      pitchOffset = Math.sin(timer * 2.5) > 0.2 ? 35 : 0;
    }

    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(cx, cy + pitchOffset, 90, 120, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Eyes
    let isClosed = false;
    if (preset === 'microsleep') {
      isClosed = true;
    } else if (preset === 'drowsy') {
      isClosed = Math.sin(timer * 2) > 0.3;
    } else {
      isClosed = Math.sin(timer * 4) > 0.95; // Brief normal blink
    }

    ctx.strokeStyle = isClosed ? '#ef4444' : '#10b981';
    ctx.lineWidth = 3;

    // Left eye
    ctx.beginPath();
    if (isClosed) {
      ctx.moveTo(cx - 50, cy - 25 + pitchOffset);
      ctx.lineTo(cx - 20, cy - 25 + pitchOffset);
    } else {
      ctx.ellipse(cx - 35, cy - 25 + pitchOffset, 15, 8, 0, 0, Math.PI * 2);
    }
    ctx.stroke();

    // Right eye
    ctx.beginPath();
    if (isClosed) {
      ctx.moveTo(cx + 20, cy - 25 + pitchOffset);
      ctx.lineTo(cx + 50, cy - 25 + pitchOffset);
    } else {
      ctx.ellipse(cx + 35, cy - 25 + pitchOffset, 15, 8, 0, 0, Math.PI * 2);
    }
    ctx.stroke();

    // Mouth
    let isYawn = (preset === 'yawn');
    ctx.strokeStyle = isYawn ? '#f59e0b' : '#10b981';
    ctx.beginPath();
    if (isYawn) {
      ctx.ellipse(cx, cy + 45 + pitchOffset, 22, 28, 0, 0, Math.PI * 2);
    } else {
      ctx.arc(cx, cy + 35 + pitchOffset, 20, 0.2, Math.PI - 0.2);
    }
    ctx.stroke();

    // Status overlay banner
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(15, 15, 300, 75);
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.strokeRect(15, 15, 300, 75);

    ctx.fillStyle = '#06b6d4';
    ctx.font = 'bold 13px Outfit, sans-serif';
    ctx.fillText('SIMULATION TEST RUNNING', 25, 38);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.fillText(`Scenario: ${preset.toUpperCase()}`, 25, 58);
    ctx.fillText('Testing biometric detectors & alerts', 25, 74);
  }
}
