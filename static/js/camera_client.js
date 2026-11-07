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
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCtx = this.offscreenCanvas.getContext('2d');
    this.browserLoopActive = false;
    this.browserLoopTimeout = null;

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
    const btnStartBrowser = document.getElementById('btn-start-browser-cam');
    if (btnStartBrowser) {
      btnStartBrowser.addEventListener('click', () => {
        const prompt = document.getElementById('cloud-camera-prompt');
        if (prompt) prompt.style.display = 'none';
        const sourceSelect = document.getElementById('camera-source-select');
        if (sourceSelect) sourceSelect.value = 'browser';
        this.setMode('browser');
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
        this.cameraAvailable = data.camera_available;
        const isRemote = window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';
        if (data.camera_available === false || isRemote) {
          const prompt = document.getElementById('cloud-camera-prompt');
          if (prompt && this.mode === 'server') {
            prompt.style.display = 'flex';
          }
          const sourceSelect = document.getElementById('camera-source-select');
          if (sourceSelect) {
            sourceSelect.value = 'browser';
          }
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
    const prompt = document.getElementById('cloud-camera-prompt');
    if (prompt) {
      prompt.style.display = (mode === 'server' && this.cameraAvailable === false) ? 'flex' : 'none';
    }
    if (this.mode === mode) return;

    this.stopCurrentMode();
    this.mode = mode;

    if (!this.isCameraOn) {
      this.updateCameraUI();
      return;
    }

    if (mode === 'server') {
      if (this.videoEl) this.videoEl.style.display = 'none';
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
      if (this.videoEl) this.videoEl.style.display = 'block';
      this.canvasEl.style.display = 'block';
      this.canvasEl.style.position = 'absolute';
      this.startBrowserWebcam();
    } else if (mode === 'simulation') {
      if (this.videoEl) this.videoEl.style.display = 'none';
      this.serverImgEl.style.display = 'none';
      this.canvasEl.style.display = 'block';
      this.canvasEl.style.position = 'static';
      this.startSimulation();
    }
  }

  _stopBrowserStream() {
    this.browserLoopActive = false;
    if (this.browserLoopTimeout) {
      clearTimeout(this.browserLoopTimeout);
      this.browserLoopTimeout = null;
    }
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
    this._stopBrowserStream();
    try {
      try {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 }
          },
          audio: false
        });
      } catch (constraintErr) {
        // Fallback to standard video stream if exact resolution is occupied
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });
      }
      this.videoEl.srcObject = this.mediaStream;
      await this.videoEl.play();

      // Wait until video has valid dimensions so face is never distorted
      if (!this.videoEl.videoWidth || !this.videoEl.videoHeight) {
        await new Promise((resolve) => {
          this.videoEl.onloadedmetadata = () => resolve();
          setTimeout(resolve, 350);
        });
      }

      const vw = this.videoEl.videoWidth || 640;
      const vh = this.videoEl.videoHeight || 480;

      // Capture at optimized 480px width (cuts network payload by 75% for zero-lag upload)
      const capW = 480;
      const capH = Math.round((vh / vw) * capW);
      this.offscreenCanvas.width = capW;
      this.offscreenCanvas.height = capH;

      // Ensure visible canvas is ready
      const ctx = this.canvasEl.getContext('2d');
      this.canvasEl.width = 640;
      this.canvasEl.height = 480;

      this.browserLoopActive = true;
      this.isProcessing = false;

      const prompt = document.getElementById('cloud-camera-prompt');
      if (prompt) prompt.style.display = 'none';

      // Non-blocking adaptive dispatch loop (eliminates blinking, queues, and hanging)
      const dispatchFrame = async () => {
        if (!this.browserLoopActive || !this.isCameraOn || this.mode !== 'browser') {
          return;
        }

        try {
          // Draw clean upright video frame to in-memory offscreen canvas
          this.offscreenCtx.drawImage(this.videoEl, 0, 0, capW, capH);

          const base64Data = this.offscreenCanvas.toDataURL('image/jpeg', 0.60);

          const response = await fetch('/api/process_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Data, return_annotated: false })
          });

          if (response.ok) {
            const data = await response.json();
            if (data.success) {
              // Real-time HUD layer rendered directly over native 60fps video
              this.drawHudOverlay(data.landmarks, data.telemetry);
            }
            if (data.telemetry && this.onTelemetry) {
              this.onTelemetry(data.telemetry);
            }
          }
        } catch (err) {
          console.warn('Frame processing dispatch error:', err);
        }

        // Schedule next frame with a gentle 45ms pacing (clean 12-15 FPS without network backlog)
        if (this.browserLoopActive && this.isCameraOn && this.mode === 'browser') {
          this.browserLoopTimeout = setTimeout(dispatchFrame, 45);
        }
      };

      dispatchFrame();

    } catch (err) {
      console.error('Browser webcam error:', err);
      let msg = 'Unable to access camera: ' + err.message;
      if (err.name === 'NotReadableError' || (err.message && err.message.toLowerCase().includes('in use'))) {
        msg = '📷 Camera Hardware Locked: Another app is currently using your webcam (such as Zoom, Microsoft Teams, Skype, or another browser tab).\n\nPlease close Zoom or other apps using your camera, then click "Connect Device Camera" again.';
      } else if (err.name === 'NotAllowedError') {
        msg = 'Camera permission was denied.\n\nPlease click the lock or camera icon in your browser address bar and choose "Allow".';
      }
      alert(msg);
    }
  }

    drawHudOverlay(summary, telemetry) {
    const ctx = this.canvasEl.getContext('2d');
    const cw = this.canvasEl.width;
    const ch = this.canvasEl.height;

    // Clear transparent overlay
    ctx.clearRect(0, 0, cw, ch);

    if (!summary || !summary.bbox) {
      // Draw subtle searching target box
      const cx = cw / 2;
      const cy = ch / 2;
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(cx - 70, cy - 70, 140, 140);
      ctx.fillStyle = 'rgba(0, 240, 255, 0.7)';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('SCANNING FOR FACE...', cx, cy - 80);
      return;
    }

    const isClosed = telemetry ? telemetry.is_closed : false;
    const isYawning = telemetry ? telemetry.is_yawning : false;
    const isNodding = telemetry ? telemetry.is_nodding : false;

    // 1. Draw Corner HUD brackets around face area directly
    const [x1, y1, x2, y2] = summary.bbox;
    const bx1 = x1 * cw;
    const bx2 = x2 * cw;
    const by1 = y1 * ch;
    const by2 = y2 * ch;
    const bw = bx2 - bx1;
    const bh = by2 - by1;

    const bracketLen = Math.min(22, Math.max(12, bw * 0.22));
    ctx.strokeStyle = isClosed ? '#ef4444' : '#00f0ff';
    ctx.lineWidth = 2;

    // Top-Left
    ctx.beginPath();
    ctx.moveTo(bx1, by1 + bracketLen);
    ctx.lineTo(bx1, by1);
    ctx.lineTo(bx1 + bracketLen, by1);
    ctx.stroke();

    // Top-Right
    ctx.beginPath();
    ctx.moveTo(bx2 - bracketLen, by1);
    ctx.lineTo(bx2, by1);
    ctx.lineTo(bx2, by1 + bracketLen);
    ctx.stroke();

    // Bottom-Left
    ctx.beginPath();
    ctx.moveTo(bx1, by2 - bracketLen);
    ctx.lineTo(bx1, by2);
    ctx.lineTo(bx1 + bracketLen, by2);
    ctx.stroke();

    // Bottom-Right
    ctx.beginPath();
    ctx.moveTo(bx2 - bracketLen, by2);
    ctx.lineTo(bx2, by2);
    ctx.lineTo(bx2, by2 - bracketLen);
    ctx.stroke();

    // 2. Draw Eye Contours
    const eyeColor = isClosed ? '#ef4444' : '#38bdf8';
    ctx.strokeStyle = eyeColor;
    ctx.lineWidth = isClosed ? 2.5 : 1.5;

    const drawContour = (pts) => {
      if (!pts || pts.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(pts[0][0] * cw, pts[0][1] * ch);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i][0] * cw, pts[i][1] * ch);
      }
      ctx.closePath();
      ctx.stroke();
    };

    drawContour(summary.left_eye);
    drawContour(summary.right_eye);

    // 3. Draw Mouth
    if (summary.mouth) {
      ctx.strokeStyle = isYawning ? '#f59e0b' : '#10b981';
      ctx.lineWidth = isYawning ? 2.5 : 1.5;
      drawContour(summary.mouth);
    }

    // 4. Draw 3D Nose Pose Direction Vector
    if (summary.nose_vector) {
      const [nx1, ny1, nx2, ny2] = summary.nose_vector;
      const sx = nx1 * cw;
      const sy = ny1 * ch;
      const ex = nx2 * cw;
      const ey = ny2 * ch;

      ctx.strokeStyle = '#00f0ff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
      ctx.stroke();

      ctx.fillStyle = '#00f0ff';
      ctx.beginPath();
      ctx.arc(sx, sy, 3, 0, Math.PI * 2);
      ctx.fill();
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
