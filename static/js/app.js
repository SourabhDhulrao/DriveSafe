/**
 * DriveSafe - Master Dashboard Controller
 * Synchronizes telemetry, risk gauge animations, audio alerts, and auto-calibration.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Chart and Audio
  const riskChart = new RiskChartManager('risk-trend-chart');

  // DOM Elements
  const riskScoreEl = document.getElementById('risk-score-value');
  const riskStateBadge = document.getElementById('risk-state-badge');
  const gaugeProgress = document.getElementById('gauge-progress-circle');
  const criticalBanner = document.getElementById('critical-alert-banner');
  const criticalReasonEl = document.getElementById('critical-alert-reason');

  // Telemetry Metric Elements
  const earValueEl = document.getElementById('val-ear');
  const earBaselineEl = document.getElementById('val-ear-baseline');
  const earBar = document.getElementById('bar-ear');
  const perclosValueEl = document.getElementById('val-perclos');
  const perclosBar = document.getElementById('bar-perclos');
  const blinkRateEl = document.getElementById('val-blink-rate');
  const blinkDurEl = document.getElementById('val-blink-duration');
  const marValueEl = document.getElementById('val-mar');
  const yawnCountEl = document.getElementById('val-yawn-count');
  const headAnglesEl = document.getElementById('val-head-angles');
  const nodCountEl = document.getElementById('val-nod-count');
  const confidenceEl = document.getElementById('val-confidence');
  const fpsBadge = document.getElementById('hud-fps-readout');

  // Risk Breakdown Bars
  const barEyeRisk = document.getElementById('bar-risk-eye');
  const barBlinkRisk = document.getElementById('bar-risk-blink');
  const barPostureRisk = document.getElementById('bar-risk-posture');
  const barYawnRisk = document.getElementById('bar-risk-yawn');
  const valEyeRisk = document.getElementById('val-risk-eye');
  const valBlinkRisk = document.getElementById('val-risk-blink');
  const valPostureRisk = document.getElementById('val-risk-posture');
  const valYawnRisk = document.getElementById('val-risk-yawn');

  // Header and Session Elements
  const sessionTimerEl = document.getElementById('session-timer');
  const audioMuteBtn = document.getElementById('btn-audio-mute');
  const audioMuteIcon = document.getElementById('audio-mute-icon');
  const calibrateBtn = document.getElementById('btn-calibrate');
  const calibOverlay = document.getElementById('calib-overlay');
  const calibCountdown = document.getElementById('calib-countdown');
  const calibQualityTag = document.getElementById('calib-quality-tag');
  const systemStatusPill = document.getElementById('system-status-pill');
  const systemStatusText = document.getElementById('system-status-text');

  // Modals
  const calibModal = document.getElementById('calib-modal');
  const eventsModal = document.getElementById('events-modal');
  const historyModal = document.getElementById('history-modal');

  // Camera Controller Initialization
  const cameraController = new CameraController(
    (telemetry) => {
      updateDashboard(telemetry);
    },
    (isCameraOn) => {
      handleCameraStateChange(isCameraOn);
    }
  );

  function handleCameraStateChange(isOn) {
    if (systemStatusText && systemStatusPill) {
      if (isOn) {
        systemStatusText.textContent = 'MONITORING';
        systemStatusPill.classList.remove('status-paused');
      } else {
        systemStatusText.textContent = 'CAMERA OFF';
        systemStatusPill.classList.add('status-paused');
      }
    }

    if (!isOn) {
      if (fpsBadge) fpsBadge.textContent = '0 FPS';
      if (confidenceEl) confidenceEl.textContent = '0%';
      if (criticalBanner) criticalBanner.classList.remove('active');
      window.audioManager.stopAlarmLoop();
    }
  }

  // State Tracking
  let sessionSeconds = 0;
  let previousRiskCode = 'safe';
  const GAUGE_CIRCUMFERENCE = 628;

  // Session Stopwatch
  setInterval(() => {
    sessionSeconds++;
    const mins = String(Math.floor(sessionSeconds / 60)).padStart(2, '0');
    const secs = String(sessionSeconds % 60).padStart(2, '0');
    if (sessionTimerEl) sessionTimerEl.textContent = `${mins}:${secs}`;
  }, 1000);

  /**
   * Primary Telemetry Update Routine
   */
  function updateDashboard(t) {
    if (!t) return;

    const score = t.risk_score || 0;
    const stateCode = t.risk_code || 'safe';
    const stateColor = t.risk_color || '#10b981';

    // 1. Update Gauge & Dynamic Colors
    document.documentElement.style.setProperty('--active-state-color', stateColor);
    document.documentElement.style.setProperty('--active-glow-color', `${stateColor}66`);

    if (riskScoreEl) riskScoreEl.textContent = Math.round(score);
    if (riskStateBadge) {
      riskStateBadge.textContent = t.risk_state || 'Safe';
      riskStateBadge.style.background = stateColor;
    }

    if (gaugeProgress) {
      const offset = GAUGE_CIRCUMFERENCE - (score / 100) * GAUGE_CIRCUMFERENCE;
      gaugeProgress.style.strokeDashoffset = offset;
    }

    // 2. Telemetry Cards
    if (earValueEl) {
      earValueEl.textContent = (t.ear || 0).toFixed(3);
      if (t.is_closed) {
        earValueEl.style.color = '#ef4444';
      } else {
        earValueEl.style.color = '#f1f5f9';
      }
    }
    if (earBar) {
      const earPct = Math.min(100, (t.ear / 0.40) * 100);
      earBar.style.width = `${earPct}%`;
      earBar.style.background = t.is_closed ? '#ef4444' : '#06b6d4';
    }
    if (earBaselineEl && t.calibration && t.calibration.baselines) {
      earBaselineEl.textContent = `Base: ${t.calibration.baselines.baseline_ear || 0.30}`;
    }

    if (perclosValueEl) {
      const perclosPct = ((t.perclos || 0) * 100).toFixed(1);
      perclosValueEl.textContent = `${perclosPct}%`;
      if (perclosBar) {
        perclosBar.style.width = `${Math.min(100, perclosPct * 2.5)}%`;
        perclosBar.style.background = (t.perclos > 0.15) ? '#ef4444' : '#06b6d4';
      }
    }

    if (blinkRateEl) blinkRateEl.textContent = `${t.blink_rate_bpm || 0}`;
    if (blinkDurEl) blinkDurEl.textContent = `Last: ${Math.round(t.last_blink_ms || 0)}ms | Avg: ${Math.round(t.avg_blink_ms || 200)}ms`;

    if (marValueEl) {
      marValueEl.textContent = (t.mar || 0).toFixed(3);
      marValueEl.style.color = t.is_yawning ? '#f59e0b' : '#f1f5f9';
    }
    if (yawnCountEl) yawnCountEl.textContent = `${t.yawn_count || 0}`;

    if (headAnglesEl) {
      headAnglesEl.textContent = `P: ${Math.round(t.pitch || 0)}° | Y: ${Math.round(t.yaw || 0)}° | R: ${Math.round(t.roll || 0)}°`;
    }
    if (nodCountEl) {
      nodCountEl.textContent = `${t.nod_count || 0}`;
      if (t.is_nodding) nodCountEl.style.color = '#f97316';
      else nodCountEl.style.color = '#f1f5f9';
    }

    if (confidenceEl) confidenceEl.textContent = `${t.confidence || 0}%`;
    if (fpsBadge) fpsBadge.textContent = `${Math.round(t.fps || 0)} FPS`;

    // Night Mode Active HUD Indicator
    const hudNightChip = document.getElementById('hud-night-chip');
    const valLuminance = document.getElementById('val-luminance');
    if (hudNightChip) {
      if (t.night_mode) {
        hudNightChip.style.display = 'inline-flex';
        if (valLuminance) {
          valLuminance.textContent = `${Math.round(t.ambient_luminance || 0)}L`;
        }
      } else {
        hudNightChip.style.display = 'none';
      }
    }

    // Sunglasses Active HUD Indicator
    const hudSunglassesChip = document.getElementById('hud-sunglasses-chip');
    if (hudSunglassesChip) {
      if (t.sunglasses_detected || t.sunglasses_mode) {
        hudSunglassesChip.style.display = 'inline-flex';
      } else {
        hudSunglassesChip.style.display = 'none';
      }
    }

    // 3. Human-Readable Vital Status Pills
    const vitalEyesPill = document.getElementById('vital-status-eyes');
    const vitalEyesText = document.getElementById('vital-status-eyes-text');
    if (vitalEyesPill && vitalEyesText) {
      if (t.sunglasses_detected || t.sunglasses_mode) {
        vitalEyesText.textContent = '🕶️ Sunglasses Active';
        vitalEyesPill.className = 'vital-status-pill warning';
      } else if (t.is_closed) {
        vitalEyesText.textContent = (t.closure_duration_sec > 1.0) ? 'Microsleep!' : 'Eyes Closed';
        vitalEyesPill.className = 'vital-status-pill alert';
      } else if (t.perclos > 0.15) {
        vitalEyesText.textContent = 'Drowsy Flutter';
        vitalEyesPill.className = 'vital-status-pill warning';
      } else {
        vitalEyesText.textContent = 'Eyes Open';
        vitalEyesPill.className = 'vital-status-pill safe';
      }
    }

    const vitalBlinksPill = document.getElementById('vital-status-blinks');
    const vitalBlinksText = document.getElementById('vital-status-blinks-text');
    if (vitalBlinksPill && vitalBlinksText) {
      if (t.avg_blink_ms > 300) {
        vitalBlinksText.textContent = 'Sluggish Blinks';
        vitalBlinksPill.className = 'vital-status-pill warning';
      } else if (t.blink_rate_bpm > 28) {
        vitalBlinksText.textContent = 'Elevated Blinks';
        vitalBlinksPill.className = 'vital-status-pill warning';
      } else {
        vitalBlinksText.textContent = 'Alert Rate';
        vitalBlinksPill.className = 'vital-status-pill safe';
      }
    }

    const vitalPosturePill = document.getElementById('vital-status-posture');
    const vitalPostureText = document.getElementById('vital-status-posture-text');
    if (vitalPosturePill && vitalPostureText) {
      if (t.is_nodding) {
        vitalPostureText.textContent = 'Head Droop';
        vitalPosturePill.className = 'vital-status-pill alert';
      } else if (Math.abs(t.pitch || 0) > 15 || Math.abs(t.yaw || 0) > 25) {
        vitalPostureText.textContent = 'Gaze Drift';
        vitalPosturePill.className = 'vital-status-pill warning';
      } else {
        vitalPostureText.textContent = 'Focused Ahead';
        vitalPosturePill.className = 'vital-status-pill safe';
      }
    }

    const vitalYawnPill = document.getElementById('vital-status-yawn');
    const vitalYawnText = document.getElementById('vital-status-yawn-text');
    if (vitalYawnPill && vitalYawnText) {
      if (t.is_yawning) {
        vitalYawnText.textContent = 'Yawn Active';
        vitalYawnPill.className = 'vital-status-pill alert';
      } else if (t.yawn_count > 0) {
        vitalYawnText.textContent = `${t.yawn_count} Yawns`;
        vitalYawnPill.className = 'vital-status-pill warning';
      } else {
        vitalYawnText.textContent = 'Normal';
        vitalYawnPill.className = 'vital-status-pill safe';
      }
    }

    // 4. Risk Decomposition Breakdown
    const b = t.risk_breakdown || {};
    const eyeScore = b.perclos_score || 0;
    const blinkScore = b.blink_score || 0;
    const postureScore = b.head_posture_score || 0;
    const yawnScore = b.yawn_score || 0;

    if (barEyeRisk) barEyeRisk.style.width = `${eyeScore}%`;
    if (valEyeRisk) valEyeRisk.textContent = `${Math.round(eyeScore)}%`;

    if (barBlinkRisk) barBlinkRisk.style.width = `${blinkScore}%`;
    if (valBlinkRisk) valBlinkRisk.textContent = `${Math.round(blinkScore)}%`;

    if (barPostureRisk) barPostureRisk.style.width = `${postureScore}%`;
    if (valPostureRisk) valPostureRisk.textContent = `${Math.round(postureScore)}%`;

    if (barYawnRisk) barYawnRisk.style.width = `${yawnScore}%`;
    if (valYawnRisk) valYawnRisk.textContent = `${Math.round(yawnScore)}%`;

    // 4. Critical Alert Banner & Audio Alarms
    if (t.is_critical || stateCode === 'critical') {
      if (criticalBanner) {
        criticalBanner.classList.add('active');
        if (criticalReasonEl) {
          criticalReasonEl.textContent = t.override_reason || 'Severe driver fatigue detected! Pull over immediately!';
        }
      }
      window.audioManager.startCriticalAlarmLoop();
    } else {
      if (criticalBanner) criticalBanner.classList.remove('active');
      window.audioManager.stopAlarmLoop();

      if (stateCode === 'high_risk' && previousRiskCode !== 'high_risk') {
        window.audioManager.playHighRiskAlert();
      } else if (stateCode === 'caution' && previousRiskCode === 'safe') {
        window.audioManager.playCautionChime();
      }
    }
    previousRiskCode = stateCode;

    // 5. Update Scrolling Wave Chart
    riskChart.update(score, stateColor);

    // 6. Handle In-Progress Auto-Calibration Display
    if (t.calibration && t.calibration.state === 'CALIBRATING') {
      if (calibOverlay) calibOverlay.classList.add('active');
      if (calibCountdown) calibCountdown.textContent = Math.ceil(t.calibration.remaining_sec || 0);
    } else {
      if (calibOverlay) calibOverlay.classList.remove('active');
      if (t.calibration && t.calibration.baselines && calibQualityTag) {
        calibQualityTag.textContent = t.calibration.baselines.calibration_quality || 'Calibrated';
      }
    }
  }

  // Polling loop for direct server stream mode
  setInterval(() => {
    if (cameraController.mode === 'server' && cameraController.isCameraOn) {
      fetch('/api/telemetry')
        .then(res => res.json())
        .then(t => updateDashboard(t))
        .catch(() => {});
    }
  }, 75);

  /**
   * Event Listeners & Modal Controls
   */
  // Audio Mute Toggle
  if (audioMuteBtn) {
    audioMuteBtn.addEventListener('click', () => {
      const isMuted = window.audioManager.toggleMute();
      audioMuteBtn.classList.toggle('btn-primary', !isMuted);
      audioMuteBtn.classList.toggle('btn-outline', isMuted);
      if (audioMuteIcon) {
        audioMuteIcon.innerHTML = isMuted
          ? '<path d="M11 5L6 9H2v6h4l5 4V5z"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/>'
          : '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>';
      }
    });
  }

  // Camera Source Switcher
  const cameraSourceSelect = document.getElementById('camera-source-select');
  if (cameraSourceSelect) {
    cameraSourceSelect.addEventListener('change', (e) => {
      cameraController.setMode(e.target.value);
      const simControls = document.getElementById('sim-preset-controls');
      if (simControls) {
        simControls.style.display = (e.target.value === 'simulation') ? 'flex' : 'none';
      }
    });
  }

  // Simulation Preset Buttons
  document.querySelectorAll('.btn-sim-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.btn-sim-preset').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      cameraController.setSimPreset(btn.dataset.preset);
    });
  });

  // Start Calibration Trigger
  if (calibrateBtn) {
    calibrateBtn.addEventListener('click', () => {
      fetch('/api/calibration/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ duration_sec: 10 })
      })
        .then(res => res.json())
        .then(() => {
          window.audioManager.playCalibrationChime();
        });
    });
  }

  // Incident Log Modal
  const openEventsBtn = document.getElementById('btn-open-events');
  if (openEventsBtn && eventsModal) {
    openEventsBtn.addEventListener('click', () => {
      fetch('/api/events')
        .then(res => res.json())
        .then(events => {
          const tbody = document.getElementById('events-table-body');
          if (tbody) {
            if (events.length === 0) {
              tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:#64748b;">No safety incidents recorded yet. Clean driving session!</td></tr>';
            } else {
              tbody.innerHTML = events.map(e => `
                <tr>
                  <td><code>${e.timestamp.split(' ')[1] || e.timestamp}</code></td>
                  <td><span class="event-badge ${e.event_type}">${e.event_type}</span></td>
                  <td><strong>${e.risk_score}</strong></td>
                  <td>EAR: ${e.ear ? e.ear.toFixed(2) : '-'} | Pitch: ${e.pitch ? e.pitch.toFixed(0) : '-'}°</td>
                  <td>${e.description}</td>
                </tr>
              `).join('');
            }
          }
          eventsModal.classList.add('open');
        });
    });
  }

  // Modal Closers
  document.querySelectorAll('.modal-closer').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.modal-backdrop').forEach(m => m.classList.remove('open'));
    });
  });

  // Toggle HUD contours directly
  const toggleHudBtn = document.getElementById('btn-toggle-hud');
  let hudVisible = true;
  if (toggleHudBtn) {
    toggleHudBtn.addEventListener('click', () => {
      hudVisible = !hudVisible;
      fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ show_hud_overlay: hudVisible })
      });
      toggleHudBtn.classList.toggle('btn-primary', hudVisible);
      toggleHudBtn.classList.toggle('btn-outline', !hudVisible);
    });
  }

  // Toggle Night Vision Mode (Auto -> ON -> OFF -> Auto)
  const toggleNightBtn = document.getElementById('btn-toggle-night');
  const nightBtnText = document.getElementById('night-btn-text');
  const nightModes = ['auto', 'on', 'off'];
  let currentNightModeIdx = 0; // 'auto'
  if (toggleNightBtn) {
    toggleNightBtn.addEventListener('click', () => {
      currentNightModeIdx = (currentNightModeIdx + 1) % nightModes.length;
      const nextMode = nightModes[currentNightModeIdx];
      fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ night_mode: nextMode })
      })
        .then(res => res.json())
        .then(() => {
          toggleNightBtn.classList.remove('night-on', 'night-off');
          if (nextMode === 'on') {
            toggleNightBtn.classList.add('night-on');
            if (nightBtnText) nightBtnText.textContent = 'Night Vision: ON';
          } else if (nextMode === 'off') {
            toggleNightBtn.classList.add('night-off');
            if (nightBtnText) nightBtnText.textContent = 'Night Vision: OFF';
          } else {
            if (nightBtnText) nightBtnText.textContent = 'Night Vision: Auto';
          }
        })
        .catch(err => console.warn('Night mode toggle error:', err));
    });
  }

  // Toggle Sunglasses Force / Auto Mode
  const toggleSunglassesBtn = document.getElementById('btn-toggle-sunglasses');
  const sunglassesBtnText = document.getElementById('sunglasses-btn-text');
  let forceSunglasses = false;
  if (toggleSunglassesBtn) {
    toggleSunglassesBtn.addEventListener('click', () => {
      forceSunglasses = !forceSunglasses;
      fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force_sunglasses: forceSunglasses })
      })
        .then(res => res.json())
        .then(() => {
          toggleSunglassesBtn.classList.toggle('active', forceSunglasses);
          if (sunglassesBtnText) {
            sunglassesBtnText.textContent = forceSunglasses ? '🕶️ Goggles: Force ON' : '🕶️ Goggles: Auto';
          }
        })
        .catch(err => console.warn('Sunglasses toggle error:', err));
    });
  }
});
