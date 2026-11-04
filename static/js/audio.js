/**
 * DriveSafe - Original Web Audio API Alert Synthesizer
 * Generates pure procedural acoustic alarms, urgent ADAS safety sirens, and harmonic chimes.
 * Operates without external MP3 dependencies for zero latency and instant response.
 */
class AudioManager {
  constructor() {
    this.ctx = null;
    this.isMuted = false;
    this.volume = 0.85;
    this.lastAlarmTime = 0;
    this.alarmInterval = null;
    this.isAlarmLooping = false;
  }

  _initContext() {
    if (!this.ctx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioContext();
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  toggleMute() {
    this.isMuted = !this.isMuted;
    if (this.isMuted) {
      this.stopAlarmLoop();
    }
    return this.isMuted;
  }

  /**
   * Harmonious confirmation chime for auto-calibration completion
   */
  playCalibrationChime() {
    if (this.isMuted) return;
    this._initContext();

    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(523.25, now); // C5
    osc.frequency.exponentialRampToValueAtTime(783.99, now + 0.16); // G5
    osc.frequency.exponentialRampToValueAtTime(1046.50, now + 0.32); // C6

    gain.gain.setValueAtTime(0.01, now);
    gain.gain.exponentialRampToValueAtTime(0.3 * this.volume, now + 0.08);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);

    osc.connect(gain);
    gain.connect(this.ctx.destination);

    osc.start(now);
    osc.stop(now + 0.45);
  }

  /**
   * Subtle advisory chime for mild caution state
   */
  playCautionChime() {
    if (this.isMuted) return;
    this._initContext();

    const now = this.ctx.currentTime;
    if (now - this.lastAlarmTime < 2.5) return;
    this.lastAlarmTime = now;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'triangle';
    osc.frequency.setValueAtTime(659.25, now); // E5
    osc.frequency.setValueAtTime(523.25, now + 0.14); // C5

    gain.gain.setValueAtTime(0.25 * this.volume, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.42);

    osc.connect(gain);
    gain.connect(this.ctx.destination);

    osc.start(now);
    osc.stop(now + 0.42);
  }

  /**
   * Dual-pulse warning chime for high risk state
   */
  playHighRiskAlert() {
    if (this.isMuted) return;
    this._initContext();

    const now = this.ctx.currentTime;
    if (now - this.lastAlarmTime < 1.8) return;
    this.lastAlarmTime = now;

    // Fast double pulse
    [0, 0.14].forEach((delay) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(880, now + delay); // A5

      gain.gain.setValueAtTime(0.35 * this.volume, now + delay);
      gain.gain.exponentialRampToValueAtTime(0.001, now + delay + 0.11);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start(now + delay);
      osc.stop(now + delay + 0.11);
    });
  }

  /**
   * Urgent automotive ADAS emergency dual-tone siren
   * Procedural dual-oscillator with frequency modulation to immediately awaken the driver
   */
  playCriticalEmergencyTone() {
    if (this.isMuted) return;
    this._initContext();

    const now = this.ctx.currentTime;
    const osc1 = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc1.type = 'sawtooth';
    osc2.type = 'square';

    // Urgent rapid ADAS warble (1100Hz -> 1500Hz -> 900Hz)
    osc1.frequency.setValueAtTime(1100, now);
    osc1.frequency.linearRampToValueAtTime(1500, now + 0.09);
    osc1.frequency.linearRampToValueAtTime(900, now + 0.20);

    osc2.frequency.setValueAtTime(550, now);
    osc2.frequency.linearRampToValueAtTime(750, now + 0.09);
    osc2.frequency.linearRampToValueAtTime(450, now + 0.20);

    gain.gain.setValueAtTime(0.48 * this.volume, now);
    gain.gain.exponentialRampToValueAtTime(0.01, now + 0.22);

    osc1.connect(gain);
    osc2.connect(gain);
    gain.connect(this.ctx.destination);

    osc1.start(now);
    osc2.start(now);
    osc1.stop(now + 0.22);
    osc2.stop(now + 0.22);
  }

  startCriticalAlarmLoop() {
    if (this.isAlarmLooping || this.isMuted) return;
    this.isAlarmLooping = true;
    this._initContext();

    this.playCriticalEmergencyTone();
    this.alarmInterval = setInterval(() => {
      if (this.isAlarmLooping && !this.isMuted) {
        this.playCriticalEmergencyTone();
      }
    }, 260); // Rapid urgent pulse cycle
  }

  stopAlarmLoop() {
    this.isAlarmLooping = false;
    if (this.alarmInterval) {
      clearInterval(this.alarmInterval);
      this.alarmInterval = null;
    }
  }
}

// Global singleton instance
window.audioManager = new AudioManager();

 