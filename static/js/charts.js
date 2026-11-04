/**
 * DriveSafe - Live Risk Telemetry Chart Manager
 * Renders real-time 60-second scrolling drowsiness risk wave using Chart.js.
 */
class RiskChartManager {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.chart = null;
    this.maxPoints = 60;
    this.labels = Array.from({ length: this.maxPoints }, (_, i) => `${this.maxPoints - i}s`);
    this.data = Array(this.maxPoints).fill(0);
    this._initChart();
  }

  _initChart() {
    if (!this.canvas) return;
    const ctx = this.canvas.getContext('2d');

    // Create glowing vertical gradient fill
    const gradient = ctx.createLinearGradient(0, 0, 0, 160);
    gradient.addColorStop(0, 'rgba(6, 182, 212, 0.45)');
    gradient.addColorStop(0.5, 'rgba(6, 182, 212, 0.15)');
    gradient.addColorStop(1, 'rgba(6, 182, 212, 0.0)');

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.labels,
        datasets: [{
          label: 'Drowsiness Risk',
          data: this.data,
          borderColor: '#06b6d4',
          borderWidth: 2.2,
          backgroundColor: gradient,
          fill: true,
          tension: 0.35,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointBackgroundColor: '#fff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 0 // Disable animations for smooth 30fps streaming
        },
        interaction: {
          intersect: false,
          mode: 'index'
        },
        scales: {
          x: {
            display: true,
            grid: {
              color: 'rgba(255, 255, 255, 0.04)',
              drawBorder: false
            },
            ticks: {
              color: '#64748b',
              font: { family: "'JetBrains Mono', monospace", size: 10 },
              maxTicksLimit: 6
            }
          },
          y: {
            min: 0,
            max: 100,
            grid: {
              color: (context) => {
                if (context.tick.value === 30) return 'rgba(16, 185, 129, 0.25)';
                if (context.tick.value === 60) return 'rgba(245, 158, 11, 0.25)';
                if (context.tick.value === 80) return 'rgba(239, 68, 68, 0.35)';
                return 'rgba(255, 255, 255, 0.04)';
              },
              drawBorder: false
            },
            ticks: {
              stepSize: 20,
              color: '#64748b',
              font: { family: "'JetBrains Mono', monospace", size: 10 }
            }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(15, 22, 36, 0.9)',
            titleFont: { family: "'Outfit', sans-serif", size: 12 },
            bodyFont: { family: "'JetBrains Mono', monospace", size: 12 },
            borderColor: 'rgba(255, 255, 255, 0.15)',
            borderWidth: 1,
            displayColors: false,
            callbacks: {
              label: (context) => `Risk Score: ${context.parsed.y.toFixed(1)} / 100`
            }
          }
        }
      }
    });
  }

  update(currentScore, riskColor = '#06b6d4') {
    if (!this.chart) return;

    this.data.shift();
    this.data.push(currentScore);

    const dataset = this.chart.data.datasets[0];
    dataset.borderColor = riskColor;

    // Adjust fill color dynamically to match risk severity
    const ctx = this.canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 160);
    gradient.addColorStop(0, `${riskColor}55`);
    gradient.addColorStop(1, `${riskColor}00`);
    dataset.backgroundColor = gradient;

    this.chart.update('none');
  }

  reset() {
    if (!this.chart) return;
    this.data.fill(0);
    this.chart.update('none');
  }
}
