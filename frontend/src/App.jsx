import "./App.css";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from "recharts";
import {
  Activity,
  Volume2,
  Thermometer,
  Zap
} from "lucide-react";
function App() {
  const trendData = [
  { day: 1, score: 97.6 },
  { day: 2, score: 95.7 },
  { day: 3, score: 94.8 },
  { day: 4, score: 93.5 },
  { day: 5, score: 92.1 },
  { day: 6, score: 90.8 },
  { day: 7, score: 89.6 },
  { day: 8, score: 88.4 },
  { day: 9, score: 87.2 },
  { day: 10, score: 86.5 },
  { day: 11, score: 85.1 },
  { day: 12, score: 84.3 },
  { day: 13, score: 83.6 },
  { day: 14, score: 82.4 },
  { day: 15, score: 81.7 },
  { day: 16, score: 80.5 },
  { day: 17, score: 79.8 },
  { day: 18, score: 78.6 },
  { day: 19, score: 77.9 },
  { day: 20, score: 76.8 },
  { day: 21, score: 75.5 },
  { day: 22, score: 74.9 },
  { day: 23, score: 73.8 },
  { day: 24, score: 72.6 },
  { day: 25, score: 71.4 },
  { day: 26, score: 70.8 },
  { day: 27, score: 69.5 },
  { day: 28, score: 68.3 },
  { day: 29, score: 67.2 },
  { day: 30, score: 66.1 }
];
  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>🌬️ Wind Turbine Bearing Health Monitor</h1>
          <p>Multi-sensor fault diagnosis and predictive maintenance</p>
        </div>

        <select>
          <option>Turbine 1</option>
          <option>Turbine 2</option>
          <option>Turbine 3</option>
        </select>
      </header>

      <main>
        <section className="sensor-grid">
          <div className="card sensor-card">
  <div className="sensor-icon">
    <Volume2 size={24} />
  </div>

  <div>
    <h3>Acoustic</h3>
    <h2>65.5</h2>
  </div>
</div>

          <div className="card sensor-card">
  <div className="sensor-icon">
    <Activity size={24} />
  </div>

  <div>
    <h3>Vibration</h3>
    <h2>0.276</h2>
  </div>
</div>

          <div className="card sensor-card">
  <div className="sensor-icon">
    <Thermometer size={24} />
  </div>

  <div>
    <h3>Temperature</h3>
    <h2>55.4 °C</h2>
  </div>
</div>

          <div className="card sensor-card">
  <div className="sensor-icon">
    <Zap size={24} />
  </div>

  <div>
    <h3>Current</h3>
    <h2>Coming soon</h2>
  </div>
</div>
        </section>

        <section className="card diagnosis">
  <h2>Diagnosis Results</h2>

  <div className="diagnosis-grid">
    <div className="diagnosis-item">
      <span>Fault Type</span>
      <strong>Outer Race</strong>
    </div>

    <div className="diagnosis-item">
      <span>Severity</span>
      <strong className="severity">Moderate</strong>
    </div>

    <div className="diagnosis-item">
      <span>Confidence</span>
      <strong>83.35%</strong>
    </div>

    <div className="diagnosis-item">
      <span>Remaining Useful Life</span>
      <strong>30 Days</strong>
    </div>
  </div>

  <div className="progress">
  <div className="progress-bar" style={{ width: "83.35%" }}></div>
</div>
  <div className="diagnosis-actions">
    <button className="btn-download">⬇ Download Report</button>
  </div>
</section>

        <section className="card">
          <h2>30-Day Health Trend</h2>
          <div className="chart-placeholder">
  <ResponsiveContainer width="100%" height="100%">
    <LineChart data={trendData}>
      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />

      <XAxis
  dataKey="day"
  stroke="#94a3b8"
  tick={{ fill: "#94a3b8", fontSize: 12 }}
/>

<YAxis
  stroke="#94a3b8"
  tick={{ fill: "#94a3b8", fontSize: 12 }}
/>

      <Tooltip
  contentStyle={{
    backgroundColor: "#1e293b",
    border: "1px solid #334155",
    borderRadius: "8px",
    color: "#e2e8f0"
  }}
  labelStyle={{
    color: "#94a3b8"
  }}
/>

      <Line
        type="monotone"
        dataKey="score"
        stroke="#22d3ee"
        strokeWidth={3}
        dot={false}
      />
    </LineChart>
  </ResponsiveContainer>
</div>
        </section>

        <section className="alert">
          ⚠️ Maintenance recommended within 30 days
        </section>
      </main>

      <footer>
        Live predictions from trained model — KAIST/MaFaulDa bearing fault dataset
      </footer>
    </div>
  );
}

export default App;