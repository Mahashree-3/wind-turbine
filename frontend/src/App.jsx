import { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import './App.css'

const API_BASE = 'http://localhost:8000'

const severityColors = {
  Normal: '#22c55e',
  Minor: '#eab308',
  Moderate: '#f97316',
  Severe: '#ef4444',
}

function App() {
  const [turbine, setTurbine] = useState('Turbine 1')
  const [sensors, setSensors] = useState(null)
  const [diagnosis, setDiagnosis] = useState(null)
  const [trend, setTrend] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_BASE}/api/sensors`),
      axios.get(`${API_BASE}/api/diagnosis`),
      axios.get(`${API_BASE}/api/trend`),
    ])
      .then(([sensorsRes, diagnosisRes, trendRes]) => {
        setSensors(sensorsRes.data)
        setDiagnosis(diagnosisRes.data)
        setTrend(trendRes.data)
        setLoading(false)
      })
      .catch((err) => {
        console.error('Failed to fetch data:', err)
        setLoading(false)
      })
  }, [turbine])

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', color: '#22d3ee', fontSize: '20px' }}>
        Loading dashboard...
      </div>
    )
  }

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', padding: '2rem', fontFamily: 'sans-serif' }}>
      <div style={{ maxWidth: '1000px', margin: '0 auto' }}>

        <header style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{ fontSize: '48px' }}>🌬️</div>
          <h1 style={{ color: '#22d3ee', margin: '0.5rem 0' }}>Wind Turbine Bearing Health Monitor</h1>
          <p style={{ color: '#94a3b8' }}>Multi-sensor fault diagnosis and predictive maintenance</p>

          <select
            value={turbine}
            onChange={(e) => setTurbine(e.target.value)}
            style={{ marginTop: '1rem', padding: '0.5rem 1rem', borderRadius: '8px', background: '#1e293b', color: '#e2e8f0', border: '1px solid #334155' }}
          >
            <option>Turbine 1</option>
            <option>Turbine 2</option>
            <option>Turbine 3</option>
          </select>
        </header>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
          {sensors && Object.entries(sensors).map(([key, value]) => (
            <div key={key} style={{ background: '#1e293b', borderRadius: '12px', padding: '1rem', textAlign: 'center' }}>
              <p style={{ color: '#94a3b8', fontSize: '13px', margin: '0 0 4px', textTransform: 'capitalize' }}>{key}</p>
              <p style={{ fontSize: '20px', fontWeight: 'bold', margin: 0 }}>{String(value)}</p>
            </div>
          ))}
        </div>

        {diagnosis && (
          <div style={{ background: '#1e293b', borderRadius: '12px', padding: '1.5rem', marginBottom: '2rem' }}>
            <h2 style={{ marginTop: 0, color: '#22d3ee' }}>Diagnosis Results</h2>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <span>Fault Type: <strong>{diagnosis.fault_type}</strong></span>
              <span style={{
                background: severityColors[diagnosis.severity] || '#64748b',
                padding: '4px 12px', borderRadius: '999px', fontSize: '13px', fontWeight: 'bold'
              }}>{diagnosis.severity}</span>
            </div>
            <p>Confidence: {(diagnosis.confidence * 100).toFixed(0)}%</p>
            <div style={{ background: '#334155', borderRadius: '999px', height: '10px', overflow: 'hidden', marginBottom: '1rem' }}>
              <div style={{ width: `${diagnosis.confidence * 100}%`, background: '#22d3ee', height: '100%' }} />
            </div>
            <p style={{ fontSize: '18px' }}>Estimated Remaining Useful Life: <strong>{diagnosis.rul_days} days</strong></p>
          </div>
        )}

        {trend.length > 0 && (
          <div style={{ background: '#1e293b', borderRadius: '12px', padding: '1.5rem', marginBottom: '2rem' }}>
            <h2 style={{ marginTop: 0, color: '#22d3ee' }}>30-Day Bearing Health Trend</h2>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="day" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ background: '#1e293b', border: 'none' }} />
                <Line type="monotone" dataKey="score" stroke="#22d3ee" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {diagnosis && (
          <div style={{ background: '#7c2d12', border: '1px solid #f97316', borderRadius: '12px', padding: '1rem', marginBottom: '2rem' }}>
            ⚠️ Maintenance recommended within {diagnosis.rul_days} days
          </div>
        )}

        <footer style={{ textAlign: 'center', color: '#64748b', fontSize: '13px' }}>
          Prototype using sample data — live model integration in progress
        </footer>
      </div>
    </div>
  )
}

export default App