/**
 * Predictor - ML model interaction page.
 * Purpose: Form to send features to ML API and display predictions.
 * Modify: Update description, fields config, and prediction history UI.
 */
import { useState, useEffect, useRef } from 'react'
import { useMLPredict } from '../hooks/useMLPredict'
import MLPredictor from '../components/predictor/MLPredictor'

// TODO: Update this description to match what your model predicts.
const MODEL_DESCRIPTION =
  'This model predicts a target value based on the input features. Update MODEL_DESCRIPTION in Predictor.jsx to describe your model.'

// TODO: Configure fields to match your model's expected input features.
const PREDICTOR_FIELDS = [
  { key: 'feature1', label: 'Feature 1', type: 'number', min: 0, max: 100 },
  { key: 'feature2', label: 'Feature 2', type: 'number', min: 0, max: 100 },
  { key: 'feature3', label: 'Feature 3', type: 'number', min: 0, max: 100 },
]

const MAX_HISTORY = 5

export default function Predictor() {
  const { predict, result, loading, error, reset } = useMLPredict()
  const [history, setHistory] = useState([])
  const lastInputRef = useRef(null)

  const handlePredict = async (inputData) => {
    lastInputRef.current = inputData
    await predict(inputData)
  }

  const handleReset = () => {
    reset()
  }

  // Add to history when we get a successful result
  useEffect(() => {
    if (!result || loading) return
    const input = lastInputRef.current
    if (input == null) return
    setHistory((prev) =>
      [{ input, prediction: result.prediction, confidence: result.confidence, label: result.label }, ...prev].slice(
        0,
        MAX_HISTORY
      )
    )
    lastInputRef.current = null
  }, [result, loading])

  return (
    <div>
      <header style={{ marginBottom: '1.5rem' }}>
        <h1>ML Predictor</h1>
        <p style={{ color: 'var(--color-text-muted)', marginTop: '0.5rem' }}>{MODEL_DESCRIPTION}</p>
      </header>

      <div style={{ maxWidth: 500, marginBottom: '2rem' }}>
        <MLPredictor
          fields={PREDICTOR_FIELDS}
          onPredict={handlePredict}
          result={result}
          loading={loading}
          error={error}
          onReset={handleReset}
        />
      </div>

      <section>
        <h2>Last {MAX_HISTORY} Predictions</h2>
        {history.length === 0 ? (
          <p style={{ color: 'var(--color-text-muted)' }}>No predictions yet this session.</p>
        ) : (
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {history.map((h, i) => (
              <li
                key={i}
                style={{
                  padding: '0.75rem',
                  background: 'var(--color-surface)',
                  borderRadius: 'var(--radius)',
                  marginBottom: '0.5rem',
                  boxShadow: 'var(--shadow)',
                }}
              >
                Input: {JSON.stringify(h.input)} → Prediction: {h.prediction ?? h.label ?? '—'}
                {h.confidence != null && ` (${(h.confidence * 100).toFixed(1)}% confidence)`}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}
