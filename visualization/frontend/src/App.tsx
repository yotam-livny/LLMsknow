import { useEffect, useState } from 'react';
import { ModelSelector } from './components/ModelSelector';
import { DatasetSelector } from './components/DatasetSelector';
import { SampleBrowser } from './components/SampleBrowser';
import { CombinationDetails } from './components/CombinationDetails';
import { InferencePanel } from './components/InferencePanel';
import { VisualizationPanel } from './components/VisualizationPanel';
import { useStore } from './store/useStore';
import api from './api/client';
import './App.css';

function App() {
  const { isLoading, error, setError, inferenceRunning } = useStore();
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    console.log('App mounted, checking backend...');
    const checkBackend = async () => {
      try {
        console.log('Calling health endpoint...');
        await api.health();
        console.log('Backend is online');
        setBackendStatus('online');
      } catch (err) {
        console.error('Backend check failed:', err);
        setBackendStatus('offline');
      }
    };
    checkBackend();
  }, []);

  if (backendStatus === 'checking') {
    return (
      <div style={{ 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center', 
        justifyContent: 'center', 
        minHeight: '100vh',
        backgroundColor: '#0f1419',
        color: 'white'
      }}>
        <div style={{
          width: '40px',
          height: '40px',
          border: '3px solid #38444d',
          borderTopColor: '#1d9bf0',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite'
        }}></div>
        <p style={{ marginTop: '1rem' }}>Connecting to backend...</p>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (backendStatus === 'offline') {
    return (
      <div style={{ 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center', 
        justifyContent: 'center', 
        minHeight: '100vh',
        backgroundColor: '#0f1419',
        color: 'white',
        padding: '2rem',
        textAlign: 'center'
      }}>
        <h1>⚠️ Backend Offline</h1>
        <p>Could not connect to the API server at <code>http://localhost:8000</code></p>
        <div style={{ margin: '1.5rem 0', textAlign: 'left', background: '#1a2332', padding: '1.5rem', borderRadius: '8px' }}>
          <p>Please start the backend server:</p>
          <pre style={{ background: '#0f1419', padding: '0.75rem', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85rem', margin: '0.5rem 0' }}>cd visualization/backend && python -m api.app</pre>
          <p>Or use the run script:</p>
          <pre style={{ background: '#0f1419', padding: '0.75rem', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85rem', margin: '0.5rem 0' }}>./visualization/run.sh</pre>
        </div>
        <button 
          onClick={() => window.location.reload()}
          style={{ padding: '0.75rem 1.5rem', background: '#1d9bf0', border: 'none', borderRadius: '8px', color: 'white', fontSize: '1rem', cursor: 'pointer' }}
        >
          Retry Connection
        </button>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🔍 LLMsKnow Visualization</h1>
        <p>Explore LLM layer representations and probe predictions</p>
      </header>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      {(isLoading || inferenceRunning) && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          {inferenceRunning && <p>Running inference...</p>}
        </div>
      )}

      <main className="app-main">
        <div className="selection-panel">
          <section className="panel">
            <ModelSelector />
          </section>
          <section className="panel">
            <DatasetSelector />
          </section>
          <section className="panel">
            <CombinationDetails />
          </section>
          <section className="panel sample-panel">
            <SampleBrowser />
          </section>
        </div>
        
        <div className="work-panel">
          <section className="panel inference-section">
            <InferencePanel />
          </section>
          <section className="panel visualization-section">
            <VisualizationPanel />
          </section>
        </div>
      </main>

      <footer className="app-footer">
        <p>
          Based on <a href="https://arxiv.org/abs/2410.02707" target="_blank" rel="noopener noreferrer">
            "LLMs Know More Than They Show"
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
