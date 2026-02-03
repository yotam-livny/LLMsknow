import { useEffect, useState } from 'react';
import { ModelSelector } from './components/ModelSelector';
import { DatasetSelector } from './components/DatasetSelector';
import { SampleBrowser } from './components/SampleBrowser';
import { CombinationDetails } from './components/CombinationDetails';
import { useStore } from './store/useStore';
import api from './api/client';
import './App.css';

function App() {
  const { isLoading, error, setError } = useStore();
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    const checkBackend = async () => {
      try {
        await api.health();
        setBackendStatus('online');
      } catch {
        setBackendStatus('offline');
      }
    };
    checkBackend();
  }, []);

  if (backendStatus === 'checking') {
    return (
      <div className="app loading-screen">
        <div className="spinner"></div>
        <p>Connecting to backend...</p>
      </div>
    );
  }

  if (backendStatus === 'offline') {
    return (
      <div className="app error-screen">
        <h1>⚠️ Backend Offline</h1>
        <p>Could not connect to the API server at <code>http://localhost:8000</code></p>
        <div className="instructions">
          <p>Please start the backend server:</p>
          <pre>cd visualization/backend && python -m api.app</pre>
          <p>Or use the run script:</p>
          <pre>./visualization/run.sh</pre>
        </div>
        <button onClick={() => window.location.reload()}>Retry Connection</button>
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

      {isLoading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
        </div>
      )}

      <main className="app-main">
        <div className="left-panel">
          <section className="panel">
            <ModelSelector />
          </section>
          <section className="panel">
            <DatasetSelector />
          </section>
        </div>
        
        <div className="right-panel">
          <section className="panel">
            <CombinationDetails />
          </section>
          <section className="panel">
            <SampleBrowser />
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
