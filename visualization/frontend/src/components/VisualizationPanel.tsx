import { useStore } from '../store/useStore';
import { LayerView } from './LayerView';
import { AttentionView } from './AttentionView';

export function VisualizationPanel() {
  const { 
    inferenceResult, 
    layerData, 
    attentionData,
    viewMode,
    setViewMode,
    showInputTokens,
    setShowInputTokens
  } = useStore();

  const hasData = inferenceResult && (layerData || attentionData);

  return (
    <div className="visualization-panel">
      <div className="viz-header">
        <h3>📊 Visualization</h3>
        
        {hasData && (
          <div className="viz-controls">
            <div className="view-mode-selector">
              <button 
                className={viewMode === 'layer' ? 'active' : ''}
                onClick={() => setViewMode('layer')}
                disabled={!layerData}
              >
                Layer Flow
              </button>
              <button 
                className={viewMode === 'attention' ? 'active' : ''}
                onClick={() => setViewMode('attention')}
                disabled={!attentionData}
              >
                Attention Flow
              </button>
            </div>

            <label className="toggle-label">
              <input 
                type="checkbox" 
                checked={showInputTokens}
                onChange={(e) => setShowInputTokens(e.target.checked)}
              />
              Show input tokens
            </label>
          </div>
        )}
      </div>

      <div className="viz-content">
        {!hasData ? (
          <div className="empty-state">
            <div className="empty-icon">📈</div>
            <h4>No visualization data</h4>
            <p>Select a sample and run inference to see layer representations and attention patterns</p>
            <div className="steps">
              <div className="step">1. Select a model</div>
              <div className="step">2. Select a dataset</div>
              <div className="step">3. Choose a sample</div>
              <div className="step">4. Click "Run Inference"</div>
            </div>
          </div>
        ) : (
          <div className="viz-views">
            {viewMode === 'layer' && layerData && <LayerView />}
            {viewMode === 'attention' && attentionData && <AttentionView />}
          </div>
        )}
      </div>
    </div>
  );
}
