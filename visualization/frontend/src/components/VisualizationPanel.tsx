import { useState } from 'react';
import { useStore } from '../store/useStore';
import { AttentionView } from './AttentionView';
import { AlternativesView } from './AlternativesView';
import { CorrectnessEvolutionView } from './CorrectnessEvolutionView';

type ViewTab = 'attention' | 'alternatives' | 'correctness';

export function VisualizationPanel() {
  const [activeTab, setActiveTab] = useState<ViewTab>('attention');
  const { 
    inferenceResult, 
    attentionData,
  } = useStore();

  const hasData = inferenceResult && attentionData;
  const hasAlternatives = inferenceResult?.token_alternatives && inferenceResult.token_alternatives.length > 0;
  const hasLayerData = inferenceResult?.has_layer_data;

  return (
    <div className="visualization-panel">
      <div className="viz-header">
        <h3>📊 Visualization</h3>
        
        {hasData && (
          <div className="viz-tabs">
            <button 
              className={activeTab === 'attention' ? 'active' : ''}
              onClick={() => setActiveTab('attention')}
            >
              🔍 Attention
            </button>
            <button 
              className={activeTab === 'alternatives' ? 'active' : ''}
              onClick={() => setActiveTab('alternatives')}
              disabled={!hasAlternatives}
              title={hasAlternatives ? 'View token alternatives' : 'Restart backend to enable'}
            >
              🎯 Alternatives
            </button>
            <button 
              className={activeTab === 'correctness' ? 'active' : ''}
              onClick={() => setActiveTab('correctness')}
              disabled={!hasLayerData}
              title={hasLayerData ? 'View correctness evolution across layers' : 'Run inference with layer extraction'}
            >
              📈 Correctness
            </button>
          </div>
        )}
      </div>

      <div className="viz-content">
        {!hasData ? (
          <div className="empty-state">
            <div className="empty-icon">🔍</div>
            <h4>No visualization data</h4>
            <p>Select a sample and run inference to see attention patterns</p>
            <div className="steps">
              <div className="step">1. Select a model</div>
              <div className="step">2. Select a dataset</div>
              <div className="step">3. Choose a sample</div>
              <div className="step">4. Click "Run Inference"</div>
            </div>
          </div>
        ) : (
          <>
            {activeTab === 'attention' && <AttentionView />}
            {activeTab === 'alternatives' && <AlternativesView />}
            {activeTab === 'correctness' && <CorrectnessEvolutionView />}
          </>
        )}
      </div>
    </div>
  );
}
