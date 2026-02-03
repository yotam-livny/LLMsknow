import { useEffect, useState } from 'react';
import { useStore } from '../store/useStore';
import { apiClient } from '../api/client';

interface LogitLensToken {
  token_id: number;
  token_text: string;
  probability: number;
}

interface LogitLensLayerResult {
  layer: number;
  top_tokens: LogitLensToken[];
  target_token_rank: number | null;
  target_token_prob: number | null;
}

interface LogitLensData {
  token_position: number;
  token_text: string;
  token_id: number;
  prediction_position: number;
  prediction_token_text: string;
  layers: LogitLensLayerResult[];
}

export function LogitLensView() {
  const { inferenceResult, selectedTokenIndex, setSelectedTokenIndex } = useStore();
  const [logitLensData, setLogitLensData] = useState<LogitLensData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [topK, setTopK] = useState(5);

  // Use global selectedTokenIndex, default to first output token if not set
  const selectedPosition = selectedTokenIndex !== null ? selectedTokenIndex : 
    inferenceResult?.tokens.find(t => !t.is_input && t.position > 0)?.position ?? null;

  // Fetch logit lens data when position changes
  useEffect(() => {
    // Need position > 0 for logit lens (we analyze N-1 to predict N)
    if (selectedPosition === null || selectedPosition === 0 || !inferenceResult?.has_layer_data) {
      setLogitLensData(null);
      return;
    }

    const fetchLogitLens = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await apiClient.post('/inference/logit-lens', {
          token_position: selectedPosition,
          top_k: topK
        });
        setLogitLensData(response.data);
      } catch (err: any) {
        console.error('Failed to fetch logit lens:', err);
        setError(err.response?.data?.detail || 'Failed to compute logit lens');
      } finally {
        setLoading(false);
      }
    };

    fetchLogitLens();
  }, [selectedPosition, topK, inferenceResult?.has_layer_data]);

  if (!inferenceResult) {
    return (
      <div className="logit-lens-view empty">
        <p>Run inference to see logit lens analysis</p>
      </div>
    );
  }

  if (!inferenceResult.has_layer_data) {
    return (
      <div className="logit-lens-view empty">
        <p>Layer data not available</p>
        <p className="hint">Run inference with layer extraction enabled</p>
      </div>
    );
  }

  // Filter output tokens (excluding position 0 since we need N-1 for prediction)
  const outputTokens = inferenceResult.tokens.filter(t => !t.is_input && t.position > 0);

  return (
    <div className="logit-lens-view">
      <div className="view-header">
        <h3>🔬 Logit Lens</h3>
        <p className="subtitle">How token predictions evolve through layers</p>
      </div>

      {/* Controls */}
      <div className="logit-lens-controls">
        <div className="control-group">
          <label>Selected token:</label>
          <span className="selected-token-display">
            {selectedPosition !== null && inferenceResult.tokens[selectedPosition] ? (
              <>#{selectedPosition}: "{inferenceResult.tokens[selectedPosition].text.replace(/\n/g, '↵').replace(/ /g, '·').slice(0, 20)}"</>
            ) : (
              'Click a token above to analyze'
            )}
          </span>
        </div>
        <div className="control-group">
          <label>Top-K:</label>
          <select value={topK} onChange={(e) => setTopK(parseInt(e.target.value))}>
            <option value={3}>3</option>
            <option value={5}>5</option>
            <option value={10}>10</option>
          </select>
        </div>
      </div>
      
      {selectedPosition === 0 && (
        <div className="warning-state">
          <p>⚠️ Cannot analyze position 0 - select an output token (position &gt; 0)</p>
        </div>
      )}

      {loading && (
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Computing logit lens...</p>
        </div>
      )}

      {error && (
        <div className="error-state">
          <p>⚠️ {error}</p>
        </div>
      )}

      {logitLensData && !loading && (
        <div className="logit-lens-results">
          <div className="target-token-info">
            <div className="info-row">
              <span className="label">Predicting token:</span>
              <span className="token-badge target">
                "{logitLensData.token_text.replace(/\n/g, '↵').replace(/ /g, '·') || '□'}"
              </span>
              <span className="position-info">(position {logitLensData.token_position})</span>
            </div>
            <div className="info-row">
              <span className="label">Using hidden state at:</span>
              <span className="token-badge source">
                "{logitLensData.prediction_token_text.replace(/\n/g, '↵').replace(/ /g, '·') || '□'}"
              </span>
              <span className="position-info">(position {logitLensData.prediction_position})</span>
            </div>
            <p className="explanation-note">
              💡 To predict token N, the model uses the hidden state at position N-1
            </p>
          </div>

          <div className="layers-grid">
            {logitLensData.layers.map((layerResult) => (
              <div key={layerResult.layer} className="layer-card">
                <div className="layer-header">
                  <span className="layer-num">Layer {layerResult.layer}</span>
                  {layerResult.target_token_rank && (
                    <span className={`target-rank ${layerResult.target_token_rank <= 3 ? 'high' : layerResult.target_token_rank <= 10 ? 'medium' : 'low'}`}>
                      Target: #{layerResult.target_token_rank}
                    </span>
                  )}
                </div>
                
                <div className="predictions-list">
                  {layerResult.top_tokens.map((token, idx) => {
                    const isTarget = token.token_id === logitLensData.token_id;
                    return (
                      <div 
                        key={idx} 
                        className={`prediction ${isTarget ? 'is-target' : ''}`}
                      >
                        <div 
                          className="prob-bar"
                          style={{ width: `${token.probability * 100}%` }}
                        />
                        <span className="rank">#{idx + 1}</span>
                        <span className="token-text">
                          "{token.token_text.replace(/\n/g, '↵').replace(/ /g, '·') || '□'}"
                        </span>
                        <span className="prob">{(token.probability * 100).toFixed(1)}%</span>
                      </div>
                    );
                  })}
                </div>

                {layerResult.target_token_prob !== null && layerResult.target_token_rank && layerResult.target_token_rank > topK && (
                  <div className="target-not-in-top">
                    Target "{logitLensData.token_text}" at rank #{layerResult.target_token_rank} 
                    ({(layerResult.target_token_prob * 100).toFixed(2)}%)
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="logit-lens-explanation">
            <h4>How to read</h4>
            <ul>
              <li><strong>Each card</strong> = What the model would predict at that layer</li>
              <li><strong>Target rank</strong> = Position of the actual token in the layer's predictions</li>
              <li><strong>Green highlight</strong> = The actual generated token appears in top predictions</li>
              <li><strong>Evolution</strong> = Watch how the target token rises through layers</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
