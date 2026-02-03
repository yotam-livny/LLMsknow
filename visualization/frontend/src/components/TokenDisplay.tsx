import { TokenInfo } from '../api/client';
import { useStore } from '../store/useStore';

interface TokenDisplayProps {
  tokens: TokenInfo[];
}

export function TokenDisplay({ tokens }: TokenDisplayProps) {
  const { 
    selectedTokenIndex, 
    setSelectedTokenIndex,
    showInputTokens,
    inferenceResult
  } = useStore();

  const displayTokens = showInputTokens 
    ? tokens 
    : tokens.filter(t => !t.is_input);

  const inputCount = tokens.filter(t => t.is_input).length;
  const outputCount = tokens.length - inputCount;

  return (
    <div className="token-display">
      <div className="token-header">
        <h4>Tokens</h4>
        <div className="token-stats">
          <span className="stat input">Input: {inputCount}</span>
          <span className="stat output">Output: {outputCount}</span>
        </div>
      </div>

      <div className="token-container">
        {displayTokens.map((token, idx) => {
          const actualIndex = showInputTokens ? token.position : idx;
          const isSelected = selectedTokenIndex === token.position;
          
          return (
            <span
              key={token.position}
              className={`token ${token.is_input ? 'input' : 'output'} ${isSelected ? 'selected' : ''}`}
              onClick={() => setSelectedTokenIndex(token.position)}
              title={`Position: ${token.position}, ID: ${token.id}`}
            >
              {token.text.replace(/\n/g, '↵').replace(/ /g, '·') || '□'}
            </span>
          );
        })}
      </div>

      {selectedTokenIndex !== null && (
        <div className="selected-token-info">
          <span>Selected: Position {selectedTokenIndex}</span>
          <span className="token-text">
            "{tokens[selectedTokenIndex]?.text || ''}"
          </span>
        </div>
      )}

      {inferenceResult?.probe_predictions && Object.keys(inferenceResult.probe_predictions).length > 0 && (
        <div className="probe-predictions">
          <h4>Probe Predictions</h4>
          {Object.entries(inferenceResult.probe_predictions).map(([layer, pred]) => (
            <div key={layer} className={`prediction ${pred.prediction === 0 ? 'correct' : 'incorrect'}`}>
              <span>Layer {layer}:</span>
              <span className="confidence">
                {pred.prediction === 0 ? '✓ Correct' : '✗ Incorrect'} 
                ({(pred.confidence * 100).toFixed(1)}%)
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
