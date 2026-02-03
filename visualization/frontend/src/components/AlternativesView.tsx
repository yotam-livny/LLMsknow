import { useStore } from '../store/useStore';

export function AlternativesView() {
  const { 
    inferenceResult,
    selectedTokenIndex,
    setSelectedTokenIndex
  } = useStore();

  if (!inferenceResult) {
    return (
      <div className="alternatives-view empty">
        <p>Run inference to see token alternatives</p>
      </div>
    );
  }

  const alternatives = inferenceResult.token_alternatives;
  const inputCount = inferenceResult.input_token_count;
  const outputTokens = inferenceResult.tokens.filter(t => !t.is_input);

  if (!alternatives || alternatives.length === 0) {
    return (
      <div className="alternatives-view empty">
        <p>No token alternatives available</p>
        <p className="hint">Restart the backend to enable this feature</p>
      </div>
    );
  }

  return (
    <div className="alternatives-view">
      <div className="view-header">
        <h3>🎯 Token Alternatives</h3>
        <p className="subtitle">What tokens nearly got selected at each generation step</p>
      </div>

      <div className="alternatives-grid">
        {outputTokens.map((token, outputIdx) => {
          const stepAlternatives = alternatives[outputIdx];
          if (!stepAlternatives) return null;

          const isSelected = selectedTokenIndex === token.position;
          const chosenToken = stepAlternatives[0]; // First one is the chosen token
          const otherOptions = stepAlternatives.slice(1); // Rest are alternatives

          return (
            <div 
              key={token.position} 
              className={`token-step ${isSelected ? 'selected' : ''}`}
              onClick={() => setSelectedTokenIndex(token.position)}
            >
              <div className="step-header">
                <span className="step-num">#{outputIdx + 1}</span>
                <span className="chosen-token">
                  "{token.text.replace(/\n/g, '↵').replace(/ /g, '·') || '□'}"
                </span>
                <span className="chosen-prob">{(chosenToken?.probability * 100).toFixed(1)}%</span>
              </div>
              
              <div className="alternatives-list">
                {otherOptions.map((alt, altIdx) => (
                  <div key={altIdx} className="alternative">
                    <div 
                      className="alt-bar" 
                      style={{ width: `${alt.probability * 100}%` }}
                    />
                    <span className="alt-text">
                      "{alt.token_text.replace(/\n/g, '↵').replace(/ /g, '·') || '□'}"
                    </span>
                    <span className="alt-prob">{(alt.probability * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>

              {/* Confidence indicator */}
              <div className="confidence-meter">
                <div 
                  className="confidence-fill"
                  style={{ 
                    width: `${(chosenToken?.probability || 0) * 100}%`,
                    backgroundColor: chosenToken?.probability > 0.8 ? '#22c55e' : 
                                    chosenToken?.probability > 0.5 ? '#fbbf24' : '#ef4444'
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="alternatives-explanation">
        <h4>How to read</h4>
        <ul>
          <li><strong>Top row</strong> = The token that was actually generated (with probability)</li>
          <li><strong>Below</strong> = Alternative tokens the model considered</li>
          <li><strong>Bar at bottom</strong> = Model confidence (green = high, yellow = medium, red = low)</li>
          <li><strong>Low confidence</strong> = Model was uncertain, could have said something else</li>
        </ul>
      </div>
    </div>
  );
}
