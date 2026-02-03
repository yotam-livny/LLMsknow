import { useStore } from '../store/useStore';

interface TokenInfo {
  id: number;
  text: string;
  position: number;
  is_input: boolean;
}

interface TokenDisplayProps {
  tokens: TokenInfo[];
}

export function TokenDisplay({ tokens }: TokenDisplayProps) {
  const { 
    selectedTokenIndex, 
    setSelectedTokenIndex,
    showInputTokens,
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
    </div>
  );
}
