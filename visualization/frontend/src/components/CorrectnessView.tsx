import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useStore } from '../store/useStore';

export function CorrectnessView() {
  const svgRef = useRef<SVGSVGElement>(null);
  const { 
    inferenceResult,
    layerData,
    selectedTokenIndex,
    setSelectedTokenIndex
  } = useStore();

  const hasProbes = inferenceResult?.probe_predictions && Object.keys(inferenceResult.probe_predictions).length > 0;

  useEffect(() => {
    if (!svgRef.current || !inferenceResult) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const tokens = inferenceResult.tokens;
    const numTokens = tokens.length;

    const margin = { top: 40, right: 30, bottom: 80, left: 50 };
    const barWidth = Math.min(30, 700 / numTokens);
    const width = numTokens * barWidth;
    const height = 250 - margin.top - margin.bottom;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // For now, show token positions and highlight the "answer tokens" (output tokens)
    // These are the tokens where correctness is typically measured
    
    const inputCount = inferenceResult.input_token_count;
    
    // Color scale: input tokens are blue, output tokens are green/red based on correctness
    const getTokenColor = (idx: number) => {
      if (idx < inputCount) {
        return '#38bdf8'; // Input: sky blue
      } else {
        // Output token - if we have actual correctness, color by that
        if (inferenceResult.actual_correct !== null) {
          return inferenceResult.actual_correct ? '#22c55e' : '#ef4444';
        }
        return '#a855f7'; // Unknown: purple
      }
    };

    // Y scale for bar heights (just show position importance for now)
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    // Draw token bars
    g.selectAll('rect.token-bar')
      .data(tokens)
      .enter()
      .append('rect')
      .attr('class', 'token-bar')
      .attr('x', (_, i) => i * barWidth + 2)
      .attr('y', (_, i) => i < inputCount ? height * 0.3 : 0)
      .attr('width', barWidth - 4)
      .attr('height', (_, i) => i < inputCount ? height * 0.7 : height)
      .attr('fill', (_, i) => getTokenColor(i))
      .attr('opacity', (_, i) => i === selectedTokenIndex ? 1 : 0.6)
      .attr('stroke', (_, i) => i === selectedTokenIndex ? '#fff' : 'none')
      .attr('stroke-width', 2)
      .attr('rx', 3)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        const idx = tokens.indexOf(d);
        d3.select(this).attr('opacity', 1);
        
        const tooltip = g.append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${idx * barWidth + barWidth / 2},${-10})`);
        
        tooltip.append('rect')
          .attr('x', -60)
          .attr('y', -35)
          .attr('width', 120)
          .attr('height', 32)
          .attr('fill', '#1a2332')
          .attr('stroke', '#38444d')
          .attr('rx', 4);
        
        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('y', -18)
          .style('fill', '#fff')
          .style('font-size', '11px')
          .text(`"${d.text.slice(0, 15)}"`);
        
        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('y', -4)
          .style('fill', '#aaa')
          .style('font-size', '10px')
          .text(idx < inputCount ? 'Input token' : 'Generated token');
      })
      .on('mouseout', function(event, d) {
        const idx = tokens.indexOf(d);
        d3.select(this).attr('opacity', idx === selectedTokenIndex ? 1 : 0.6);
        g.selectAll('.tooltip').remove();
      })
      .on('click', (event, d) => {
        const idx = tokens.indexOf(d);
        setSelectedTokenIndex(idx);
      });

    // Divider line between input and output
    g.append('line')
      .attr('x1', inputCount * barWidth)
      .attr('y1', -20)
      .attr('x2', inputCount * barWidth)
      .attr('y2', height + 10)
      .attr('stroke', '#888')
      .attr('stroke-dasharray', '4,4');

    g.append('text')
      .attr('x', inputCount * barWidth)
      .attr('y', height + 25)
      .attr('text-anchor', 'middle')
      .style('fill', '#888')
      .style('font-size', '10px')
      .text('↑ Answer starts');

    // Labels
    g.append('text')
      .attr('x', inputCount * barWidth / 2)
      .attr('y', height + 50)
      .attr('text-anchor', 'middle')
      .style('fill', '#38bdf8')
      .style('font-size', '11px')
      .text(`Input (${inputCount} tokens)`);

    g.append('text')
      .attr('x', inputCount * barWidth + (numTokens - inputCount) * barWidth / 2)
      .attr('y', height + 50)
      .attr('text-anchor', 'middle')
      .style('fill', inferenceResult.actual_correct ? '#22c55e' : inferenceResult.actual_correct === false ? '#ef4444' : '#a855f7')
      .style('font-size', '11px')
      .text(`Output (${numTokens - inputCount} tokens)`);

    // Title
    g.append('text')
      .attr('x', width / 2)
      .attr('y', -20)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('fill', '#fff')
      .text('Token Sequence Analysis');

  }, [inferenceResult, selectedTokenIndex]);

  if (!inferenceResult) {
    return (
      <div className="correctness-view empty">
        <p>Run inference to see correctness analysis</p>
      </div>
    );
  }

  const actualCorrect = inferenceResult.actual_correct;

  return (
    <div className="correctness-view">
      <div className="view-header">
        <h3>✓ Correctness Analysis</h3>
      </div>

      {/* Correctness Summary */}
      <div className="correctness-summary">
        <div className="summary-cards">
          <div className={`card ${actualCorrect === true ? 'correct' : actualCorrect === false ? 'incorrect' : 'unknown'}`}>
            <div className="card-icon">
              {actualCorrect === true ? '✓' : actualCorrect === false ? '✗' : '?'}
            </div>
            <div className="card-content">
              <div className="card-label">Actual Correctness</div>
              <div className="card-value">
                {actualCorrect === true ? 'Correct' : actualCorrect === false ? 'Incorrect' : 'Unknown'}
              </div>
            </div>
          </div>

          {hasProbes && inferenceResult.probe_predictions && (
            <div className={`card ${
              Object.values(inferenceResult.probe_predictions).some(p => p.probabilities[0] > 0.5) 
                ? 'correct' : 'incorrect'
            }`}>
              <div className="card-icon">🧠</div>
              <div className="card-content">
                <div className="card-label">Model's Self-Assessment</div>
                <div className="card-value">
                  {(() => {
                    const preds = Object.values(inferenceResult.probe_predictions);
                    const lastPred = preds[preds.length - 1];
                    if (lastPred) {
                      return lastPred.probabilities[0] > 0.5 
                        ? `Thinks correct (${(lastPred.probabilities[0] * 100).toFixed(0)}%)`
                        : `Thinks incorrect (${(lastPred.probabilities[1] * 100).toFixed(0)}%)`;
                    }
                    return 'No probe data';
                  })()}
                </div>
              </div>
            </div>
          )}
        </div>

        {actualCorrect !== null && hasProbes && (
          <div className="calibration-note">
            {(() => {
              const preds = Object.values(inferenceResult.probe_predictions!);
              const lastPred = preds[preds.length - 1];
              if (lastPred) {
                const modelThinkCorrect = lastPred.probabilities[0] > 0.5;
                if (modelThinkCorrect === actualCorrect) {
                  return <span className="match">✓ Model's self-assessment matches actual correctness</span>;
                } else {
                  return <span className="mismatch">⚠ Model's self-assessment differs from actual correctness</span>;
                }
              }
              return null;
            })()}
          </div>
        )}
      </div>

      <svg ref={svgRef}></svg>

      <div className="correctness-explanation">
        <h4>What am I seeing?</h4>
        <ul>
          <li><strong>Blue bars (short)</strong> = Input tokens (the question)</li>
          <li><strong>Colored bars (tall)</strong> = Generated tokens (the answer)</li>
          <li><strong>Click any token</strong> to analyze its attention pattern in other views</li>
        </ul>
        <p className="insight">
          💡 <strong>Tip:</strong> The last few output tokens are often most important for correctness detection.
          Try selecting the last generated token to see where the model's "confidence" comes from.
        </p>
      </div>

      {/* Answer display */}
      <div className="answer-display">
        <div className="question">
          <strong>Question:</strong> {inferenceResult.question}
        </div>
        <div className="answer">
          <strong>Generated Answer:</strong> {inferenceResult.generated_answer}
        </div>
        {inferenceResult.expected_answer && (
          <div className="expected">
            <strong>Expected Answer:</strong> {inferenceResult.expected_answer}
          </div>
        )}
      </div>
    </div>
  );
}
