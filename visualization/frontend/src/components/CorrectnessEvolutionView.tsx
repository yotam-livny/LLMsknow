import { useEffect, useState, useRef } from 'react';
import { useStore } from '../store/useStore';
import * as d3 from 'd3';
import { apiClient } from '../api/client';

interface LayerProbeResult {
  layer: number;
  prediction: number;
  confidence: number;
  prob_correct: number;
  prob_incorrect: number;
}

interface ExactAnswerInfo {
  exact_answer: string | null;
  start_char: number;
  end_char: number;
  extraction_method: string;
  valid: boolean;
  token_positions: number[];
}

interface CorrectnessEvolutionData {
  question: string;
  generated_answer: string;
  expected_answer: string | null;
  actual_correct: boolean | null;
  exact_answer: ExactAnswerInfo;
  measured_token_position: number;
  measured_token_text: string;
  layer_predictions: LayerProbeResult[];
  first_confident_layer: number | null;
  peak_confidence_layer: number;
  peak_confidence: number;
  interpretation: string;
}

export function CorrectnessEvolutionView() {
  const { inferenceResult } = useStore();
  const [evolutionData, setEvolutionData] = useState<CorrectnessEvolutionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTokenPosition, setSelectedTokenPosition] = useState<number | null>(null);
  const chartRef = useRef<SVGSVGElement>(null);

  // Stable reference for inference result ID to track changes
  const inferenceId = inferenceResult?.generated_answer || null;
  const hasLayerData = inferenceResult?.has_layer_data || false;
  const tokenCount = inferenceResult?.tokens?.length || 0;

  // Reset token selection to last token when inference changes
  useEffect(() => {
    if (tokenCount > 0) {
      setSelectedTokenPosition(tokenCount - 1);
    } else {
      setSelectedTokenPosition(null);
    }
  }, [inferenceId, tokenCount]);

  // Fetch correctness evolution when token selection changes (and is valid)
  useEffect(() => {
    if (!hasLayerData || selectedTokenPosition === null) {
      setEvolutionData(null);
      return;
    }

    const fetchEvolution = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await apiClient.post('/inference/correctness-evolution', {
          use_current_session: true,
          token_position: selectedTokenPosition
        });
        setEvolutionData(response.data);
      } catch (err: any) {
        console.error('Failed to fetch correctness evolution:', err);
        setError(err.response?.data?.detail || 'Failed to analyze correctness evolution');
      } finally {
        setLoading(false);
      }
    };

    fetchEvolution();
  }, [hasLayerData, selectedTokenPosition]);

  // Render D3 chart when data changes
  useEffect(() => {
    if (!chartRef.current || !evolutionData?.layer_predictions.length) return;

    const svg = d3.select(chartRef.current);
    svg.selectAll('*').remove();

    const predictions = evolutionData.layer_predictions;
    
    // Dimensions
    const margin = { top: 30, right: 30, bottom: 50, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(predictions, d => d.layer) || 32])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).ticks(8).tickFormat(d => `L${d}`))
      .selectAll('text')
      .style('font-size', '12px');

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d => `${(d as number * 100).toFixed(0)}%`))
      .selectAll('text')
      .style('font-size', '12px');

    // Axis labels
    g.append('text')
      .attr('x', width / 2)
      .attr('y', height + 40)
      .attr('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('fill', '#888')
      .text('Layer');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -45)
      .attr('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('fill', '#888')
      .text('P(Correct)');

    // Line for P(correct)
    const line = d3.line<LayerProbeResult>()
      .x(d => xScale(d.layer))
      .y(d => yScale(d.prob_correct))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(predictions)
      .attr('fill', 'none')
      .attr('stroke', '#4CAF50')
      .attr('stroke-width', 3)
      .attr('d', line);

    // Area fill under the line
    const area = d3.area<LayerProbeResult>()
      .x(d => xScale(d.layer))
      .y0(height)
      .y1(d => yScale(d.prob_correct))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(predictions)
      .attr('fill', 'url(#gradient)')
      .attr('opacity', 0.3)
      .attr('d', area);

    // Gradient
    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'gradient')
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '0%').attr('y2', '100%');
    gradient.append('stop').attr('offset', '0%').attr('stop-color', '#4CAF50');
    gradient.append('stop').attr('offset', '100%').attr('stop-color', 'transparent');

    // Data points
    g.selectAll('.dot')
      .data(predictions)
      .enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('cx', d => xScale(d.layer))
      .attr('cy', d => yScale(d.prob_correct))
      .attr('r', 6)
      .attr('fill', d => d.prob_correct > 0.5 ? '#4CAF50' : '#ff4444')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this).attr('r', 9);
        // Tooltip
        const tooltip = g.append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${xScale(d.layer)},${yScale(d.prob_correct) - 15})`);
        
        tooltip.append('rect')
          .attr('x', -50).attr('y', -30)
          .attr('width', 100).attr('height', 25)
          .attr('fill', '#333').attr('rx', 4);
        
        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('y', -12)
          .attr('fill', '#fff')
          .style('font-size', '11px')
          .text(`L${d.layer}: ${(d.prob_correct * 100).toFixed(1)}% correct`);
      })
      .on('mouseout', function() {
        d3.select(this).attr('r', 6);
        g.selectAll('.tooltip').remove();
      });

    // 50% threshold line
    g.append('line')
      .attr('x1', 0).attr('x2', width)
      .attr('y1', yScale(0.5)).attr('y2', yScale(0.5))
      .attr('stroke', '#888')
      .attr('stroke-dasharray', '5,5')
      .attr('stroke-width', 1);

    // First confident layer marker
    if (evolutionData.first_confident_layer !== null) {
      const fcl = evolutionData.first_confident_layer;
      const fclPred = predictions.find(p => p.layer === fcl);
      if (fclPred) {
        g.append('line')
          .attr('x1', xScale(fcl)).attr('x2', xScale(fcl))
          .attr('y1', 0).attr('y2', height)
          .attr('stroke', '#ffc107')
          .attr('stroke-dasharray', '3,3')
          .attr('stroke-width', 2);
        
        g.append('text')
          .attr('x', xScale(fcl) + 5)
          .attr('y', 15)
          .attr('fill', '#ffc107')
          .style('font-size', '11px')
          .text('First confident');
      }
    }

  }, [evolutionData]);

  // Highlight exact answer in generated text
  const renderHighlightedAnswer = () => {
    if (!evolutionData?.exact_answer?.valid || !evolutionData.exact_answer.exact_answer) {
      return evolutionData?.generated_answer || '';
    }

    const { start_char, end_char, exact_answer } = evolutionData.exact_answer;
    const answer = evolutionData.generated_answer;

    if (start_char < 0 || exact_answer === "NO ANSWER") {
      return answer;
    }

    const before = answer.substring(0, start_char);
    const highlighted = answer.substring(start_char, end_char);
    const after = answer.substring(end_char);

    return (
      <>
        {before}
        <mark className="exact-answer-highlight">{highlighted}</mark>
        {after}
      </>
    );
  };

  if (!inferenceResult) {
    return (
      <div className="correctness-evolution-view empty">
        <p>Run inference to analyze correctness evolution</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="correctness-evolution-view loading">
        <div className="spinner" />
        <p>Analyzing correctness evolution...</p>
        <p className="hint">Extracting exact answer tokens and running probes across layers</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="correctness-evolution-view error">
        <p>⚠️ {error}</p>
      </div>
    );
  }

  if (!evolutionData) {
    return (
      <div className="correctness-evolution-view empty">
        <p>No correctness data available</p>
      </div>
    );
  }

  // Get probe prediction info for consolidated panel
  const getProbeInfoInput = () => {
    if (!inferenceResult?.probe_predictions) return null;
    const entries = Object.entries(inferenceResult.probe_predictions);
    if (entries.length === 0) return null;
    const [layer, pred] = entries[entries.length - 1];
    return {
      layer: parseInt(layer),
      pCorrect: pred.probabilities[0],
      isCorrect: pred.probabilities[0] > 0.5,
      tokenPos: inferenceResult.input_token_count - 1
    };
  };

  const getProbeInfoOutput = () => {
    if (!inferenceResult?.probe_predictions_output) return null;
    const entries = Object.entries(inferenceResult.probe_predictions_output);
    if (entries.length === 0) return null;
    const [layer, pred] = entries[entries.length - 1];
    return {
      layer: parseInt(layer),
      pCorrect: pred.probabilities[0],
      isCorrect: pred.probabilities[0] > 0.5,
      tokenPos: inferenceResult.total_token_count - 1
    };
  };

  const probeInfoInput = getProbeInfoInput();
  const probeInfoOutput = getProbeInfoOutput();

  return (
    <div className="correctness-evolution-view">
      <div className="view-header">
        <h3>📈 Correctness Evolution</h3>
        <p className="subtitle">
          How the model's internal "belief" about correctness evolves across layers
        </p>
      </div>

      {/* Consolidated Correctness Panel - At Top */}
      <div className="correctness-panel-consolidated">
        <div className="correctness-row probe-row">
          <div className="correctness-item">
            <span className="label">🧠 Before Generation:</span>
            {probeInfoInput ? (
              <span className={`value ${probeInfoInput.isCorrect ? 'correct' : 'incorrect'}`}>
                {probeInfoInput.isCorrect ? '✓ Correct' : '✗ Incorrect'}
                <span className="detail">({(probeInfoInput.pCorrect * 100).toFixed(1)}% at L{probeInfoInput.layer}, tok {probeInfoInput.tokenPos})</span>
              </span>
            ) : (
              <span className="value unknown">No probe</span>
            )}
          </div>
          
          <div className="correctness-item">
            <span className="label">🧠 After Generation:</span>
            {probeInfoOutput ? (
              <span className={`value ${probeInfoOutput.isCorrect ? 'correct' : 'incorrect'}`}>
                {probeInfoOutput.isCorrect ? '✓ Correct' : '✗ Incorrect'}
                <span className="detail">({(probeInfoOutput.pCorrect * 100).toFixed(1)}% at L{probeInfoOutput.layer}, tok {probeInfoOutput.tokenPos})</span>
              </span>
            ) : (
              <span className="value unknown">No probe</span>
            )}
          </div>
        </div>

        <div className="correctness-row ground-truth-row">
          <div className="correctness-item">
            <span className="label">📋 Ground Truth:</span>
            {inferenceResult?.actual_correct !== null ? (
              <span className={`value ${inferenceResult.actual_correct ? 'correct' : 'incorrect'}`}>
                {inferenceResult.actual_correct ? '✓ Correct' : '✗ Incorrect'}
              </span>
            ) : (
              <span className="value unknown">Unknown</span>
            )}
          </div>
          
          {inferenceResult?.expected_answer && (
            <div className="correctness-item">
              <span className="label">Expected:</span>
              <span className="value expected">{inferenceResult.expected_answer}</span>
            </div>
          )}
        </div>

        {probeInfoOutput && inferenceResult?.actual_correct !== null && (
          <div className={`calibration-row ${probeInfoOutput.isCorrect === inferenceResult.actual_correct ? 'match' : 'mismatch'}`}>
            {probeInfoOutput.isCorrect === inferenceResult.actual_correct ? (
              <>✓ Model's final self-assessment matches reality</>
            ) : (
              <>⚠ Model is miscalibrated (final assessment differs from reality)</>
            )}
          </div>
        )}
      </div>

      {/* Exact Answer Section */}
      <div className="exact-answer-section">
        <h4>🎯 Exact Answer Tokens</h4>
        {evolutionData.exact_answer.valid ? (
          <div className="exact-answer-content">
            <div className="highlighted-answer">
              {renderHighlightedAnswer()}
            </div>
            <div className="exact-answer-meta">
              <span className="badge method">{evolutionData.exact_answer.extraction_method}</span>
              {evolutionData.exact_answer.token_positions.length > 0 && (
                <span className="badge positions">
                  Token positions: {evolutionData.exact_answer.token_positions.join(', ')}
                </span>
              )}
            </div>
          </div>
        ) : (
          <div className="no-exact-answer">
            <p>Could not extract exact answer tokens</p>
            <p className="hint">The model may not have provided a clear answer</p>
          </div>
        )}
      </div>

      {/* Token Position Selector */}
      <div className="token-selector-section">
        <h4>📍 Measurement Position</h4>
        <div className="token-selector-content">
          <div className="current-measurement">
            <span className="label">Currently measuring at:</span>
            <span className="token-badge">
              Token #{evolutionData.measured_token_position}: 
              "{evolutionData.measured_token_text.replace(/\n/g, '↵').replace(/ /g, '·') || '□'}"
            </span>
          </div>
          
          <div className="token-selector">
            <label>Select token position:</label>
            <select 
              value={selectedTokenPosition ?? (inferenceResult?.tokens.length ? inferenceResult.tokens.length - 1 : 0)}
              onChange={(e) => {
                setSelectedTokenPosition(parseInt(e.target.value));
              }}
            >
              <optgroup label="Input Tokens">
                {inferenceResult?.tokens.filter(t => t.is_input).map(token => (
                  <option key={token.position} value={token.position}>
                    #{token.position}: "{token.text.replace(/\n/g, '↵').replace(/ /g, '·').slice(0, 20)}"
                  </option>
                ))}
              </optgroup>
              <optgroup label="Output Tokens">
                {inferenceResult?.tokens.filter(t => !t.is_input).map(token => (
                  <option key={token.position} value={token.position}>
                    #{token.position}: "{token.text.replace(/\n/g, '↵').replace(/ /g, '·').slice(0, 20)}"
                  </option>
                ))}
              </optgroup>
            </select>
          </div>

          <p className="hint">
            Select a token position to measure probe predictions at that point.
          </p>
        </div>
      </div>

      {/* Evolution Chart */}
      {evolutionData.layer_predictions.length > 0 ? (
        <div className="evolution-chart-section">
          <h4>📊 Confidence Across Layers</h4>
          <svg ref={chartRef}></svg>
          
          <div className="chart-legend">
            <div className="legend-item">
              <span className="dot green"></span>
              <span>P(Correct) &gt; 50%</span>
            </div>
            <div className="legend-item">
              <span className="dot red"></span>
              <span>P(Correct) &lt; 50%</span>
            </div>
            {evolutionData.first_confident_layer !== null && (
              <div className="legend-item">
                <span className="line yellow"></span>
                <span>First confident layer (≥70%)</span>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="no-probes">
          <p>⚠️ No multi-layer probe predictions available</p>
          <p className="hint">
            Train probes at multiple layers to see the full evolution.
            Currently only layer 15 probe is available.
          </p>
        </div>
      )}

      {/* Interpretation */}
      <div className="interpretation-section">
        <h4>💡 Interpretation</h4>
        <p className="interpretation-text">{evolutionData.interpretation}</p>
        
        {/* Ground truth comparison */}
        {evolutionData.actual_correct !== null && (
          <div className={`ground-truth-badge ${evolutionData.actual_correct ? 'correct' : 'incorrect'}`}>
            Ground Truth: {evolutionData.actual_correct ? '✓ Correct' : '✗ Incorrect'}
          </div>
        )}
      </div>

      {/* Summary Stats */}
      <div className="summary-stats">
        {evolutionData.first_confident_layer !== null && (
          <div className="stat">
            <span className="stat-label">First confident at</span>
            <span className="stat-value">Layer {evolutionData.first_confident_layer}</span>
          </div>
        )}
        <div className="stat">
          <span className="stat-label">Peak confidence</span>
          <span className="stat-value">
            {(evolutionData.peak_confidence * 100).toFixed(1)}% at L{evolutionData.peak_confidence_layer}
          </span>
        </div>
        {evolutionData.exact_answer.token_positions.length > 0 && (
          <div className="stat">
            <span className="stat-label">Measured at</span>
            <span className="stat-value">
              Token #{evolutionData.exact_answer.token_positions[evolutionData.exact_answer.token_positions.length - 1]}
            </span>
          </div>
        )}
      </div>

      {/* Important note about token positions */}
      {evolutionData.exact_answer.token_positions.length > 0 && (
        <div className="token-position-note">
          <strong>Note:</strong> This view measures probe predictions at the <em>exact answer tokens</em> 
          (positions {evolutionData.exact_answer.token_positions.join(', ')}), 
          which is where truthfulness information is concentrated according to the paper. 
          The "Probe Prediction" shown above the tokens measures at the last <em>input</em> token (before generation).
        </div>
      )}

      {/* Paper reference */}
      <div className="paper-reference">
        <small>
          Based on: <em>"LLMs Know More Than They Show"</em> - truthfulness information 
          is concentrated in exact answer tokens.
        </small>
      </div>
    </div>
  );
}
