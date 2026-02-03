import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useStore } from '../store/useStore';

export function AttentionView() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedLayer, setSelectedLayer] = useState(15);
  const [selectedHead, setSelectedHead] = useState<number | 'avg'>('avg');
  const { 
    attentionData, 
    inferenceResult, 
    selectedTokenIndex,
    setSelectedTokenIndex 
  } = useStore();

  const tokenIdx = selectedTokenIndex ?? (inferenceResult ? inferenceResult.input_token_count - 1 : 0);

  useEffect(() => {
    if (!svgRef.current || !attentionData || !inferenceResult) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const numHeads = attentionData.num_heads;
    const tokens = inferenceResult.tokens;
    const numTokens = tokens.length;

    const margin = { top: 50, right: 30, bottom: 50, left: 150 };
    const barHeight = 28;
    const width = 700 - margin.left - margin.right;
    const height = numTokens * barHeight;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Get attention based on selected head
    let attention: number[] = Array(numTokens).fill(0);
    
    if (selectedHead === 'avg') {
      // Average across all heads
      let validHeads = 0;
      for (let head = 0; head < numHeads; head++) {
        const pattern = attentionData.patterns[selectedLayer]?.[head];
        const tokenAttention = pattern?.[tokenIdx];
        if (tokenAttention) {
          tokenAttention.forEach((att, i) => {
            if (i < numTokens) attention[i] += att;
          });
          validHeads++;
        }
      }
      if (validHeads > 0) {
        attention = attention.map(a => a / validHeads);
      }
    } else {
      // Single head
      const pattern = attentionData.patterns[selectedLayer]?.[selectedHead];
      const tokenAttention = pattern?.[tokenIdx];
      if (tokenAttention) {
        tokenAttention.forEach((att, i) => {
          if (i < numTokens) attention[i] = att;
        });
      }
    }

    const maxAtt = Math.max(...attention, 0.01);

    // X scale for attention bars
    const xScale = d3.scaleLinear()
      .domain([0, maxAtt * 1.1])
      .range([0, width]);

    // Color scale
    const colorScale = d3.scaleSequential(d3.interpolateBlues)
      .domain([0, maxAtt]);

    // Draw bars for each token
    g.selectAll('rect.att-bar')
      .data(attention)
      .enter()
      .append('rect')
      .attr('class', 'att-bar')
      .attr('x', 0)
      .attr('y', (_, i) => i * barHeight + 3)
      .attr('width', d => xScale(d))
      .attr('height', barHeight - 6)
      .attr('fill', d => colorScale(d))
      .attr('rx', 4)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this).attr('stroke', '#fff').attr('stroke-width', 2);
        
        const idx = attention.indexOf(d);
        
        const tooltip = g.append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${xScale(d) + 10},${idx * barHeight + barHeight / 2})`);
        
        tooltip.append('rect')
          .attr('x', 0)
          .attr('y', -12)
          .attr('width', 80)
          .attr('height', 24)
          .attr('fill', '#1a2332')
          .attr('stroke', '#38444d')
          .attr('rx', 4);
        
        tooltip.append('text')
          .attr('x', 40)
          .attr('y', 4)
          .attr('text-anchor', 'middle')
          .style('fill', '#fff')
          .style('font-size', '11px')
          .text(`${(d * 100).toFixed(1)}%`);
      })
      .on('mouseout', function() {
        d3.select(this).attr('stroke', 'none');
        g.selectAll('.tooltip').remove();
      })
      .on('click', (_, d) => {
        const idx = attention.indexOf(d);
        setSelectedTokenIndex(idx);
      });

    // Highlight selected token row
    g.append('rect')
      .attr('x', -margin.left + 5)
      .attr('y', tokenIdx * barHeight)
      .attr('width', margin.left + width - 10)
      .attr('height', barHeight)
      .attr('fill', 'none')
      .attr('stroke', '#fbbf24')
      .attr('stroke-width', 2)
      .attr('rx', 4);

    // Token labels on left
    g.selectAll('text.token-label')
      .data(tokens)
      .enter()
      .append('text')
      .attr('class', 'token-label')
      .attr('x', -10)
      .attr('y', (_, i) => i * barHeight + barHeight / 2 + 5)
      .attr('text-anchor', 'end')
      .style('fill', (_, i) => i === tokenIdx ? '#fbbf24' : '#aaa')
      .style('font-size', '12px')
      .style('font-weight', (_, i) => i === tokenIdx ? 'bold' : 'normal')
      .text((d, i) => `${i}: ${d.text.slice(0, 12)}`);

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d => `${((d as number) * 100).toFixed(0)}%`))
      .selectAll('text')
      .style('fill', '#aaa')
      .style('font-size', '12px');

    g.append('text')
      .attr('x', width / 2)
      .attr('y', height + 40)
      .attr('text-anchor', 'middle')
      .style('fill', '#aaa')
      .style('font-size', '13px')
      .text('Attention Weight');

    // Title
    const sourceToken = tokens[tokenIdx]?.text || `Token ${tokenIdx}`;
    const headLabel = selectedHead === 'avg' ? 'all heads (avg)' : `head ${selectedHead}`;
    g.append('text')
      .attr('x', width / 2)
      .attr('y', -25)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .style('fill', '#fff')
      .text(`Where "${sourceToken.slice(0, 15)}" attends`);

    g.append('text')
      .attr('x', width / 2)
      .attr('y', -6)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', '#888')
      .text(`Layer ${selectedLayer}, ${headLabel}`);

  }, [attentionData, inferenceResult, tokenIdx, selectedLayer, selectedHead]);

  if (!attentionData || !inferenceResult) {
    return (
      <div className="attention-view empty">
        <p>Run inference to see attention patterns</p>
      </div>
    );
  }

  return (
    <div className="attention-view">
      <div className="view-header">
        <h3>🔍 Attention Pattern</h3>
        <div className="attention-controls">
          <div className="control-group">
            <label>Source token:</label>
            <select 
              value={tokenIdx} 
              onChange={(e) => setSelectedTokenIndex(Number(e.target.value))}
            >
              {inferenceResult.tokens.map((token, idx) => (
                <option key={idx} value={idx}>
                  {idx}: "{token.text.slice(0, 12) || '(empty)'}"
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="layer-head-controls">
        <div className="control-group">
          <label>Layer:</label>
          <input 
            type="range" 
            min={0} 
            max={attentionData.num_layers - 1} 
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(Number(e.target.value))}
            className="layer-slider"
          />
          <span className="slider-value">{selectedLayer}</span>
        </div>
        <div className="control-group">
          <label>Head:</label>
          <select 
            value={selectedHead} 
            onChange={(e) => setSelectedHead(e.target.value === 'avg' ? 'avg' : Number(e.target.value))}
          >
            <option value="avg">Average (all heads)</option>
            {Array.from({ length: attentionData.num_heads }, (_, i) => (
              <option key={i} value={i}>Head {i}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="attention-content">
        <div className="attention-chart-wrapper">
          <svg ref={svgRef}></svg>
        </div>
      </div>

      <div className="attention-explanation">
        <h4>How to read</h4>
        <ul>
          <li><strong>Yellow border</strong> = Source token (the one we're analyzing)</li>
          <li><strong>Bar length</strong> = How much the source token "looks at" each target</li>
          <li><strong>Click a bar</strong> to switch to that token as source</li>
        </ul>
      </div>
    </div>
  );
}
