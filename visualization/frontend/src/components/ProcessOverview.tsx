import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useStore } from '../store/useStore';

export function ProcessOverview() {
  const svgRef = useRef<SVGSVGElement>(null);
  const { 
    attentionData, 
    inferenceResult, 
    selectedTokenIndex,
    setSelectedTokenIndex,
    setSelectedLayerIndex,
    setSelectedHeadIndex,
    setViewMode
  } = useStore();

  // Token index to visualize
  const tokenIdx = selectedTokenIndex ?? (inferenceResult ? inferenceResult.input_token_count - 1 : 0);

  useEffect(() => {
    if (!svgRef.current || !attentionData || !inferenceResult) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const numLayers = attentionData.num_layers;
    const numHeads = attentionData.num_heads;

    const margin = { top: 60, right: 100, bottom: 60, left: 70 };
    const cellSize = Math.min(16, 550 / Math.max(numLayers, numHeads));
    const width = numLayers * cellSize;
    const height = numHeads * cellSize;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // For each layer/head, compute the MAX attention the selected token gives to any other token
    // This shows "how focused" each head is when processing this token
    const gridData: { layer: number; head: number; maxAtt: number; targetToken: number }[] = [];
    let globalMax = 0;

    for (let layer = 0; layer < numLayers; layer++) {
      for (let head = 0; head < numHeads; head++) {
        const pattern = attentionData.patterns[layer]?.[head];
        const tokenAttention = pattern?.[tokenIdx] || [];
        
        let maxAtt = 0;
        let targetToken = 0;
        tokenAttention.forEach((att, idx) => {
          if (att > maxAtt) {
            maxAtt = att;
            targetToken = idx;
          }
        });
        
        globalMax = Math.max(globalMax, maxAtt);
        gridData.push({ layer, head, maxAtt, targetToken });
      }
    }

    // Color scale: dark gray (no focus) to white (high focus)
    const colorScale = d3.scaleLinear<string>()
      .domain([0, globalMax])
      .range(['#2a2a2a', '#ffffff']);

    // Draw cells
    g.selectAll('rect.cell')
      .data(gridData)
      .enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('x', d => d.layer * cellSize)
      .attr('y', d => d.head * cellSize)
      .attr('width', cellSize - 1)
      .attr('height', cellSize - 1)
      .attr('fill', d => colorScale(d.maxAtt))
      .attr('rx', 2)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this).attr('stroke', '#fff').attr('stroke-width', 2);
        
        const targetText = inferenceResult.tokens[d.targetToken]?.text || `pos ${d.targetToken}`;
        
        const tooltip = g.append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${d.layer * cellSize + cellSize / 2},${d.head * cellSize - 10})`);

        tooltip.append('rect')
          .attr('x', -80)
          .attr('y', -45)
          .attr('width', 160)
          .attr('height', 42)
          .attr('fill', '#1a2332')
          .attr('stroke', '#38444d')
          .attr('rx', 4);

        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('y', -28)
          .style('fill', '#fff')
          .style('font-size', '11px')
          .text(`Layer ${d.layer}, Head ${d.head}`);

        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('y', -12)
          .style('fill', '#aaa')
          .style('font-size', '10px')
          .text(`${(d.maxAtt * 100).toFixed(1)}% → "${targetText.slice(0, 12)}"`);
      })
      .on('mouseout', function() {
        d3.select(this).attr('stroke', 'none');
        g.selectAll('.tooltip').remove();
      })
      .on('click', (event, d) => {
        setSelectedLayerIndex(d.layer);
        setSelectedHeadIndex(d.head);
        setViewMode('attention');
      });

    // Layer axis (bottom)
    for (let i = 0; i < numLayers; i += 4) {
      g.append('text')
        .attr('x', i * cellSize + cellSize / 2)
        .attr('y', height + 15)
        .attr('text-anchor', 'middle')
        .style('fill', '#888')
        .style('font-size', '9px')
        .text(i);
    }

    g.append('text')
      .attr('x', width / 2)
      .attr('y', height + 40)
      .attr('text-anchor', 'middle')
      .style('fill', '#888')
      .style('font-size', '12px')
      .text('Layer →');

    // Head axis (left)
    for (let i = 0; i < numHeads; i += 4) {
      g.append('text')
        .attr('x', -8)
        .attr('y', i * cellSize + cellSize / 2 + 3)
        .attr('text-anchor', 'end')
        .style('fill', '#888')
        .style('font-size', '9px')
        .text(i);
    }

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -45)
      .attr('text-anchor', 'middle')
      .style('fill', '#888')
      .style('font-size', '12px')
      .text('← Head');

    // Title
    const tokenText = inferenceResult.tokens[tokenIdx]?.text || `Token ${tokenIdx}`;
    g.append('text')
      .attr('x', width / 2)
      .attr('y', -35)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('fill', '#fff')
      .text(`Token "${tokenText.slice(0, 15)}" - Max Attention per Head`);

    g.append('text')
      .attr('x', width / 2)
      .attr('y', -18)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', '#888')
      .text('White = focused, Dark = distributed. Click cell for details.');

    // Color legend
    const legendWidth = 15;
    const legendHeight = height * 0.6;
    const legendX = width + 15;
    const legendY = (height - legendHeight) / 2;

    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'legend-gradient')
      .attr('x1', '0%').attr('y1', '100%')
      .attr('x2', '0%').attr('y2', '0%');

    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#2a2a2a');
    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#ffffff');

    g.append('rect')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#legend-gradient)');

    g.append('text')
      .attr('x', legendX + legendWidth + 5)
      .attr('y', legendY + 5)
      .style('fill', '#888').style('font-size', '9px')
      .text(`${(globalMax * 100).toFixed(0)}%`);

    g.append('text')
      .attr('x', legendX + legendWidth + 5)
      .attr('y', legendY + legendHeight)
      .style('fill', '#888').style('font-size', '9px')
      .text('0%');

  }, [attentionData, inferenceResult, tokenIdx]);

  if (!attentionData || !inferenceResult) {
    return (
      <div className="process-overview empty">
        <p>Run inference to see the process overview</p>
      </div>
    );
  }

  return (
    <div className="process-overview">
      <div className="view-header">
        <h3>🗺️ Attention Overview</h3>
        <div className="overview-controls">
          <div className="token-selector">
            <label>Analyzing token:</label>
            <select 
              value={tokenIdx} 
              onChange={(e) => setSelectedTokenIndex(Number(e.target.value))}
            >
              {inferenceResult.tokens.map((token, idx) => (
                <option key={idx} value={idx}>
                  {idx}: "{token.text.slice(0, 12) || '(empty)'}" {idx === inferenceResult.input_token_count - 1 ? '← last input' : ''}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <svg ref={svgRef}></svg>

      <div className="overview-explanation">
        <h4>What am I seeing?</h4>
        <p>
          This grid shows <strong>where token "{inferenceResult.tokens[tokenIdx]?.text.slice(0, 10)}" focuses its attention</strong> across all {attentionData.num_layers} layers and {attentionData.num_heads} heads.
        </p>
        <ul>
          <li><strong>Each cell</strong> = one attention head at one layer</li>
          <li><strong>White</strong> = head has sharp focus (high max attention)</li>
          <li><strong>Dark gray</strong> = head distributes attention broadly (no focus)</li>
          <li><strong>Click a cell</strong> to see the full attention pattern</li>
        </ul>
      </div>
    </div>
  );
}
