import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useStore } from '../store/useStore';

export function AttentionView() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const { attentionData, inferenceResult } = useStore();

  useEffect(() => {
    if (!svgRef.current || !attentionData || !inferenceResult) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 40, right: 20, bottom: 60, left: 60 };
    const width = 500 - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Get attention pattern for selected layer and head
    const pattern = attentionData.patterns[selectedLayer]?.[selectedHead];
    if (!pattern) return;

    const seqLen = pattern.length;

    // Scales
    const xScale = d3.scaleBand()
      .domain(d3.range(seqLen).map(String))
      .range([0, width])
      .padding(0.02);

    const yScale = d3.scaleBand()
      .domain(d3.range(seqLen).map(String))
      .range([0, height])
      .padding(0.02);

    const colorScale = d3.scaleSequential(d3.interpolateBlues)
      .domain([0, 1]);

    // Prepare data
    const cells: { x: number; y: number; value: number }[] = [];
    pattern.forEach((row, i) => {
      row.forEach((value, j) => {
        cells.push({ x: j, y: i, value });
      });
    });

    // Draw cells
    g.selectAll('rect')
      .data(cells)
      .enter()
      .append('rect')
      .attr('x', d => xScale(String(d.x)) || 0)
      .attr('y', d => yScale(String(d.y)) || 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.value))
      .append('title')
      .text(d => {
        const fromToken = inferenceResult.tokens[d.y]?.text.slice(0, 10) || `pos ${d.y}`;
        const toToken = inferenceResult.tokens[d.x]?.text.slice(0, 10) || `pos ${d.x}`;
        return `"${fromToken}" → "${toToken}"\nAttention: ${d.value.toFixed(4)}`;
      });

    // X axis (Key positions)
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat((_, i) => i % 5 === 0 ? String(i) : ''))
      .selectAll('text')
      .style('font-size', '10px');

    g.append('text')
      .attr('x', width / 2)
      .attr('y', height + 45)
      .attr('text-anchor', 'middle')
      .style('fill', '#888')
      .text('Key Position (attends to)');

    // Y axis (Query positions)
    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat((_, i) => i % 5 === 0 ? String(i) : ''))
      .selectAll('text')
      .style('font-size', '10px');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -45)
      .attr('text-anchor', 'middle')
      .style('fill', '#888')
      .text('Query Position (from)');

    // Title
    g.append('text')
      .attr('x', width / 2)
      .attr('y', -15)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('fill', '#fff')
      .text(`Attention Pattern - Layer ${selectedLayer}, Head ${selectedHead}`);

  }, [attentionData, inferenceResult, selectedLayer, selectedHead]);

  if (!attentionData || !inferenceResult) {
    return (
      <div className="attention-view empty">
        <p>Run inference to see attention patterns</p>
      </div>
    );
  }

  const stats = attentionData.statistics[selectedLayer]?.[selectedHead];

  return (
    <div className="attention-view">
      <div className="view-header">
        <h3>Attention Flow</h3>
        <div className="selectors">
          <label>
            Layer:
            <select 
              value={selectedLayer} 
              onChange={(e) => setSelectedLayer(Number(e.target.value))}
            >
              {Array.from({ length: attentionData.num_layers }, (_, i) => (
                <option key={i} value={i}>Layer {i}</option>
              ))}
            </select>
          </label>
          <label>
            Head:
            <select 
              value={selectedHead} 
              onChange={(e) => setSelectedHead(Number(e.target.value))}
            >
              {Array.from({ length: attentionData.num_heads }, (_, i) => (
                <option key={i} value={i}>Head {i}</option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="attention-content">
        <svg ref={svgRef}></svg>
        
        {stats && (
          <div className="attention-stats">
            <h4>Head Statistics</h4>
            <div className="stats-grid">
              <div className="stat">
                <label>Entropy</label>
                <span>{stats.entropy.toFixed(4)}</span>
                <small>Higher = more distributed</small>
              </div>
              <div className="stat">
                <label>Sparsity</label>
                <span>{(stats.sparsity * 100).toFixed(1)}%</span>
                <small>Near-zero weights</small>
              </div>
              <div className="stat">
                <label>Max Attention</label>
                <span>{stats.max_attention.toFixed(4)}</span>
                <small>Strongest focus</small>
              </div>
              <div className="stat">
                <label>Self-Attention</label>
                <span>{stats.mean_self_attention.toFixed(4)}</span>
                <small>Diagonal average</small>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="head-overview">
        <h4>All Heads Overview</h4>
        <div className="head-grid">
          {Array.from({ length: attentionData.num_layers }, (_, layer) => (
            <div key={layer} className="layer-row">
              <span className="layer-label">L{layer}</span>
              {Array.from({ length: attentionData.num_heads }, (_, head) => {
                const headStats = attentionData.statistics[layer]?.[head];
                const entropy = headStats?.entropy || 0;
                const isSelected = layer === selectedLayer && head === selectedHead;
                return (
                  <div
                    key={head}
                    className={`head-cell ${isSelected ? 'selected' : ''}`}
                    style={{
                      backgroundColor: `hsl(200, ${Math.min(entropy * 20, 100)}%, ${30 + entropy * 10}%)`
                    }}
                    onClick={() => {
                      setSelectedLayer(layer);
                      setSelectedHead(head);
                    }}
                    title={`Layer ${layer}, Head ${head}\nEntropy: ${entropy.toFixed(3)}`}
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
