import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useStore } from '../store/useStore';

export function LayerView() {
  const svgRef = useRef<SVGSVGElement>(null);
  const { 
    layerData, 
    inferenceResult, 
    selectedTokenIndex, 
    selectedLayerIndex,
    setSelectedLayerIndex,
    setSelectedTokenIndex 
  } = useStore();

  useEffect(() => {
    if (!svgRef.current || !layerData || !inferenceResult) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 40, right: 20, bottom: 60, left: 60 };
    const width = 800 - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Prepare data - layer norms at each token position
    const numLayers = layerData.num_layers;
    const seqLen = layerData.seq_len;
    
    // Compute norms for each (layer, token) cell
    const heatmapData: { layer: number; token: number; value: number }[] = [];
    let maxValue = 0;
    let minValue = Infinity;

    Object.entries(layerData.layers).forEach(([layerIdx, layerVectors]) => {
      const layer = parseInt(layerIdx);
      (layerVectors as number[][]).forEach((vector, tokenIdx) => {
        // Compute L2 norm of the vector
        const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
        heatmapData.push({ layer, token: tokenIdx, value: norm });
        maxValue = Math.max(maxValue, norm);
        minValue = Math.min(minValue, norm);
      });
    });

    // Scales
    const xScale = d3.scaleBand()
      .domain(d3.range(seqLen).map(String))
      .range([0, width])
      .padding(0.02);

    const yScale = d3.scaleBand()
      .domain(d3.range(numLayers).map(String))
      .range([0, height])
      .padding(0.02);

    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([minValue, maxValue]);

    // Draw cells
    g.selectAll('rect')
      .data(heatmapData)
      .enter()
      .append('rect')
      .attr('x', d => xScale(String(d.token)) || 0)
      .attr('y', d => yScale(String(d.layer)) || 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', d => {
        if (d.layer === selectedLayerIndex && d.token === selectedTokenIndex) {
          return '#ff0';
        }
        return 'none';
      })
      .attr('stroke-width', d => {
        if (d.layer === selectedLayerIndex && d.token === selectedTokenIndex) {
          return 2;
        }
        return 0;
      })
      .style('cursor', 'pointer')
      .on('click', (_, d) => {
        setSelectedLayerIndex(d.layer);
        setSelectedTokenIndex(d.token);
      })
      .append('title')
      .text(d => `Layer ${d.layer}, Token ${d.token}\nNorm: ${d.value.toFixed(4)}`);

    // X axis
    const tokenLabels = inferenceResult.tokens.map((t, i) => {
      const text = t.text.slice(0, 3).replace(/\s/g, '·');
      return text || '·';
    });

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
      .text('Token Position');

    // Y axis
    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat((_, i) => i % 4 === 0 ? String(i) : ''))
      .selectAll('text')
      .style('font-size', '10px');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -45)
      .attr('text-anchor', 'middle')
      .style('fill', '#888')
      .text('Layer');

    // Title
    g.append('text')
      .attr('x', width / 2)
      .attr('y', -15)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('fill', '#fff')
      .text('Layer Representation Norms');

    // Color legend
    const legendWidth = 200;
    const legendHeight = 10;
    const legendX = width - legendWidth - 10;
    const legendY = -25;

    const legendScale = d3.scaleLinear()
      .domain([minValue, maxValue])
      .range([0, legendWidth]);

    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d => (d as number).toFixed(1));

    const legendGradient = g.append('defs')
      .append('linearGradient')
      .attr('id', 'legend-gradient')
      .attr('x1', '0%')
      .attr('x2', '100%');

    legendGradient.selectAll('stop')
      .data(d3.range(0, 1.01, 0.1))
      .enter()
      .append('stop')
      .attr('offset', d => `${d * 100}%`)
      .attr('stop-color', d => colorScale(minValue + d * (maxValue - minValue)));

    g.append('rect')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#legend-gradient)');

    g.append('g')
      .attr('transform', `translate(${legendX},${legendY + legendHeight})`)
      .call(legendAxis)
      .selectAll('text')
      .style('font-size', '8px')
      .style('fill', '#888');

  }, [layerData, inferenceResult, selectedTokenIndex, selectedLayerIndex]);

  if (!layerData || !inferenceResult) {
    return (
      <div className="layer-view empty">
        <p>Run inference to see layer representations</p>
      </div>
    );
  }

  return (
    <div className="layer-view">
      <div className="view-header">
        <h3>Layer Flow</h3>
        <div className="layer-stats">
          <span>{layerData.num_layers} layers</span>
          <span>{layerData.seq_len} tokens</span>
          <span>Hidden size: {layerData.hidden_size}</span>
        </div>
      </div>
      <svg ref={svgRef}></svg>
      
      {selectedLayerIndex !== null && (
        <div className="layer-details">
          <h4>Layer {selectedLayerIndex} Stats</h4>
          {layerData.layer_stats[selectedLayerIndex] && (
            <div className="stats-grid">
              <div className="stat">
                <label>Mean</label>
                <span>{layerData.layer_stats[selectedLayerIndex].mean.toFixed(4)}</span>
              </div>
              <div className="stat">
                <label>Std</label>
                <span>{layerData.layer_stats[selectedLayerIndex].std.toFixed(4)}</span>
              </div>
              <div className="stat">
                <label>Norm</label>
                <span>{layerData.layer_stats[selectedLayerIndex].norm.toFixed(4)}</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
