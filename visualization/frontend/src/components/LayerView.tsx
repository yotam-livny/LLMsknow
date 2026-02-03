import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useStore } from '../store/useStore';

export function LayerView() {
  const svgRef = useRef<SVGSVGElement>(null);
  const { 
    layerData, 
    inferenceResult, 
    selectedTokenIndex,
    setSelectedTokenIndex 
  } = useStore();

  const tokenIdx = selectedTokenIndex ?? (inferenceResult ? inferenceResult.input_token_count - 1 : 0);
  const hasProbes = inferenceResult?.probe_predictions && Object.keys(inferenceResult.probe_predictions).length > 0;

  useEffect(() => {
    if (!svgRef.current || !inferenceResult) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 50, right: 40, bottom: 60, left: 70 };
    const width = 650 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // If we have probe predictions, show correctness confidence across layers
    if (hasProbes && inferenceResult.probe_predictions) {
      const probeData = Object.entries(inferenceResult.probe_predictions)
        .map(([layerStr, pred]) => ({
          layer: parseInt(layerStr),
          pCorrect: pred.probabilities[0],
          prediction: pred.prediction
        }))
        .sort((a, b) => a.layer - b.layer);

      if (probeData.length === 0) {
        g.append('text')
          .attr('x', width / 2)
          .attr('y', height / 2)
          .attr('text-anchor', 'middle')
          .style('fill', '#888')
          .text('No probe predictions available');
        return;
      }

      const xScale = d3.scaleLinear()
        .domain([d3.min(probeData, d => d.layer)!, d3.max(probeData, d => d.layer)!])
        .range([0, width]);

      const yScale = d3.scaleLinear()
        .domain([0, 1])
        .range([height, 0]);

      // Grid
      g.append('g')
        .attr('class', 'grid')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale).tickSize(-height).tickFormat(() => ''))
        .style('stroke-dasharray', '3,3')
        .style('opacity', 0.2);

      // 50% threshold line
      g.append('line')
        .attr('x1', 0)
        .attr('y1', yScale(0.5))
        .attr('x2', width)
        .attr('y2', yScale(0.5))
        .attr('stroke', '#888')
        .attr('stroke-dasharray', '5,5')
        .attr('stroke-width', 1);

      g.append('text')
        .attr('x', width + 5)
        .attr('y', yScale(0.5) + 4)
        .style('fill', '#888')
        .style('font-size', '10px')
        .text('50%');

      // Area fill
      const area = d3.area<typeof probeData[0]>()
        .x(d => xScale(d.layer))
        .y0(yScale(0.5))
        .y1(d => yScale(d.pCorrect))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(probeData)
        .attr('fill', d => {
          const avgCorrect = d3.mean(d, p => p.pCorrect) || 0.5;
          return avgCorrect > 0.5 ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)';
        })
        .attr('d', area);

      // Line
      const line = d3.line<typeof probeData[0]>()
        .x(d => xScale(d.layer))
        .y(d => yScale(d.pCorrect))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(probeData)
        .attr('fill', 'none')
        .attr('stroke', '#22c55e')
        .attr('stroke-width', 2.5)
        .attr('d', line);

      // Points
      g.selectAll('circle')
        .data(probeData)
        .enter()
        .append('circle')
        .attr('cx', d => xScale(d.layer))
        .attr('cy', d => yScale(d.pCorrect))
        .attr('r', 6)
        .attr('fill', d => d.pCorrect > 0.5 ? '#22c55e' : '#ef4444')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
          d3.select(this).attr('r', 9);
          
          const tooltip = g.append('g')
            .attr('class', 'tooltip')
            .attr('transform', `translate(${xScale(d.layer)},${yScale(d.pCorrect) - 15})`);
          
          tooltip.append('rect')
            .attr('x', -55)
            .attr('y', -35)
            .attr('width', 110)
            .attr('height', 32)
            .attr('fill', '#1a2332')
            .attr('stroke', '#38444d')
            .attr('rx', 4);
          
          tooltip.append('text')
            .attr('text-anchor', 'middle')
            .attr('y', -18)
            .style('fill', '#fff')
            .style('font-size', '11px')
            .text(`Layer ${d.layer}`);
          
          tooltip.append('text')
            .attr('text-anchor', 'middle')
            .attr('y', -4)
            .style('fill', d.pCorrect > 0.5 ? '#22c55e' : '#ef4444')
            .style('font-size', '11px')
            .text(`${(d.pCorrect * 100).toFixed(1)}% correct`);
        })
        .on('mouseout', function() {
          d3.select(this).attr('r', 6);
          g.selectAll('.tooltip').remove();
        });

      // X axis
      g.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale).ticks(8).tickFormat(d => `L${d}`))
        .selectAll('text')
        .style('fill', '#888');

      g.append('text')
        .attr('x', width / 2)
        .attr('y', height + 45)
        .attr('text-anchor', 'middle')
        .style('fill', '#888')
        .style('font-size', '12px')
        .text('Layer');

      // Y axis
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5).tickFormat(d => `${(d as number * 100).toFixed(0)}%`))
        .selectAll('text')
        .style('fill', '#888');

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', -50)
        .attr('text-anchor', 'middle')
        .style('fill', '#888')
        .style('font-size', '12px')
        .text('P(Correct)');

      // Title
      g.append('text')
        .attr('x', width / 2)
        .attr('y', -25)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('fill', '#fff')
        .text('Correctness Confidence Across Layers');

      // Final prediction
      const finalPred = probeData[probeData.length - 1];
      const finalColor = finalPred.pCorrect > 0.5 ? '#22c55e' : '#ef4444';
      
      g.append('text')
        .attr('x', width / 2)
        .attr('y', -8)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .style('fill', finalColor)
        .text(`Final: ${finalPred.pCorrect > 0.5 ? 'CORRECT' : 'INCORRECT'} (${(finalPred.pCorrect * 100).toFixed(1)}%)`);

    } else {
      // No probes - show message
      g.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2 - 20)
        .attr('text-anchor', 'middle')
        .style('fill', '#888')
        .style('font-size', '14px')
        .text('No probe predictions available');

      g.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2 + 10)
        .attr('text-anchor', 'middle')
        .style('fill', '#666')
        .style('font-size', '12px')
        .text('Train a probe to see correctness predictions');
    }

  }, [inferenceResult, hasProbes]);

  if (!inferenceResult) {
    return (
      <div className="layer-view empty">
        <p>Run inference to see layer analysis</p>
      </div>
    );
  }

  return (
    <div className="layer-view">
      <div className="view-header">
        <h3>📈 Correctness Analysis</h3>
      </div>
      
      <svg ref={svgRef}></svg>
      
      <div className="layer-explanation">
        <h4>What am I seeing?</h4>
        {hasProbes ? (
          <>
            <p>
              This shows <strong>how confident the model is that its answer is correct</strong> at each layer.
            </p>
            <ul>
              <li><strong>Above 50%</strong> = model thinks it's correct (green)</li>
              <li><strong>Below 50%</strong> = model thinks it's incorrect (red)</li>
              <li><strong>Rising line</strong> = model becomes more confident as it processes</li>
              <li><strong>The final layer</strong> gives the model's final prediction</li>
            </ul>
            <p className="insight">
              💡 <strong>Insight:</strong> If the line stays low throughout, the model "knows" it doesn't know the answer.
              If it rises sharply at the end, the model may be guessing.
            </p>
          </>
        ) : (
          <>
            <p>
              Probes detect whether the model "knows" if its answer is correct by analyzing internal representations.
            </p>
            <p>
              To enable this view, train a probe using:<br/>
              <code>python src/probe.py --model MODEL --dataset DATASET --save_clf</code>
            </p>
          </>
        )}
      </div>
    </div>
  );
}
