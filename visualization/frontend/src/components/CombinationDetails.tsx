import { useStore } from '../store/useStore';

export function CombinationDetails() {
  const { selectedCombination, selectedSample } = useStore();

  if (!selectedCombination) {
    return (
      <div className="combination-details disabled">
        <p>Select a model and dataset to see details</p>
      </div>
    );
  }

  return (
    <div className="combination-details">
      <h3>Selected Configuration</h3>
      
      <div className="details-grid">
        <div className="detail-item">
          <label>Model:</label>
          <span>{selectedCombination.model_name}</span>
        </div>
        <div className="detail-item">
          <label>Dataset:</label>
          <span>{selectedCombination.dataset_name}</span>
        </div>
        <div className="detail-item">
          <label>Status:</label>
          <span className={`status-badge ${selectedCombination.status.toLowerCase()}`}>
            {selectedCombination.status}
          </span>
        </div>
        <div className="detail-item">
          <label>Samples Processed:</label>
          <span>
            {selectedCombination.samples_processed.toLocaleString()} / {selectedCombination.samples_total.toLocaleString()}
            ({(selectedCombination.samples_coverage * 100).toFixed(1)}%)
          </span>
        </div>
        {selectedCombination.accuracy !== null && (
          <div className="detail-item">
            <label>Accuracy:</label>
            <span>{(selectedCombination.accuracy * 100).toFixed(1)}%</span>
          </div>
        )}
        <div className="detail-item">
          <label>Has Probe:</label>
          <span>{selectedCombination.has_probe ? 'Yes' : 'No'}</span>
        </div>
        {selectedCombination.probe_config && (
          <div className="detail-item">
            <label>Probe Config:</label>
            <span>Layer {selectedCombination.probe_config.layer}, Token: {selectedCombination.probe_config.token}</span>
          </div>
        )}
      </div>

      {selectedSample && (
        <div className="selected-sample">
          <h4>Selected Sample #{selectedSample.index}</h4>
          <div className="sample-content">
            <div className="question">
              <strong>Question:</strong>
              <p>{selectedSample.question}</p>
            </div>
            {selectedSample.answer && (
              <div className="answer">
                <strong>Ground Truth Answer:</strong>
                <p>{selectedSample.answer}</p>
              </div>
            )}
          </div>
          
          {selectedCombination.ready_for_visualization ? (
            <div className="visualization-ready">
              <p>✓ Ready for visualization</p>
              <p className="note">
                Layer visualization and attention analysis features coming soon!
              </p>
            </div>
          ) : (
            <div className="visualization-not-ready">
              <p>⚠ Run the pipeline to enable visualization:</p>
              <code>./run_pipeline.sh</code>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
