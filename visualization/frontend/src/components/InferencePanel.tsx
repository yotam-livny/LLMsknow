import { useState } from 'react';
import { useStore } from '../store/useStore';
import api from '../api/client';
import { TokenDisplay } from './TokenDisplay';

export function InferencePanel() {
  const {
    selectedModelId,
    selectedDatasetId,
    selectedSample,
    modelLoaded,
    modelLoading,
    inferenceRunning,
    inferenceResult,
    setModelLoading,
    setModelLoaded,
    setInferenceRunning,
    setInferenceResult,
    setLayerData,
    setAttentionData,
    setError,
  } = useStore();

  const [customQuestion, setCustomQuestion] = useState('');
  const [useCustomInput, setUseCustomInput] = useState(false);

  const question = useCustomInput ? customQuestion : selectedSample?.question;
  const expectedAnswer = useCustomInput ? undefined : selectedSample?.answer;

  const canRun = selectedModelId && question && !inferenceRunning;

  const handleLoadModel = async () => {
    if (!selectedModelId) return;
    
    setModelLoading(true);
    setError(null);
    
    try {
      await api.loadModel(selectedModelId);
      setModelLoaded(true);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load model');
    } finally {
      setModelLoading(false);
    }
  };

  const handleRunInference = async () => {
    if (!selectedModelId || !question) return;

    setInferenceRunning(true);
    setError(null);

    try {
      // Load model if not loaded
      if (!modelLoaded) {
        setModelLoading(true);
        await api.loadModel(selectedModelId);
        setModelLoaded(true);
        setModelLoading(false);
      }

      // Run inference
      const response = await api.runInference({
        model_id: selectedModelId,
        question: question,
        expected_answer: expectedAnswer || undefined,
        dataset_id: selectedDatasetId || undefined,
        extract_layers: true,
        extract_attention: true,
      });

      setInferenceResult(response.data);

      // Fetch layer data
      if (response.data.has_layer_data) {
        const layerResponse = await api.getLayerData();
        setLayerData(layerResponse.data);
      }

      // Fetch attention data
      if (response.data.has_attention_data) {
        const attentionResponse = await api.getAttentionData();
        setAttentionData(attentionResponse.data);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Inference failed');
    } finally {
      setInferenceRunning(false);
    }
  };

  return (
    <div className="inference-panel">
      <div className="inference-header">
        <h3>🔬 Inference</h3>
        <div className="model-status">
          {modelLoading ? (
            <span className="status loading">Loading model...</span>
          ) : modelLoaded ? (
            <span className="status loaded">✓ Model loaded</span>
          ) : (
            <span className="status not-loaded">Model not loaded</span>
          )}
        </div>
      </div>

      <div className="input-section">
        <div className="input-mode-toggle">
          <button 
            className={!useCustomInput ? 'active' : ''} 
            onClick={() => setUseCustomInput(false)}
            disabled={!selectedSample}
          >
            From Dataset
          </button>
          <button 
            className={useCustomInput ? 'active' : ''} 
            onClick={() => setUseCustomInput(true)}
          >
            Custom Input
          </button>
        </div>

        {useCustomInput ? (
          <textarea
            className="custom-input"
            value={customQuestion}
            onChange={(e) => setCustomQuestion(e.target.value)}
            placeholder="Enter your question here..."
            rows={3}
          />
        ) : selectedSample ? (
          <div className="sample-preview">
            <div className="field">
              <label>Question:</label>
              <p>{selectedSample.question}</p>
            </div>
            {selectedSample.answer && (
              <div className="field">
                <label>Expected Answer:</label>
                <p>{selectedSample.answer}</p>
              </div>
            )}
          </div>
        ) : (
          <div className="no-sample">
            <p>Select a sample from the dataset browser, or switch to custom input</p>
          </div>
        )}
      </div>

      <div className="action-buttons">
        {!modelLoaded && (
          <button 
            className="load-model-btn"
            onClick={handleLoadModel}
            disabled={!selectedModelId || modelLoading}
          >
            {modelLoading ? 'Loading...' : 'Load Model'}
          </button>
        )}
        
        <button 
          className="run-btn"
          onClick={handleRunInference}
          disabled={!canRun}
        >
          {inferenceRunning ? (
            <>
              <span className="spinner small"></span>
              Running...
            </>
          ) : (
            '▶ Run Inference'
          )}
        </button>
      </div>

      {inferenceResult && (
        <div className="inference-result">
          <h4>Generated Answer</h4>
          <div className="answer-text">
            {inferenceResult.generated_answer || '(empty response)'}
          </div>

          {inferenceResult.expected_answer && (
            <div className="expected-answer-inline">
              <span className="label">Expected:</span>
              <span className="value">{inferenceResult.expected_answer}</span>
            </div>
          )}

          <TokenDisplay tokens={inferenceResult.tokens} />
        </div>
      )}
    </div>
  );
}
