import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface ModelSummary {
  model_id: string;
  display_name: string;
  friendly_name: string;
  total_datasets: number;
  ready_datasets: number;
  partial_datasets: number;
  has_any_data: boolean;
}

export interface ProbeConfig {
  layer: number;
  token: string;
}

export interface CombinationInfo {
  model_id: string;
  model_name: string;
  model_friendly: string;
  dataset_id: string;
  dataset_name: string;
  output_id: string;
  has_answers: boolean;
  has_input_output_ids: boolean;
  has_probe: boolean;
  probe_config: ProbeConfig | null;
  samples_processed: number;
  samples_total: number;
  samples_coverage: number;
  accuracy: number | null;
  status: 'READY' | 'PARTIAL' | 'NOT_PROCESSED';
  ready_for_visualization: boolean;
  ready_for_probe_predictions: boolean;
}

export interface DatasetInfo {
  id: string;
  name: string;
  filename: string;
  output_id: string;
  question_col: string;
  answer_col: string;
  category: string;
  exists: boolean;
  total_samples: number;
  columns: string[];
}

export interface DatasetSample {
  index: number;
  question: string;
  answer: string | null;
}

export interface DatasetSamplesResponse {
  dataset_id: string;
  samples: DatasetSample[];
  page: number;
  page_size: number;
  total_samples: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

// Model status
export interface ModelStatus {
  loaded: boolean;
  model_id: string | null;
  device: string;
}

// Inference types
export interface TokenInfo {
  id: number;
  text: string;
  position: number;
  is_input: boolean;
}

export interface ProbePrediction {
  layer: number;
  token: string;
  prediction: number;
  confidence: number;
  probabilities: number[];
}

export interface InferenceRequest {
  model_id: string;
  question: string;
  expected_answer?: string;
  dataset_id?: string;
  sample_idx?: number;
  max_new_tokens?: number;
  extract_layers?: boolean;
  extract_attention?: boolean;
}

export interface TokenAlternative {
  token_id: number;
  token_text: string;
  probability: number;
}

export interface InferenceResponse {
  model_id: string;
  question: string;
  generated_answer: string;
  expected_answer: string | null;
  tokens: TokenInfo[];
  input_token_count: number;
  output_token_count: number;
  total_token_count: number;
  token_alternatives: TokenAlternative[][] | null;
  actual_correct: boolean | null;
  probe_predictions: Record<number, ProbePrediction> | null;
  probe_available: boolean;
  has_layer_data: boolean;
  has_attention_data: boolean;
}

export interface LayerStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  norm: number;
}

export interface LayerDataResponse {
  model_id: string;
  layers: Record<number, number[][]>;
  layer_stats: Record<number, LayerStats>;
  seq_len: number;
  num_layers: number;
  hidden_size: number;
}

export interface AttentionHeadStats {
  entropy: number;
  sparsity: number;
  max_attention: number;
  mean_self_attention: number;
}

export interface AttentionDataResponse {
  model_id: string;
  patterns: Record<number, Record<number, number[][]>>;
  statistics: Record<number, Record<number, AttentionHeadStats>>;
  seq_len: number;
  num_layers: number;
  num_heads: number;
}

// API functions
export const api = {
  // Health
  health: () => apiClient.get('/health'),
  
  // Models
  getModels: () => apiClient.get<ModelSummary[]>('/models'),
  getModelCombinations: (modelId: string) => 
    apiClient.get<CombinationInfo[]>(`/models/${encodeURIComponent(modelId)}/combinations`),
  
  // Combinations
  getCombinations: (params?: { model_id?: string; ready_only?: boolean }) =>
    apiClient.get<CombinationInfo[]>('/combinations', { params }),
  
  // Datasets
  getDatasets: () => apiClient.get<DatasetInfo[]>('/datasets'),
  getDataset: (datasetId: string) => 
    apiClient.get<DatasetInfo>(`/datasets/${datasetId}`),
  getDatasetSamples: (datasetId: string, params?: { page?: number; page_size?: number; search?: string }) =>
    apiClient.get<DatasetSamplesResponse>(`/datasets/${datasetId}/samples`, { params }),
  getSample: (datasetId: string, index: number) =>
    apiClient.get<DatasetSample>(`/datasets/${datasetId}/samples/${index}`),
  
  // Model management
  getModelStatus: () => apiClient.get<ModelStatus>('/model/status'),
  loadModel: (modelId: string, useQuantization = false) =>
    apiClient.post<ModelStatus>('/model/load', { model_id: modelId, use_quantization: useQuantization }),
  unloadModel: () => apiClient.post<ModelStatus>('/model/unload'),
  
  // Inference
  runInference: (request: InferenceRequest) =>
    apiClient.post<InferenceResponse>('/inference', request, { timeout: 300000 }), // 5 min timeout for inference
  getLayerData: () => apiClient.get<LayerDataResponse>('/inference/layers'),
  getAttentionData: () => apiClient.get<AttentionDataResponse>('/inference/attention'),
};

export default api;
