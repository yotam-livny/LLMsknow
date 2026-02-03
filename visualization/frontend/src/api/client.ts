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
};

export default api;
