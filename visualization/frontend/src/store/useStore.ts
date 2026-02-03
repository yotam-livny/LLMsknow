import { create } from 'zustand';
import { 
  ModelSummary, CombinationInfo, DatasetSample, 
  InferenceResponse, LayerDataResponse, AttentionDataResponse 
} from '../api/client';

type ViewMode = 'layer' | 'attention' | 'dimension';

interface AppState {
  // Selected values
  selectedModelId: string | null;
  selectedDatasetId: string | null;
  selectedCombination: CombinationInfo | null;
  selectedSample: DatasetSample | null;
  selectedTokenIndex: number | null;
  selectedLayerIndex: number | null;
  selectedHeadIndex: number | null;
  
  // Data
  models: ModelSummary[];
  combinations: CombinationInfo[];
  
  // Inference state
  modelLoaded: boolean;
  modelLoading: boolean;
  inferenceRunning: boolean;
  inferenceResult: InferenceResponse | null;
  layerData: LayerDataResponse | null;
  attentionData: AttentionDataResponse | null;
  
  // Visualization state
  viewMode: ViewMode;
  showInputTokens: boolean;
  
  // UI state
  isLoading: boolean;
  error: string | null;
  
  // Actions - Selection
  setSelectedModelId: (modelId: string | null) => void;
  setSelectedDatasetId: (datasetId: string | null) => void;
  setSelectedCombination: (combination: CombinationInfo | null) => void;
  setSelectedSample: (sample: DatasetSample | null) => void;
  setSelectedTokenIndex: (index: number | null) => void;
  setSelectedLayerIndex: (index: number | null) => void;
  setSelectedHeadIndex: (index: number | null) => void;
  
  // Actions - Data
  setModels: (models: ModelSummary[]) => void;
  setCombinations: (combinations: CombinationInfo[]) => void;
  
  // Actions - Model
  setModelLoaded: (loaded: boolean) => void;
  setModelLoading: (loading: boolean) => void;
  
  // Actions - Inference
  setInferenceRunning: (running: boolean) => void;
  setInferenceResult: (result: InferenceResponse | null) => void;
  setLayerData: (data: LayerDataResponse | null) => void;
  setAttentionData: (data: AttentionDataResponse | null) => void;
  clearInferenceData: () => void;
  
  // Actions - Visualization
  setViewMode: (mode: ViewMode) => void;
  setShowInputTokens: (show: boolean) => void;
  
  // Actions - UI
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialState = {
  selectedModelId: null,
  selectedDatasetId: null,
  selectedCombination: null,
  selectedSample: null,
  selectedTokenIndex: null,
  selectedLayerIndex: null,
  selectedHeadIndex: null,
  models: [],
  combinations: [],
  modelLoaded: false,
  modelLoading: false,
  inferenceRunning: false,
  inferenceResult: null,
  layerData: null,
  attentionData: null,
  viewMode: 'layer' as ViewMode,
  showInputTokens: true,
  isLoading: false,
  error: null,
};

export const useStore = create<AppState>((set) => ({
  ...initialState,
  
  // Selection actions
  setSelectedModelId: (modelId) => set({ 
    selectedModelId: modelId,
    selectedDatasetId: null,
    selectedCombination: null,
    selectedSample: null,
    inferenceResult: null,
    layerData: null,
    attentionData: null,
    modelLoaded: false,
  }),
  
  setSelectedDatasetId: (datasetId) => set({ 
    selectedDatasetId: datasetId,
    selectedSample: null,
  }),
  
  setSelectedCombination: (combination) => set({ selectedCombination: combination }),
  setSelectedSample: (sample) => set({ 
    selectedSample: sample,
    inferenceResult: null,
    layerData: null,
    attentionData: null,
    selectedTokenIndex: null,
    selectedLayerIndex: null,
    selectedHeadIndex: null,
  }),
  setSelectedTokenIndex: (index) => set({ selectedTokenIndex: index }),
  setSelectedLayerIndex: (index) => set({ selectedLayerIndex: index }),
  setSelectedHeadIndex: (index) => set({ selectedHeadIndex: index }),
  
  // Data actions
  setModels: (models) => set({ models }),
  setCombinations: (combinations) => set({ combinations }),
  
  // Model actions
  setModelLoaded: (loaded) => set({ modelLoaded: loaded }),
  setModelLoading: (loading) => set({ modelLoading: loading }),
  
  // Inference actions
  setInferenceRunning: (running) => set({ inferenceRunning: running }),
  setInferenceResult: (result) => set({ inferenceResult: result }),
  setLayerData: (data) => set({ layerData: data }),
  setAttentionData: (data) => set({ attentionData: data }),
  clearInferenceData: () => set({
    inferenceResult: null,
    layerData: null,
    attentionData: null,
    selectedTokenIndex: null,
    selectedLayerIndex: null,
    selectedHeadIndex: null,
  }),
  
  // Visualization actions
  setViewMode: (mode) => set({ viewMode: mode }),
  setShowInputTokens: (show) => set({ showInputTokens: show }),
  
  // UI actions
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  reset: () => set(initialState),
}));
