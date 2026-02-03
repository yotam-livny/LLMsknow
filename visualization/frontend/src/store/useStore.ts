import { create } from 'zustand';
import { ModelSummary, CombinationInfo, DatasetSample } from '../api/client';

interface AppState {
  // Selected values
  selectedModelId: string | null;
  selectedDatasetId: string | null;
  selectedCombination: CombinationInfo | null;
  selectedSample: DatasetSample | null;
  
  // Data
  models: ModelSummary[];
  combinations: CombinationInfo[];
  
  // UI state
  isLoading: boolean;
  error: string | null;
  
  // Actions
  setSelectedModelId: (modelId: string | null) => void;
  setSelectedDatasetId: (datasetId: string | null) => void;
  setSelectedCombination: (combination: CombinationInfo | null) => void;
  setSelectedSample: (sample: DatasetSample | null) => void;
  setModels: (models: ModelSummary[]) => void;
  setCombinations: (combinations: CombinationInfo[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialState = {
  selectedModelId: null,
  selectedDatasetId: null,
  selectedCombination: null,
  selectedSample: null,
  models: [],
  combinations: [],
  isLoading: false,
  error: null,
};

export const useStore = create<AppState>((set) => ({
  ...initialState,
  
  setSelectedModelId: (modelId) => set({ 
    selectedModelId: modelId,
    selectedDatasetId: null,
    selectedCombination: null,
    selectedSample: null,
  }),
  
  setSelectedDatasetId: (datasetId) => set({ 
    selectedDatasetId: datasetId,
    selectedSample: null,
  }),
  
  setSelectedCombination: (combination) => set({ selectedCombination: combination }),
  setSelectedSample: (sample) => set({ selectedSample: sample }),
  setModels: (models) => set({ models }),
  setCombinations: (combinations) => set({ combinations }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  reset: () => set(initialState),
}));
