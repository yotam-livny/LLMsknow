import { useEffect } from 'react';
import { useStore } from '../store/useStore';
import api from '../api/client';

export function ModelSelector() {
  const { models, selectedModelId, setModels, setSelectedModelId, setLoading, setError } = useStore();

  useEffect(() => {
    const fetchModels = async () => {
      setLoading(true);
      try {
        const response = await api.getModels();
        setModels(response.data);
      } catch (err) {
        setError('Failed to fetch models');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchModels();
  }, [setModels, setLoading, setError]);

  return (
    <div className="model-selector">
      <label htmlFor="model-select">Select Model:</label>
      <select
        id="model-select"
        value={selectedModelId || ''}
        onChange={(e) => setSelectedModelId(e.target.value || null)}
      >
        <option value="">-- Select a model --</option>
        {models.map((model) => (
          <option key={model.model_id} value={model.model_id}>
            {model.display_name} ({model.ready_datasets} datasets ready)
          </option>
        ))}
      </select>
    </div>
  );
}
