import { useEffect } from 'react';
import { useStore } from '../store/useStore';
import api from '../api/client';

export function DatasetSelector() {
  const { 
    selectedModelId, 
    combinations, 
    selectedDatasetId,
    setCombinations, 
    setSelectedDatasetId,
    setSelectedCombination,
    setLoading, 
    setError 
  } = useStore();

  useEffect(() => {
    if (!selectedModelId) {
      setCombinations([]);
      return;
    }

    const fetchCombinations = async () => {
      setLoading(true);
      try {
        const response = await api.getModelCombinations(selectedModelId);
        setCombinations(response.data);
      } catch (err) {
        setError('Failed to fetch combinations');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchCombinations();
  }, [selectedModelId, setCombinations, setLoading, setError]);

  const handleSelect = (datasetId: string) => {
    setSelectedDatasetId(datasetId);
    const combo = combinations.find(c => c.dataset_id === datasetId);
    setSelectedCombination(combo || null);
  };

  if (!selectedModelId) {
    return <div className="dataset-selector disabled">Select a model first</div>;
  }

  return (
    <div className="dataset-selector">
      <h3>Available Datasets</h3>
      <table className="dataset-table">
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Processed</th>
            <th>Probe</th>
            <th>Accuracy</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {combinations.map((combo) => (
            <tr 
              key={combo.dataset_id}
              className={`${selectedDatasetId === combo.dataset_id ? 'selected' : ''} ${combo.status.toLowerCase()}`}
              onClick={() => handleSelect(combo.dataset_id)}
            >
              <td>
                <span className={`status-icon ${combo.status.toLowerCase()}`}>
                  {combo.status === 'READY' ? '✓' : combo.status === 'PARTIAL' ? '⚠' : '○'}
                </span>
                {combo.dataset_name}
              </td>
              <td>
                {combo.samples_processed > 0 
                  ? `${combo.samples_processed.toLocaleString()} / ${combo.samples_total.toLocaleString()}`
                  : '—'
                }
              </td>
              <td>{combo.has_probe ? '✓' : '✗'}</td>
              <td>{combo.accuracy !== null ? `${(combo.accuracy * 100).toFixed(1)}%` : '—'}</td>
              <td>
                <span className={`status-badge ${combo.status.toLowerCase()}`}>
                  {combo.status}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
