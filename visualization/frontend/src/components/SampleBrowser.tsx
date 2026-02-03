import { useState, useEffect } from 'react';
import { useStore } from '../store/useStore';
import api from '../api/client';

interface DatasetSample {
  index: number;
  question: string;
  answer: string | null;
}

export function SampleBrowser() {
  const { selectedDatasetId, selectedCombination, setSelectedSample } = useStore();
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedDatasetId) {
      setSamples([]);
      return;
    }

    const fetchSamples = async () => {
      setLoading(true);
      try {
        const response = await api.getDatasetSamples(selectedDatasetId, {
          page,
          page_size: 10,
          search: search || undefined,
        });
        setSamples(response.data.samples);
        setTotalPages(response.data.total_pages);
      } catch (err) {
        console.error('Failed to fetch samples:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchSamples();
  }, [selectedDatasetId, page, search]);

  // Reset page when dataset changes
  useEffect(() => {
    setPage(1);
    setSearch('');
  }, [selectedDatasetId]);

  if (!selectedDatasetId || !selectedCombination) {
    return (
      <div className="sample-browser disabled">
        <p>Select a dataset to browse samples</p>
      </div>
    );
  }

  return (
    <div className="sample-browser">
      <div className="sample-browser-header">
        <h3>Samples from {selectedCombination.dataset_name}</h3>
        <input
          type="text"
          placeholder="Search samples..."
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
        />
      </div>

      {loading ? (
        <div className="loading">Loading samples...</div>
      ) : (
        <>
          <div className="samples-list">
            {samples.map((sample) => (
              <div
                key={sample.index}
                className="sample-card"
                onClick={() => setSelectedSample(sample)}
              >
                <div className="sample-index">#{sample.index}</div>
                <div className="sample-question">{sample.question}</div>
                {sample.answer && (
                  <div className="sample-answer">
                    <strong>Answer:</strong> {sample.answer}
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="pagination">
            <button 
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
            >
              ← Previous
            </button>
            <span>Page {page} of {totalPages}</span>
            <button 
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
            >
              Next →
            </button>
          </div>
        </>
      )}
    </div>
  );
}
