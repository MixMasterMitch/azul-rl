import { LegalAction } from '../types';

interface Props {
  actions: LegalAction[];
  onAction: (action: number) => void;
}

export function ActionPanel({ actions, onAction }: Props) {
  // Group actions by source
  const grouped: Record<string, LegalAction[]> = {};
  for (const action of actions) {
    const parts = action.name.match(/pick\(([^,]+),/);
    const source = parts ? parts[1] : 'other';
    if (!grouped[source]) grouped[source] = [];
    grouped[source].push(action);
  }

  return (
    <div style={{
      background: '#16213e',
      borderRadius: '12px',
      padding: '1.5rem',
    }}>
      <h3 style={{ marginBottom: '1rem' }}>Choose an action</h3>
      {Object.entries(grouped).map(([source, sourceActions]) => (
        <div key={source} style={{ marginBottom: '1rem' }}>
          <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: '0.5rem', textTransform: 'capitalize' }}>
            {source}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {sourceActions.map(action => (
              <button
                key={action.index}
                onClick={() => onAction(action.index)}
                style={{
                  padding: '0.4rem 0.8rem',
                  borderRadius: '6px',
                  background: '#0f3460',
                  color: '#eee',
                  border: '1px solid #2196F3',
                  cursor: 'pointer',
                  fontSize: '0.8rem',
                  transition: 'background 0.2s',
                }}
                onMouseOver={e => (e.currentTarget.style.background = '#1565c0')}
                onMouseOut={e => (e.currentTarget.style.background = '#0f3460')}
                title={action.name}
              >
                {formatAction(action.name)}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function formatAction(name: string): string {
  // Simplify display: "pick(factory0,B)→line0" → "B→L1"
  const match = name.match(/pick\([^,]+,([A-Z])\)→(.+)/);
  if (!match) return name;
  const color = match[1];
  const target = match[2];
  const targetLabel = target === 'floor' ? '🗑️' : `L${parseInt(target.replace('line', '')) + 1}`;
  return `${color}→${targetLabel}`;
}
