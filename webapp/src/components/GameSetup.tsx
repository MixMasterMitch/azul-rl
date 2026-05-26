import { useState } from 'react';

interface Props {
  onStart: (numPlayers: number, opponents: string[]) => void;
  loading: boolean;
}

export function GameSetup({ onStart, loading }: Props) {
  const [numPlayers, setNumPlayers] = useState(2);
  const [opponentType, setOpponentType] = useState('heuristic');

  const handleStart = () => {
    const opponents = Array(numPlayers - 1).fill(opponentType);
    onStart(numPlayers, opponents);
  };

  return (
    <div style={{
      maxWidth: '400px',
      margin: '0 auto',
      background: '#16213e',
      padding: '2rem',
      borderRadius: '12px',
    }}>
      <h2 style={{ marginBottom: '1.5rem' }}>New Game</h2>

      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', marginBottom: '0.5rem' }}>Players</label>
        <select
          value={numPlayers}
          onChange={e => setNumPlayers(Number(e.target.value))}
          style={{ width: '100%', padding: '0.5rem', borderRadius: '6px', background: '#0f3460', color: '#eee', border: '1px solid #444' }}
        >
          <option value={2}>2 Players</option>
          <option value={3}>3 Players</option>
          <option value={4}>4 Players</option>
        </select>
      </div>

      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'block', marginBottom: '0.5rem' }}>Opponent</label>
        <select
          value={opponentType}
          onChange={e => setOpponentType(e.target.value)}
          style={{ width: '100%', padding: '0.5rem', borderRadius: '6px', background: '#0f3460', color: '#eee', border: '1px solid #444' }}
        >
          <option value="random">Random Bot</option>
          <option value="heuristic">Heuristic Bot</option>
        </select>
      </div>

      <button
        onClick={handleStart}
        disabled={loading}
        style={{
          width: '100%',
          padding: '0.75rem',
          borderRadius: '8px',
          background: loading ? '#555' : '#2196F3',
          color: '#fff',
          border: 'none',
          fontSize: '1.1rem',
          cursor: loading ? 'not-allowed' : 'pointer',
        }}
      >
        {loading ? 'Creating...' : 'Start Game'}
      </button>
    </div>
  );
}
