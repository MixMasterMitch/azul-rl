import { useState, useCallback } from 'react';
import { GameState } from './types';
import { GameSetup } from './components/GameSetup';
import { GameBoard } from './components/GameBoard';

export default function App() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const createGame = useCallback(async (numPlayers: number, opponents: string[]) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_players: numPlayers, human_seat: 0, opponents }),
      });
      const data = await res.json();
      setGameState(data);
    } catch (e) {
      setError('Failed to create game');
    }
    setLoading(false);
  }, []);

  const applyAction = useCallback(async (action: number) => {
    if (!gameState) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/game/${gameState.game_id}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        setError(data.error ?? 'Failed to apply action');
        setLoading(false);
        return;
      }
      setGameState(data);

      // Step AI players
      if (!data.ended && data.current_player !== data.human_seat) {
        const aiRes = await fetch(`/api/game/${gameState.game_id}/step-ai`, {
          method: 'POST',
        });
        const aiData = await aiRes.json();
        if (!aiRes.ok || aiData.error) {
          setError(aiData.error ?? 'Failed to step AI');
        } else {
          setGameState(aiData);
        }
      }
    } catch (e) {
      setError('Failed to apply action');
    }
    setLoading(false);
  }, [gameState]);

  return (
    <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
      <h1 style={{ textAlign: 'center', marginBottom: '2rem', fontSize: '2.5rem', color: '#64b5f6' }}>
        Azul
      </h1>
      {error && (
        <div style={{ background: '#c62828', padding: '1rem', borderRadius: '8px', marginBottom: '1rem' }}>
          {error}
        </div>
      )}
      {!gameState ? (
        <GameSetup onStart={createGame} loading={loading} />
      ) : (
        <GameBoard state={gameState} onAction={applyAction} loading={loading} />
      )}
    </div>
  );
}
