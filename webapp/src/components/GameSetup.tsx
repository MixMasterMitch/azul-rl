import { useEffect, useState } from 'react';
import { Opponent } from '../types';

interface Props {
  onStart: (numPlayers: number, opponents: string[]) => void;
  loading: boolean;
  available: Opponent[];
}

export function GameSetup({ onStart, loading, available }: Props) {
  const [numPlayers, setNumPlayers] = useState(2);
  const [opponents, setOpponents] = useState<string[]>([]);
  const supported = available.filter(opponent => opponent.supported_players.includes(numPlayers));
  useEffect(() => {
    const valid = available.filter(opponent => opponent.supported_players.includes(numPlayers));
    const defaultId = valid.find(opponent => opponent.default)?.id ?? 'heuristic';
    setOpponents(previous => Array.from({ length: numPlayers - 1 }, (_, i) =>
      valid.some(opponent => opponent.id === previous[i]) ? previous[i] : defaultId));
  }, [numPlayers, available]);

  return (
    <section className="panel setup">
      <p className="eyebrow">Take a seat</p>
      <h2>Start a new game</h2>
      <p className="muted">Build your wall, collect bonuses, and keep tiles off your floor.</p>
      <label htmlFor="player-count">Players</label>
      <select id="player-count" value={numPlayers} onChange={e => setNumPlayers(Number(e.target.value))}>
        <option value={2}>2 players · you and one bot</option>
        <option value={3}>3 players · you and two bots</option>
        <option value={4}>4 players · you and three bots</option>
      </select>
      {Array.from({ length: numPlayers - 1 }, (_, i) => (
        <div key={i}>
          <label htmlFor={`opponent-${i}`}>Player {i + 2}</label>
          <select id={`opponent-${i}`} value={opponents[i] ?? ''} onChange={e => {
            const value = e.target.value;
            setOpponents(previous => previous.map((id, index) => index === i ? value : id));
          }}>
            {supported.map(opponent => <option key={opponent.id} value={opponent.id}>{opponent.name}</option>)}
          </select>
        </div>
      ))}
      {numPlayers > 2 && <p className="hint">Trained AI is available in two-player games. These tables use heuristic and random opponents.</p>}
      <button className="primary full" disabled={loading || !supported.length || opponents.length !== numPlayers - 1}
        onClick={() => onStart(numPlayers, opponents)}>{loading ? 'Creating game…' : 'Start game'}</button>
    </section>
  );
}
