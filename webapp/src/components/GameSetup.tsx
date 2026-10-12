import { useEffect, useState } from 'react';
import { Opponent } from '../types';
import { opponentDifficulty, opponentName } from '../opponentLabels';

interface Props {
  onStart: (numPlayers: number, humanSeat: number, opponents: string[]) => void;
  loading: boolean;
  available: Opponent[];
}

export function GameSetup({ onStart, loading, available }: Props) {
  const [numPlayers, setNumPlayers] = useState(2);
  const [humanSeat, setHumanSeat] = useState(0);
  const [opponents, setOpponents] = useState<string[]>([]);
  const seats = Array.from({ length: numPlayers }, (_, seat) => seat);
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
      <p className="muted">Choose a table size, your starting position, and an opponent for each open seat.</p>
      <label htmlFor="player-count">Players</label>
      <select id="player-count" value={numPlayers} onChange={e => {
        const count = Number(e.target.value);
        setNumPlayers(count);
        setHumanSeat(previous => Math.min(previous, count - 1));
      }}>
        <option value={2}>2 players · you and one bot</option>
        <option value={3}>3 players · you and two bots</option>
        <option value={4}>4 players · you and three bots</option>
      </select>
      <label htmlFor="human-seat">Your starting position</label>
      <select id="human-seat" value={humanSeat} aria-describedby="human-seat-hint"
        onChange={e => setHumanSeat(Number(e.target.value))}>
        {seats.map(seat => <option key={seat} value={seat}>
          Position {seat + 1}
        </option>)}
      </select>
      <p id="human-seat-hint" className="hint">Sets your turn order for the first round.</p>
      {seats.filter(seat => seat !== humanSeat).map((seat, i) => (
        <div key={i}>
          <label htmlFor={`opponent-${i}`}>Position {seat + 1}</label>
          <select id={`opponent-${i}`} value={opponents[i] ?? ''} onChange={e => {
            const value = e.target.value;
            setOpponents(previous => previous.map((id, index) => index === i ? value : id));
          }}>
            {supported.map(opponent => <option key={opponent.id} value={opponent.id}>
              {opponentName(opponent.id, opponent.name)} ({opponentDifficulty(opponent.id)})
            </option>)}
          </select>
        </div>
      ))}
      <button className="primary full" disabled={loading || !supported.length || opponents.length !== numPlayers - 1}
        onClick={() => onStart(numPlayers, humanSeat, opponents)}>{loading ? 'Creating game…' : 'Start game'}</button>
    </section>
  );
}
