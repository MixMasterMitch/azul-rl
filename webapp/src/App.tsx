import { useState, useEffect, useCallback, useRef } from 'react';
import { GameState, GameSummary, Opponent, Profile, LeaderboardEntry } from './types';
import { GameSetup } from './components/GameSetup';
import { GameBoard } from './components/GameBoard';
import { api, ApiError } from './api';
import './app.css';

type Section = 'play' | 'history' | 'leaderboard';
const gameFromHash = () => window.location.hash.startsWith('#game/') ? window.location.hash.slice(6) : null;

export default function App() {
  const [username, setUsername] = useState(localStorage.getItem('azul.username') ?? '');
  const [nameInput, setNameInput] = useState('');
  const [section, setSection] = useState<Section>('play');
  const [gameId, setGameId] = useState<string | null>(gameFromHash);
  const [game, setGame] = useState<GameState | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [available, setAvailable] = useState<Opponent[]>([]);
  const [history, setHistory] = useState<GameSummary[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [filter, setFilter] = useState('');
  const [leaders, setLeaders] = useState<LeaderboardEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [paused, setPaused] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const identityRef = useRef(username);
  identityRef.current = username;

  useEffect(() => {
    const handler = () => { setGameId(gameFromHash()); setSection('play'); setPaused(false); };
    window.addEventListener('hashchange', handler);
    return () => window.removeEventListener('hashchange', handler);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    api<{ opponents: Opponent[] }>('/opponents', '', { signal: controller.signal })
      .then(data => setAvailable(data.opponents)).catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    if (!username) return;
    const controller = new AbortController();
    api<Profile>('/me', username, { signal: controller.signal }).then(setProfile)
      .catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [username, game?.status, refresh]);

  useEffect(() => {
    if (!username || !gameId) { setGame(null); return; }
    const controller = new AbortController();
    setBusy(true); setError(null);
    api<GameState>(`/game/${encodeURIComponent(gameId)}`, username, { signal: controller.signal })
      .then(state => { if (!controller.signal.aborted) { setGame(state); setPaused(false); } })
      .catch(e => { if (!controller.signal.aborted) { setGame(null); setError(e.message); } })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [username, gameId, refresh]);

  // One persisted AI move per request. A refresh or conflict simply resumes this
  // loop from the authoritative revision; no browser-held engine state exists.
  useEffect(() => {
    if (!username || !game || game.status !== 'active' || game.current_player === game.human_seat || paused || section !== 'play') return;
    const controller = new AbortController();
    api<GameState>(`/game/${game.game_id}/step-ai`, username, {
      method: 'POST', body: JSON.stringify({ expected_revision: game.revision }), signal: controller.signal,
    }).then(state => { if (!controller.signal.aborted) setGame(state); })
      .catch(async e => {
        if (controller.signal.aborted) return;
        if (e instanceof ApiError && e.status === 409) {
          try {
            const state = await api<GameState>(`/game/${game.game_id}`, username, { signal: controller.signal });
            if (!controller.signal.aborted) setGame(state);
          } catch (reloadError) {
            if (!controller.signal.aborted) { setError((reloadError as Error).message); setPaused(true); }
          }
        } else { setError(e.message); setPaused(true); }
      });
    return () => controller.abort();
  }, [username, game, paused, section]);

  useEffect(() => {
    if (!username || section === 'play') return;
    const controller = new AbortController();
    setBusy(true); setError(null);
    const task = section === 'history'
      ? api<{ games: GameSummary[]; next_cursor: string | null }>(`/games${filter ? `?status=${filter}` : ''}`, username, { signal: controller.signal })
        .then(data => { setHistory(data.games); setCursor(data.next_cursor); })
      : api<{ entities: LeaderboardEntry[] }>('/leaderboard', username, { signal: controller.signal }).then(data => setLeaders(data.entities));
    task.catch(e => { if (!controller.signal.aborted) setError(e.message); })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [username, section, filter, refresh]);

  const openGame = (id: string) => {
    setSection('play'); setError(null); setPaused(false);
    if (window.location.hash === `#game/${id}`) setRefresh(value => value + 1);
    else window.location.hash = `game/${id}`;
  };

  const createGame = async (numPlayers: number, opponents: string[]) => {
    setBusy(true); setError(null);
    const owner = username;
    try {
      const state = await api<GameState>('/game', owner, { method: 'POST', body: JSON.stringify({ num_players: numPlayers, human_seat: 0, opponents }) });
      if (identityRef.current === owner) { setGame(state); openGame(state.game_id); }
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  const mutate = useCallback(async (action: number | 'abandon') => {
    if (!game) return;
    setBusy(true); setError(null);
    try {
      const state = await api<GameState>(`/game/${game.game_id}/${action === 'abandon' ? 'abandon' : 'action'}`, username, {
        method: 'POST', body: JSON.stringify({ expected_revision: game.revision, ...(action === 'abandon' ? {} : { action }) }),
      });
      if (identityRef.current === username) setGame(state);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setError('This game changed in another request. Reloading the current board.');
        setRefresh(value => value + 1);
      } else { setError((e as Error).message); setPaused(true); }
    } finally { setBusy(false); }
  }, [game, username]);

  const loadMore = async () => {
    if (!cursor) return;
    setBusy(true);
    try {
      const query = new URLSearchParams({ cursor, ...(filter ? { status: filter } : {}) });
      const data = await api<{ games: GameSummary[]; next_cursor: string | null }>(`/games?${query}`, username);
      setHistory(previous => [...new Map([...previous, ...data.games].map(row => [row.game_id, row])).values()]);
      setCursor(data.next_cursor);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  const aiThinking = !!game && game.status === 'active' && game.current_player !== game.human_seat && !paused;
  const newGame = () => { window.location.hash = ''; setGameId(null); setGame(null); setSection('play'); setError(null); };

  return (
    <div className="app-shell">
      <header className="app-header">
        <a href="#" className="brand" onClick={newGame}><span className="brand-tiles"><i /><i /><i /><i /></span>Azul<span className="brand-caption">Play & learn</span></a>
        {username && <div className="identity"><strong>{username}</strong><button className="quiet" onClick={() => {
          localStorage.removeItem('azul.username'); setUsername(''); setProfile(null); setGame(null); window.location.hash = ''; setError(null);
        }}>Switch name</button></div>}
      </header>
      {error && <div className="error" role="alert">{error}<button onClick={() => { setPaused(false); setRefresh(value => value + 1); }}>Retry</button></div>}
      {!username ? (
        <main className="welcome">
          <p className="eyebrow">One tile at a time</p><h1>A beautiful game.<br />A worthy opponent.</h1>
          <p className="intro">Play Azul against trained AI and handcrafted bots. Your games and progress stay with your screen name.</p>
          <form className="panel name-form" onSubmit={e => {
            e.preventDefault(); const name = nameInput.trim();
            if (!/^[A-Za-z0-9_-]{1,32}$/.test(name)) { setError('Use 1–32 letters, numbers, underscores, or hyphens.'); return; }
            localStorage.setItem('azul.username', name); setUsername(name); setError(null);
          }}>
            <label htmlFor="screen-name">Your screen name</label>
            <input id="screen-name" value={nameInput} onChange={e => setNameInput(e.target.value)} maxLength={32} autoComplete="username" placeholder="Choose a name" required />
            <p className="hint">Enter the same name next time to resume your games. No password needed.</p>
            <button className="primary full">Let’s play</button>
          </form>
        </main>
      ) : <>
        <nav className="tabs" aria-label="Main navigation">
          <button aria-current={section === 'play' ? 'page' : undefined} onClick={() => setSection('play')}>Play</button>
          <button aria-current={section === 'history' ? 'page' : undefined} onClick={() => setSection('history')}>My games</button>
          <button aria-current={section === 'leaderboard' ? 'page' : undefined} onClick={() => setSection('leaderboard')}>Leaderboard</button>
        </nav>
        <div className="stats" aria-label="Your progress">
          <div><strong>{profile?.games ?? 0}</strong><span>Games finished</span></div>
          <div><strong>{profile?.wins ?? 0}</strong><span>First-place finishes</span></div>
          <div><strong>{profile?.rating ?? 'Unplaced'}</strong><span>{profile?.placed ? 'Your rating' : `${Math.max(0, (profile?.placement_wins_required ?? 5) - (profile?.wins ?? 0))} more wins to place`}</span></div>
        </div>
        <main>
          {section === 'play' && (game ? <>
            <div className="game-toolbar"><div><p className="eyebrow">{game.num_players}-player table</p><h2>{game.opponent_names.join(' · ')}</h2></div>
              <div className="button-row"><button onClick={newGame}>New game</button>{game.can_abandon && <button className="quiet" disabled={busy || aiThinking} onClick={() => mutate('abandon')}>Abandon game</button>}</div>
            </div>
            <GameBoard state={game} onAction={action => void mutate(action)} loading={busy || aiThinking} />
          </> : busy || gameId ? <section className="panel"><p>{busy ? 'Loading your table…' : 'Unable to open this game.'}</p>{!busy && <button onClick={newGame}>Start a new game</button>}</section>
            : <GameSetup onStart={createGame} loading={busy} available={available} />)}
          {section === 'history' && <section className="panel">
            <div className="section-heading"><div><p className="eyebrow">Pick up where you left off</p><h2>My games</h2></div>
              <select aria-label="Filter games" value={filter} onChange={e => setFilter(e.target.value)}><option value="">All games</option><option value="active">Active</option><option value="completed">Completed</option><option value="abandoned">Abandoned</option></select></div>
            {!history.length && <p className="empty">{busy ? 'Loading games…' : 'No games here yet. Start a new table to begin.'}</p>}
            <div className="game-list">{history.map(row => <article key={row.game_id} className="history-row">
              <div><span className={`status ${row.status}`}>{row.status}</span><h3>{row.num_players} players · {row.opponent_names.join(', ')}</h3>
                <p className="muted">{new Date(row.updated_at).toLocaleString()}{row.status === 'completed' && ` · Scores: ${row.scores.join(' / ')}`}</p></div>
              <button onClick={() => openGame(row.game_id)}>{row.status === 'active' ? 'Resume' : 'View result'}</button>
            </article>)}</div>
            {cursor && <button disabled={busy} onClick={loadMore}>{busy ? 'Loading…' : 'Load more'}</button>}
          </section>}
          {section === 'leaderboard' && <section className="panel">
            <p className="eyebrow">Humans & bots</p><h2>Leaderboard</h2><p className="muted">Finish first five times to place. Ratings are shown separately for each table size.</p>
            <div className="table-scroll"><table><thead><tr><th>Player</th><th>Rating</th><th>2 players</th><th>3 players</th><th>4 players</th></tr></thead>
              <tbody>{leaders.map(row => <tr key={row.entity_id}><td><strong>{row.label}</strong><span className="table-kind">{row.kind === 'human' ? 'Human' : 'Bot'}</span></td>
                <td>{row.rating ?? '—'}</td><td>{row.rating_2p ?? '—'}</td><td>{row.rating_3p ?? '—'}</td><td>{row.rating_4p ?? '—'}</td></tr>)}</tbody></table></div>
            {!leaders.length && <p className="empty">{busy ? 'Loading leaderboard…' : 'No ratings available yet.'}</p>}
          </section>}
        </main>
      </>}
      <footer>Azul · 2–4 players · Trained AI for two-player tables</footer>
    </div>
  );
}
