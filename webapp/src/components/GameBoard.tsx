import { useState, useMemo, useEffect, useCallback } from 'react';
import { GameState, COLOR_NAMES, COLOR_SYMBOLS, TileSelection } from '../types';
import { getValidTargets, findAction, getAvailableColorsAtSource } from '../actionUtils';
import { getPickCount } from '../gameUtils';
import { PlayerBoard } from './PlayerBoard';
import { FactoryDisplay } from './FactoryDisplay';
import { CenterDisplay } from './CenterDisplay';

interface Props {
  state: GameState;
  onAction: (action: number) => void;
  loading: boolean;
}

export function GameBoard({ state, onAction, loading }: Props) {
  const isMyTurn = state.current_player === state.human_seat && state.status === 'active';
  const centerSource = state.center_source;
  const [selection, setSelection] = useState<TileSelection | null>(null);

  useEffect(() => {
    setSelection(null);
  }, [state.current_player, state.ended, loading]);

  const validTargets = useMemo(() => {
    if (!selection) return null;
    return getValidTargets(state.legal_actions, selection.source, selection.color);
  }, [selection, state.legal_actions]);

  const handleColorClick = useCallback((source: number, color: number) => {
    if (!isMyTurn || loading) return;
    setSelection(prev =>
      prev?.source === source && prev.color === color ? null : { source, color },
    );
  }, [isMyTurn, loading]);

  const clearSelection = useCallback(() => setSelection(null), []);

  const pickCount = useMemo(
    () => (selection ? getPickCount(state, selection) : 0),
    [selection, state],
  );

  const handleTargetClick = useCallback((target: number) => {
    if (!selection || !validTargets?.has(target)) return;
    const action = findAction(state.legal_actions, selection.source, selection.color, target);
    if (action) {
      setSelection(null);
      onAction(action.index);
    }
  }, [selection, validTargets, state.legal_actions, onAction]);

  return (
    <div data-game-status={state.status}>
      {/* Game status */}
      <div style={{
        textAlign: 'center',
        marginBottom: '1.5rem',
        padding: '0.75rem',
        background: state.ended ? '#2e7d32' : '#16213e',
        borderRadius: '8px',
      }}>
        {state.status === 'abandoned' ? (
          <span>Game abandoned — no rating change</span>
        ) : state.ended ? (
          <span style={{ fontSize: '1.2rem' }}>
            {state.winner_seats.length > 1
              ? `Shared victory: ${state.winner_seats.map(seat => seat === state.human_seat ? 'you' : `Player ${seat + 1}`).join(' and ')}`
              : state.winner_seats.includes(state.human_seat) ? '🎉 You win!' : `Player ${(state.winner_seats[0] ?? 0) + 1} wins!`}
          </span>
        ) : isMyTurn ? (
          <span style={{ color: '#81c784' }}>
            {selection
              ? `Selected ${COLOR_NAMES[selection.color]} (${COLOR_SYMBOLS[selection.color]}) — click a highlighted cell`
              : 'Your turn — click a tile color in a factory or the center'}
          </span>
        ) : (
          <span style={{ color: '#ffb74d' }}>Player {state.current_player + 1} is thinking…</span>
        )}
      </div>

      {/* Factories + center */}
      <div style={{ marginBottom: '2rem' }}>
        <h3 style={{ marginBottom: '0.75rem' }}>Factories</h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'flex-start' }}>
          {state.factories.map((factory, i) => (
            <FactoryDisplay
              key={i}
              tiles={factory}
              label={`F${i}`}
              sourceIndex={i}
              interactive={isMyTurn && !loading}
              selectedColor={selection?.color ?? null}
              selectionSource={selection?.source ?? null}
              availableColors={getAvailableColorsAtSource(state.legal_actions, i)}
              onColorClick={handleColorClick}
            />
          ))}
          <CenterDisplay
            tiles={state.center}
            sourceIndex={centerSource}
            hasFirst={state.center_has_first}
            interactive={isMyTurn && !loading}
            selectedColor={selection?.color ?? null}
            selectionSource={selection?.source ?? null}
            availableColors={getAvailableColorsAtSource(state.legal_actions, centerSource)}
            onColorClick={handleColorClick}
          />
          {isMyTurn && !loading && (
            <button
              type="button"
              onClick={clearSelection}
              disabled={!selection}
              style={{
                alignSelf: 'center',
                padding: '0.3rem 0.6rem',
                borderRadius: '6px',
                background: '#37474f',
                color: '#eee',
                border: '1px solid #78909c',
                cursor: selection ? 'pointer' : 'default',
                fontSize: '0.75rem',
                whiteSpace: 'nowrap',
                visibility: selection ? 'visible' : 'hidden',
              }}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Player boards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 350px), 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        {state.players.map((player, i) => (
          <PlayerBoard
            key={i}
            player={player}
            playerIndex={i}
            isHuman={i === state.human_seat}
            isCurrent={i === state.current_player}
            interactive={isMyTurn && !loading && i === state.human_seat}
            selection={selection}
            validTargets={i === state.human_seat ? validTargets : null}
            pickCount={i === state.human_seat ? pickCount : 0}
            onTargetClick={handleTargetClick}
          />
        ))}
      </div>

      {loading && (
        <div style={{ textAlign: 'center', padding: '1rem', color: '#aaa' }}>
          Thinking...
        </div>
      )}
    </div>
  );
}
