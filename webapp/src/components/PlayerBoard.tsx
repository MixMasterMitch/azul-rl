import {
  PlayerState,
  COLOR_HEX,
  COLOR_NAMES,
  COLOR_SYMBOLS,
  WALL_PATTERN,
  FLOOR_PENALTIES,
  FLOOR_TARGET,
  TileSelection,
  wallColumnForColor,
  CELL,
  CELL_GAP,
  ROW_WIDTH,
} from '../types';
import { Tile } from './Tile';

interface Props {
  player: PlayerState;
  playerIndex: number;
  isHuman: boolean;
  isCurrent: boolean;
  interactive?: boolean;
  selection?: TileSelection | null;
  validTargets?: Set<number> | null;
  pickCount?: number;
  onTargetClick?: (target: number) => void;
}

function hexWithAlpha(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

const cellStyle = {
  width: `${CELL}px`,
  height: `${CELL}px`,
  borderRadius: '3px',
  boxSizing: 'border-box' as const,
};

const rowGridStyle = {
  display: 'grid' as const,
  gridTemplateColumns: `repeat(5, ${CELL}px)`,
  gap: `${CELL_GAP}px`,
  width: `${ROW_WIDTH}px`,
  marginBottom: `${CELL_GAP}px`,
  boxSizing: 'border-box' as const,
};

export function PlayerBoard({
  player,
  playerIndex,
  isHuman,
  isCurrent,
  interactive = false,
  selection = null,
  validTargets = null,
  pickCount = 0,
  onTargetClick,
}: Props) {
  const placing = interactive && selection !== null && validTargets !== null;

  const firstEmptyFloorSlot = player.floor_slots?.findIndex(s => s === null || s === undefined) ?? 0;
  const emptyFloorSlots = player.floor_slots?.filter(s => s === null || s === undefined).length ?? 7;
  const floorSlotsToHighlight = placing && validTargets?.has(FLOOR_TARGET)
    ? Math.min(pickCount, emptyFloorSlots)
    : 0;

  const getWallPreviewColor = (row: number, col: number): number | null => {
    for (let r = 0; r < 5; r++) {
      const count = player.pattern_count[r];
      const color = player.pattern_color[r];
      if (count > 0 && color >= 0 && r === row && wallColumnForColor(r, color) === col) {
        return color;
      }
    }
    if (placing && selection) {
      for (const targetRow of validTargets!) {
        if (targetRow < 5 && targetRow === row && wallColumnForColor(row, selection.color) === col) {
          return selection.color;
        }
      }
    }
    return null;
  };

  const handleRowClick = (row: number) => {
    if (!placing || !validTargets?.has(row) || !onTargetClick) return;
    onTargetClick(row);
  };

  const handleFloorClick = () => {
    if (!placing || !validTargets?.has(FLOOR_TARGET) || !onTargetClick) return;
    onTargetClick(FLOOR_TARGET);
  };

  const renderPatternRow = (row: number) => {
    const capacity = row + 1;
    const count = player.pattern_count[row];
    const color = player.pattern_color[row];
    const rowValid = placing && validTargets?.has(row);
    const startCol = 5 - capacity;

    const gridCells = Array(5).fill(0).map((_, col) => {
      if (col < startCol) {
        return <div key={col} />;
      }
      const tileIdx = col - startCol;
      const filled = tileIdx < count && color >= 0;
      const highlightCell = rowValid && !filled;

      if (filled) {
        return (
          <Tile
            key={col}
            color={color}
            size={CELL}
            style={{ borderRadius: '3px' }}
            onClick={rowValid ? () => handleRowClick(row) : undefined}
            title={rowValid ? `Place on pattern line ${row + 1}` : undefined}
          />
        );
      }

      const emptyCell = (
        <div
          style={{
            ...cellStyle,
            background: '#2a2a4a',
            border: '2px solid',
            borderColor: highlightCell ? '#81c784' : '#444',
            boxShadow: highlightCell ? '0 0 4px 1px #81c784' : 'none',
            cursor: highlightCell ? 'pointer' : 'default',
          }}
        />
      );

      if (highlightCell) {
        return (
          <button
            key={col}
            type="button"
            aria-label={`Place on pattern line ${row + 1}`}
            onClick={() => handleRowClick(row)}
            style={{ padding: 0, border: 'none', background: 'none' }}
          >
            {emptyCell}
          </button>
        );
      }

      return <div key={col}>{emptyCell}</div>;
    });

    return (
      <div key={row} style={rowGridStyle}>
        {gridCells}
      </div>
    );
  };

  const renderWallRow = (row: number) => (
    <div key={row} style={rowGridStyle}>
      {[0, 1, 2, 3, 4].map(col => {
        const wallColor = WALL_PATTERN[row][col];
        const placed = player.wall[row][col];
        const previewColor = getWallPreviewColor(row, col);
        const baseTint = hexWithAlpha(COLOR_HEX[wallColor], placed ? 1 : 0.22);
        const previewTint = previewColor !== null && !placed
          ? hexWithAlpha(COLOR_HEX[previewColor], 0.55)
          : null;

        if (placed) {
          return (
            <Tile
              key={col}
              color={wallColor}
              size={CELL}
              style={{ borderRadius: '3px' }}
              title={COLOR_NAMES[wallColor]}
            />
          );
        }

        return (
          <div
            key={col}
            style={{
              ...cellStyle,
              background: previewTint ?? baseTint,
              border: previewTint
                ? `2px solid ${hexWithAlpha(COLOR_HEX[previewColor!], 0.8)}`
                : `1px solid ${hexWithAlpha(COLOR_HEX[wallColor], 0.45)}`,
              boxShadow: previewTint ? `inset 0 0 4px ${hexWithAlpha(COLOR_HEX[previewColor!], 0.4)}` : 'none',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '10px',
              color: hexWithAlpha(COLOR_HEX[wallColor], 0.5),
            }}
            title={`${COLOR_NAMES[wallColor]} slot${previewTint ? ' (incoming)' : ''}`}
          >
            {!previewTint && (
              <span style={{ opacity: 0.45 }}>{COLOR_SYMBOLS[wallColor]}</span>
            )}
          </div>
        );
      })}
    </div>
  );

  return (
    <div style={{
      background: '#16213e',
      borderRadius: '12px',
      padding: '1rem',
      border: '2px solid',
      borderColor: isCurrent ? '#64b5f6' : 'transparent',
      boxSizing: 'border-box',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
        <h4>
          Player {playerIndex + 1} {isHuman && '(You)'}
        </h4>
        <span style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#ffd54f' }}>
          {player.score} pts
        </span>
      </div>

      <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: '0.75rem', color: '#aaa', marginBottom: '0.25rem' }}>Pattern Lines</div>
          {[0, 1, 2, 3, 4].map(renderPatternRow)}
        </div>

        <div>
          <div style={{ fontSize: '0.75rem', color: '#aaa', marginBottom: '0.25rem' }}>Wall</div>
          {[0, 1, 2, 3, 4].map(renderWallRow)}
        </div>
      </div>

      <div style={{ marginTop: '0.75rem' }}>
        <div style={{ fontSize: '0.75rem', color: '#aaa', marginBottom: '0.35rem' }}>Floor Line</div>
        {placing && validTargets?.has(FLOOR_TARGET) && (
          <button type="button" onClick={handleFloorClick}
            style={{ marginBottom: '.5rem', padding: '.3rem .5rem', fontSize: '.75rem' }}>
            Place on floor
          </button>
        )}
        <div style={{ display: 'flex', gap: '4px', alignItems: 'flex-end' }}>
          {FLOOR_PENALTIES.map((penalty, i) => {
            const slot = player.floor_slots?.[i] ?? null;
            const isFirst = slot === 'first';
            const tileColor = typeof slot === 'number' ? slot : null;
            const filled = slot !== null && slot !== undefined;
            const highlightCell = floorSlotsToHighlight > 0
              && i >= firstEmptyFloorSlot
              && i < firstEmptyFloorSlot + floorSlotsToHighlight;

            const floorCell = isFirst ? (
              <div
                style={{
                  ...cellStyle,
                  background: '#fff',
                  border: '2px solid #333',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '7px',
                  fontWeight: 'bold',
                  color: '#333',
                  boxShadow: '0 0 6px rgba(255,255,255,0.6)',
                }}
                title="1st player token — holder starts next round"
              >
                1st
              </div>
            ) : tileColor !== null ? (
              <Tile color={tileColor} size={CELL} style={{ borderRadius: '3px' }} />
            ) : (
              <div
                style={{
                  ...cellStyle,
                  background: '#1a1a2e',
                  border: '2px solid',
                  borderColor: highlightCell ? '#ef5350' : '#555',
                  borderStyle: highlightCell ? 'solid' : 'dashed',
                  boxShadow: highlightCell ? '0 0 4px 1px #ef5350' : 'none',
                  cursor: highlightCell ? 'pointer' : 'default',
                }}
              />
            );

            return (
              <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px' }}>
                {highlightCell && !filled ? (
                  <button
                    type="button"
                    onClick={handleFloorClick}
                    style={{ padding: 0, border: 'none', background: 'none' }}
                  >
                    {floorCell}
                  </button>
                ) : (
                  floorCell
                )}
                <span style={{
                  fontSize: '0.6rem',
                  color: filled ? '#ef5350' : '#666',
                  fontWeight: filled ? 'bold' : 'normal',
                }}>
                  {penalty}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
