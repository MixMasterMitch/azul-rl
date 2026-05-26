import { Tile } from './Tile';

interface Props {
  tiles: number[];
  sourceIndex: number;
  hasFirst?: boolean;
  interactive?: boolean;
  selectedColor?: number | null;
  selectionSource?: number | null;
  availableColors?: Set<number>;
  onColorClick?: (sourceIndex: number, color: number) => void;
}

const TILE = 22;
const GAP = 4;
const COLS = 4;

export function CenterDisplay({
  tiles,
  sourceIndex,
  hasFirst,
  interactive = false,
  selectedColor = null,
  selectionSource = null,
  availableColors,
  onColorClick,
}: Props) {
  const totalTiles = tiles.reduce((a, b) => a + b, 0);
  if (totalTiles === 0 && !hasFirst) return null;

  const isSelected = (color: number) =>
    selectionSource === sourceIndex && selectedColor === color;

  const isAvailable = (color: number) =>
    !availableColors || availableColors.has(color);

  const handleColorClick = (color: number) => {
    if (!interactive || !onColorClick || !isAvailable(color)) return;
    onColorClick(sourceIndex, color);
  };

  const sourceSelected = selectionSource === sourceIndex;

  type GridItem = { key: string; kind: 'tile'; color: number } | { key: string; kind: 'first' };
  const gridItems: GridItem[] = [];
  if (hasFirst) gridItems.push({ key: 'first', kind: 'first' });
  for (let color = 0; color < tiles.length; color++) {
    for (let i = 0; i < tiles[color]; i++) {
      gridItems.push({ key: `${color}-${i}`, kind: 'tile', color });
    }
  }

  const colCount = Math.min(COLS, Math.max(gridItems.length, 1));
  const gridWidth = colCount * TILE + (colCount - 1) * GAP;

  return (
    <div
      style={{
        background: '#0f3460',
        borderRadius: '12px',
        padding: '8px 10px',
        border: '2px solid',
        borderColor: sourceSelected ? '#64b5f6' : 'transparent',
        boxSizing: 'border-box',
      }}
    >
      <div style={{ fontSize: '0.6rem', color: '#888', textAlign: 'center', marginBottom: '6px' }}>
        Center
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${colCount}, ${TILE}px)`,
          gap: `${GAP}px`,
          width: `${gridWidth}px`,
          justifyContent: 'center',
        }}
      >
        {gridItems.map(item => {
          if (item.kind === 'first') {
            return (
              <div
                key={item.key}
                style={{
                  width: `${TILE}px`,
                  height: `${TILE}px`,
                  borderRadius: '4px',
                  background: '#fff',
                  border: '2px solid #333',
                  fontSize: '7px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 'bold',
                  color: '#333',
                  boxSizing: 'border-box',
                }}
                title="First player marker"
              >
                1st
              </div>
            );
          }

          const selected = isSelected(item.color);
          const available = isAvailable(item.color);
          const clickable = interactive && available;

          return (
            <Tile
              key={item.key}
              color={item.color}
              size={TILE}
              selected={selected}
              dimmed={interactive && !available}
              onClick={clickable ? () => handleColorClick(item.color) : undefined}
            />
          );
        })}
      </div>
    </div>
  );
}
