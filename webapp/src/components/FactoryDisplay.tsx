import { COLOR_NAMES } from '../types';
import { Tile } from './Tile';

interface Props {
  tiles: number[];
  label: string;
  sourceIndex: number;
  interactive?: boolean;
  selectedColor?: number | null;
  selectionSource?: number | null;
  availableColors?: Set<number>;
  onColorClick?: (sourceIndex: number, color: number) => void;
}

const TILE = 22;
const SLOT = 26;

export function FactoryDisplay({
  tiles,
  label,
  sourceIndex,
  interactive = false,
  selectedColor = null,
  selectionSource = null,
  availableColors,
  onColorClick,
}: Props) {
  const totalTiles = tiles.reduce((a, b) => a + b, 0);
  if (totalTiles === 0) return null;

  const tileSlots: (number | null)[] = [];
  for (let color = 0; color < tiles.length; color++) {
    for (let i = 0; i < tiles[color]; i++) {
      tileSlots.push(color);
    }
  }
  while (tileSlots.length < 4) tileSlots.push(null);
  const displaySlots = tileSlots.slice(0, 4);

  const isSelected = (color: number) =>
    selectionSource === sourceIndex && selectedColor === color;

  const isAvailable = (color: number) =>
    !availableColors || availableColors.has(color);

  const handleColorClick = (color: number) => {
    if (!interactive || !onColorClick || !isAvailable(color)) return;
    onColorClick(sourceIndex, color);
  };

  const sourceSelected = selectionSource === sourceIndex;

  return (
    <div
      role="group"
      aria-label={`Factory ${sourceIndex + 1}`}
      style={{
        background: '#0f3460',
        borderRadius: '12px',
        width: `${SLOT * 2 + 12}px`,
        padding: '6px',
        border: '2px solid',
        borderColor: sourceSelected ? '#64b5f6' : 'transparent',
        boxSizing: 'border-box',
      }}
    >
      <div style={{ fontSize: '0.6rem', color: '#888', textAlign: 'center', marginBottom: '4px' }}>
        {label}
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '4px',
          justifyItems: 'center',
        }}
      >
        {displaySlots.map((slot, i) => {
          if (slot === null) {
            return (
              <div
                key={i}
                style={{
                  width: `${SLOT}px`,
                  height: `${SLOT}px`,
                  borderRadius: '4px',
                  border: '1px dashed #334466',
                  background: '#0a2540',
                  boxSizing: 'border-box',
                }}
              />
            );
          }

          const selected = isSelected(slot);
          const available = isAvailable(slot);
          const clickable = interactive && available && tiles[slot] > 0;

          return (
            <Tile
              key={i}
              color={slot}
              size={TILE}
              selected={selected}
              dimmed={interactive && !available}
              onClick={clickable ? () => handleColorClick(slot) : undefined}
              title={COLOR_NAMES[slot]}
            />
          );
        })}
      </div>
    </div>
  );
}
