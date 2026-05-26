import { COLOR_HEX, COLOR_NAMES, COLOR_SYMBOLS } from '../types';

interface Props {
  color: number;
  size?: number;
  selected?: boolean;
  highlighted?: boolean;
  highlightColor?: string;
  dimmed?: boolean;
  onClick?: () => void;
  title?: string;
  style?: React.CSSProperties;
}

const SYMBOL_COLOR: Record<number, string> = {
  0: '#fff',
  1: '#333',
  2: '#fff',
  3: '#fff',
  4: '#333',
};

export function Tile({
  color,
  size = 22,
  selected = false,
  highlighted = false,
  highlightColor = '#81c784',
  dimmed = false,
  onClick,
  title,
  style,
}: Props) {
  const symbolColor = SYMBOL_COLOR[color] ?? '#fff';
  const borderColor = selected
    ? '#fff'
    : highlighted
      ? highlightColor
      : 'rgba(255,255,255,0.3)';

  const content = (
    <span style={{
      fontSize: `${Math.round(size * 0.55)}px`,
      lineHeight: 1,
      color: symbolColor,
      textShadow: color === 4 ? 'none' : '0 1px 2px rgba(0,0,0,0.4)',
      pointerEvents: 'none',
    }}>
      {COLOR_SYMBOLS[color]}
    </span>
  );

  const sharedStyle: React.CSSProperties = {
    width: `${size}px`,
    height: `${size}px`,
    borderRadius: '4px',
    background: COLOR_HEX[color],
    border: '2px solid',
    borderColor,
    boxShadow: selected
      ? '0 0 6px 1px rgba(255,255,255,0.7)'
      : highlighted
        ? `0 0 4px 1px ${highlightColor}`
        : 'none',
    opacity: dimmed ? 0.35 : 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 0,
    boxSizing: 'border-box',
    ...style,
  };

  if (onClick) {
    return (
      <button
        type="button"
        onClick={onClick}
        title={title ?? COLOR_NAMES[color]}
        style={{ ...sharedStyle, cursor: 'pointer' }}
      >
        {content}
      </button>
    );
  }

  return (
    <div title={title ?? COLOR_NAMES[color]} style={sharedStyle}>
      {content}
    </div>
  );
}
