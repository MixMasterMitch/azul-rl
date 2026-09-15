export interface LegalAction {
  index: number;
  name: string;
}

export type FloorSlot = number | 'first' | null;

export interface PlayerState {
  pattern_count: number[];
  pattern_color: number[];
  wall: boolean[][];
  floor_count: number;
  floor_slots: FloorSlot[];
  score: number;
}

export interface GameState {
  game_id: string;
  revision: number;
  status: 'active' | 'completed' | 'abandoned';
  move_number: number;
  can_abandon: boolean;
  created_at: string;
  updated_at: string;
  winner_seats: number[];
  opponent_names: string[];
  num_players: number;
  human_seat: number;
  opponents: string[];
  current_player: number;
  factories: number[][];
  center: number[];
  center_has_first: boolean;
  center_source: number;
  players: PlayerState[];
  legal_actions: LegalAction[];
  ended: boolean;
  winner: number | null;
}

export interface Opponent {
  id: string;
  name: string;
  supported_players: number[];
  kind: string;
  default: boolean;
  rating?: number;
  rating_2p?: number;
  rating_3p?: number;
  rating_4p?: number;
}

export interface Profile {
  username: string;
  games: number;
  wins: number;
  placed: boolean;
  placement_wins_required: number;
  rating: number | null;
}

export interface GameSummary {
  game_id: string;
  num_players: number;
  human_seat: number;
  status: GameState['status'];
  revision: number;
  updated_at: string;
  scores: number[];
  winner_seats: number[];
  opponent_names: string[];
}

export interface LeaderboardEntry {
  entity_id: string;
  kind: 'human' | 'agent';
  label: string;
  rating?: number;
  rating_2p?: number;
  rating_3p?: number;
  rating_4p?: number;
  games?: number;
}

export const COLOR_NAMES = ['Blue', 'Yellow', 'Red', 'Black', 'White'];
export const COLOR_HEX = ['#2196F3', '#FFC107', '#F44336', '#424242', '#FAFAFA'];
export const COLOR_ABBREV = 'BYRKW';
export const COLOR_SYMBOLS = ['✿', '✴', '●', '✦', '❄'];

export const FLOOR_TARGET = 5;
export const FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3];

export const WALL_PATTERN = [
  [0, 1, 2, 3, 4],
  [4, 0, 1, 2, 3],
  [3, 4, 0, 1, 2],
  [2, 3, 4, 0, 1],
  [1, 2, 3, 4, 0],
];

export function wallColumnForColor(row: number, color: number): number {
  return (color + row) % 5;
}

export interface TileSelection {
  source: number;
  color: number;
}

export const CELL = 22;
export const CELL_GAP = 2;
export const ROW_WIDTH = 5 * CELL + 4 * CELL_GAP;
