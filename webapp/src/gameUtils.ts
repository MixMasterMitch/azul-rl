import { GameState, TileSelection } from './types';

export function getPickCount(state: GameState, selection: TileSelection): number {
  if (selection.source === state.center_source) {
    return state.center[selection.color] ?? 0;
  }
  return state.factories[selection.source]?.[selection.color] ?? 0;
}
