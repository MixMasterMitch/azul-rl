import { LegalAction, COLOR_ABBREV } from './types';

const MAX_FACTORIES = 9; // action index for center in 4p games; 2p/3p use factory{N} instead

export interface ParsedAction {
  source: number;
  color: number;
  target: number;
}

export function parseActionName(name: string): ParsedAction | null {
  const match = name.match(/^pick\((factory(\d+)|center),([BYRKW])\)→(line(\d)|floor)$/);
  if (!match) return null;

  const source = match[2] !== undefined ? parseInt(match[2], 10) : MAX_FACTORIES;
  const color = COLOR_ABBREV.indexOf(match[3]!);
  const target = match[5] !== undefined ? parseInt(match[5], 10) : 5;

  if (color < 0) return null;
  return { source, color, target };
}

export function getValidTargets(
  actions: LegalAction[],
  source: number,
  color: number,
): Set<number> {
  const targets = new Set<number>();
  for (const action of actions) {
    const parsed = parseActionName(action.name);
    if (parsed && parsed.source === source && parsed.color === color) {
      targets.add(parsed.target);
    }
  }
  return targets;
}

export function findAction(
  actions: LegalAction[],
  source: number,
  color: number,
  target: number,
): LegalAction | undefined {
  return actions.find(action => {
    const parsed = parseActionName(action.name);
    return parsed?.source === source && parsed.color === color && parsed.target === target;
  });
}

export function getAvailableColorsAtSource(
  actions: LegalAction[],
  source: number,
): Set<number> {
  const colors = new Set<number>();
  for (const action of actions) {
    const parsed = parseActionName(action.name);
    if (parsed && parsed.source === source) {
      colors.add(parsed.color);
    }
  }
  return colors;
}
