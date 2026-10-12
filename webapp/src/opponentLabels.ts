export function opponentName(id: string, name: string): string {
  if (id === 'astra' || name === 'Astra Heuristic') return 'Astra Bot';
  if (id.startsWith('net:') || name === 'Trained Azul AI' || name === 'RL Trained Bot') return 'RL Bot';
  return name;
}

export function opponentDifficulty(id: string): string {
  if (id === 'random') return 'easy';
  if (id === 'heuristic' || id === 'opus') return 'medium';
  if (id === 'astra') return 'hard';
  return 'expert';
}
