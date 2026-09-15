export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

export async function api<T>(path: string, username: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(username ? { 'X-Azul-Username': username } : {}), ...options.headers },
  });
  let data;
  try { data = await response.json(); }
  catch { throw new ApiError('The server is unavailable. Please try again.', response.status); }
  if (!response.ok) throw new ApiError(data.error ?? 'Request failed', response.status);
  return data as T;
}
