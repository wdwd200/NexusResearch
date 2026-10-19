const STORAGE_KEY = "nexus.thread_id";
const THREAD_ID_PATTERN =
  /^(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|[A-Za-z0-9][A-Za-z0-9_-]{0,63})$/;

export function isValidThreadId(threadId: string | null | undefined): threadId is string {
  return Boolean(threadId && THREAD_ID_PATTERN.test(threadId));
}

export function createThreadId(): string {
  if (crypto.randomUUID) {
    return crypto.randomUUID();
  }

  return `manual-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function getStoredThreadId(): string {
  const existing = window.localStorage.getItem(STORAGE_KEY);
  if (isValidThreadId(existing)) {
    return existing;
  }

  const threadId = createThreadId();
  window.localStorage.setItem(STORAGE_KEY, threadId);
  return threadId;
}

export function storeThreadId(threadId: string): void {
  window.localStorage.setItem(STORAGE_KEY, threadId);
}
