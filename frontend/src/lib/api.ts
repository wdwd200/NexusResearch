import { ACCESS_TOKEN, API_BASE_URL, APP_BASE_PATH, WS_BASE_URL } from "./config";
import type { CancelTaskResponse, FileListResponse, TaskResponse, UploadResponse } from "../types";

// 相对部署（API_BASE_URL 为空）时，接口路径跟随应用 base 前缀；
// 显式配置了 VITE_API_BASE_URL 时以它为准，不做拼接。
function apiUrl(path: string): string {
  if (!API_BASE_URL && APP_BASE_PATH) {
    return `${APP_BASE_PATH}${path}`;
  }
  return `${API_BASE_URL}${path}`;
}

async function requestJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (ACCESS_TOKEN) {
    headers.set("Authorization", `Bearer ${ACCESS_TOKEN}`);
  }
  const response = await fetch(input, { ...init, headers });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message =
      typeof payload === "object" && payload && "detail" in payload
        ? String(payload.detail)
        : `HTTP ${response.status}`;
    throw new Error(message);
  }

  return payload as T;
}

export async function startTask(query: string, threadId: string): Promise<TaskResponse> {
  return requestJson<TaskResponse>(apiUrl("/api/task"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      query,
      thread_id: threadId
    })
  });
}

export async function cancelTask(threadId: string): Promise<CancelTaskResponse> {
  return requestJson<CancelTaskResponse>(apiUrl(`/api/task/${encodeURIComponent(threadId)}/cancel`), {
    method: "POST"
  });
}

export async function uploadSessionFiles(
  files: File[],
  threadId: string
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("thread_id", threadId);
  files.forEach((file) => formData.append("files", file));

  return requestJson<UploadResponse>(apiUrl("/api/upload"), {
    method: "POST",
    body: formData
  });
}

export async function listSessionFiles(threadId: string): Promise<FileListResponse> {
  const url = new URL(apiUrl("/api/files"), window.location.origin);
  url.searchParams.set("thread_id", threadId);
  return requestJson<FileListResponse>(url);
}

export function getDownloadUrl(threadId: string, path: string, ticket: string): string {
  const url = new URL(apiUrl("/api/download"), window.location.origin);
  url.searchParams.set("thread_id", threadId);
  url.searchParams.set("path", path);
  url.searchParams.set("ticket", ticket);
  return url.toString();
}

export function getWebSocketUrl(threadId: string): string {
  const path = apiUrl(`/ws/${encodeURIComponent(threadId)}`);
  const url = new URL(
    path,
    WS_BASE_URL || `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`
  );
  if (ACCESS_TOKEN) {
    url.searchParams.set("token", ACCESS_TOKEN);
  }
  return url.toString();
}

export interface ResumeResponse {
  status: string;
  thread_id: string;
  decision: string;
}

export interface TaskStatus {
  thread_id: string;
  running: boolean;
  awaiting_approval: boolean;
  actions: ApprovalAction[];
}

export interface ApprovalAction {
  tool: string;
  args: Record<string, unknown>;
  description: string;
}

export async function resumeTask(
  threadId: string,
  decision: "approve" | "reject",
  feedback = ""
): Promise<ResumeResponse> {
  return requestJson<ResumeResponse>(
    apiUrl(`/api/task/${encodeURIComponent(threadId)}/resume`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, feedback })
    }
  );
}

export async function getTaskStatus(threadId: string): Promise<TaskStatus> {
  return requestJson<TaskStatus>(
    apiUrl(`/api/task/${encodeURIComponent(threadId)}/status`)
  );
}
