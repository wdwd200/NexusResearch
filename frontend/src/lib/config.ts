// 生产环境由 nginx 同域反代，默认走相对路径；
// 本地 Vite 开发时由 vite.config.ts 的 proxy 转发到 8002。
// 部署在子路径时设置 VITE_API_BASE_URL=/nexus，请求会带上该前缀。
const DEFAULT_API_BASE_URL = "";

function stripTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function deriveWsBaseUrl(apiBaseUrl: string): string {
  if (import.meta.env.VITE_WS_BASE_URL) {
    return stripTrailingSlash(import.meta.env.VITE_WS_BASE_URL);
  }

  if (apiBaseUrl.startsWith("https://")) {
    return apiBaseUrl.replace(/^https:\/\//, "wss://");
  }

  if (apiBaseUrl.startsWith("http://")) {
    return apiBaseUrl.replace(/^http:\/\//, "ws://");
  }

  if (!apiBaseUrl) {
    return `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`;
  }

  return `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`;
}

export const API_BASE_URL = stripTrailingSlash(
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL
);

// 相对基址（空字符串）时，用 Vite 注入的 BASE_URL 作为前缀，
// 这样部署到 /nexus/ 子路径时，/api 与 /ws 会自动带上 /nexus 前缀。
// 根路径部署时 BASE_URL 是 "/"，前缀为空。
export const APP_BASE_PATH = stripTrailingSlash(import.meta.env.BASE_URL || "/");

export const WS_BASE_URL = deriveWsBaseUrl(API_BASE_URL);

// 可选演示口令：只放进请求头和 WebSocket 握手，不写入下载链接。
// 这是单进程共享口令，不是用户隔离。
export const ACCESS_TOKEN = import.meta.env.VITE_ACCESS_TOKEN || "";
