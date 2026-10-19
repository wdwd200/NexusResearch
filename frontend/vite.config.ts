import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  // 子路径部署时通过环境变量 VITE_BASE_PATH 控制（例如 /nexus/）；
  // 未设置时仍是根路径，便于回退到 http://host/ 形态。
  const env = loadEnv(mode, ".", "");
  const basePath = env.VITE_BASE_PATH || "/";

  return {
    base: basePath,
    plugins: [react(), tailwindcss()],
    build: {
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (!id.includes("node_modules")) {
              return undefined;
            }
            if (
              id.includes("/react/") ||
              id.includes("/react-dom/") ||
              id.includes("/scheduler/")
            ) {
              return "react-vendor";
            }
            if (
              id.includes("/antd/") ||
              id.includes("/@ant-design/") ||
              id.includes("/rc-") ||
              id.includes("/@rc-component/")
            ) {
              return "antd-vendor";
            }
            return "vendor";
          }
        }
      }
    },
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target: "http://localhost:8002",
          changeOrigin: true
        },
        "/ws": {
          target: "ws://localhost:8002",
          ws: true
        }
      }
    }
  };
});
