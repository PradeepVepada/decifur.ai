/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  /**
   * `/api/*` → FastAPI is proxied in Node by `app/api/[...path]/route.ts` (plus dedicated
   * `app/api/chat/stream` and `app/api/papers/upload`). Rewrites are not used so a dead API
   * returns a clear 502 JSON instead of an opaque Internal Server Error.
   */
};

export default nextConfig;
