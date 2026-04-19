/**
 * Run Next.js dev server with cwd = UI/ so PostCSS/Tailwind resolve correctly
 * (avoids unstyled pages when npm's cwd differs from the app root).
 */
const { spawnSync } = require("child_process");
const path = require("path");

const uiDir = path.resolve(__dirname, "..", "UI");
const npmCmd = process.platform === "win32" ? "npm.cmd" : "npm";
const r = spawnSync(npmCmd, ["run", "dev"], {
  cwd: uiDir,
  stdio: "inherit",
  env: process.env,
  shell: true,
});
process.exit(typeof r.status === "number" ? r.status : 1);
