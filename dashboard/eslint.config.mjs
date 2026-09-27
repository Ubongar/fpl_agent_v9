// Next.js 16 removed the built-in `next lint` command, so ESLint runs
// directly here. eslint-config-next now ships native flat-config exports
// (see its dist/core-web-vitals.js) rather than the old .eslintrc-shaped
// export, so it is imported directly -- no @eslint/eslintrc FlatCompat
// bridge needed. An earlier version of this file used FlatCompat, which is
// still documented in some older tutorials but crashes against this
// package's current export shape with an opaque "converting circular
// structure to JSON" error. This is the pattern in Next's own current docs
// (nextjs.org/docs/app/api-reference/config/eslint).
import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

export default defineConfig([
  ...nextVitals,
  ...nextTypescript,
  globalIgnores([".next/**", "node_modules/**", "next-env.d.ts"]),
]);
