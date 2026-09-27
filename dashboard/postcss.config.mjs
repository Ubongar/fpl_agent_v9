// Tailwind v4 moved its PostCSS plugin into a dedicated package, and now
// handles vendor-prefixing itself -- autoprefixer is no longer needed and
// listing it alongside @tailwindcss/postcss does nothing but slow the build.
/** @type {import('postcss-load-config').Config} */
const config = { plugins: { "@tailwindcss/postcss": {} } };
export default config;
