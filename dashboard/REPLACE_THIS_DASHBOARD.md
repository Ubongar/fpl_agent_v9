# Drop-in replacement instructions

This archive is designed to replace the **contents of your current dashboard directory**.

1. Back up your current dashboard directory.
2. Delete the old dashboard source files from that directory.
3. Extract this ZIP directly into that same directory.
4. Create/update `.env.local` from `.env.example`.
5. Run `npm ci`.
6. Run `npm run dev`.

Important: this ZIP deliberately does NOT include:
- `node_modules/`
- `.next/`
- `.env`
- `tsconfig.tsbuildinfo`

The new navigation exposes **Player Compare** directly. The route is:

`/compare`

You can also open it from Player Lab after selecting players, or from a Player X-ray via **Add to comparison desk**.
