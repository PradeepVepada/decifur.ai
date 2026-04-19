# Next.js UI (shadcn-style structure)

This folder contains the rebuilt UI using:

- Next.js App Router
- TypeScript
- Tailwind CSS
- `components/ui` structure compatible with shadcn conventions

## Why `/components/ui` matters

shadcn components and generated code assume a shared primitive/component location under `components/ui`.
Keeping this path avoids import drift and makes future `shadcn` component additions straightforward.

## Run

1. Install dependencies:

```bash
cd UI
npm install
```

2. Configure API URL:

```bash
cp .env.example .env.local
```

3. Start backend (from project root):

```bash
/opt/anaconda3/bin/python api.py
```

4. Start frontend:

```bash
cd UI
npm run dev
```

Open `http://localhost:3000`.

## Included pages

- `/` redirects to `/signin`
- `/signin` and `/signup` (wired to auth API)
- `/chat` full app UI (corpus/web mode, sources, conversations, memory, upload)
