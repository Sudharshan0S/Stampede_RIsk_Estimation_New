# CrowdVision AI - Frontend

React + Vite + Tailwind + Recharts UI for the Stampede Risk Estimation backend. Responsive from small phones (320 px) to desktop, portrait and landscape.

```bash
npm install
cp .env.example .env        # VITE_API_BASE_URL = where the Flask backend runs
npm run dev                 # http://localhost:5173      (add  -- --host  to open it from a phone on the same Wi-Fi)
npm run build               # production build in dist/
```

Deploy on Vercel: import the repository, set **Root Directory = `frontend`**, add the environment variable `VITE_API_BASE_URL` (your deployed backend URL), deploy.
See the top-level `../README.md` for the full setup, model placement and backend deployment.
