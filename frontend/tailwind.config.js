/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      screens: {
        xs: "420px", // large phones; below this (e.g. iPhone SE / small Android) layouts stay single-column
      },
      fontFamily: {
        display: ["'Space Grotesk'", "system-ui", "sans-serif"],
        body: ["'Inter'", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      colors: {
        base: {
          950: "#05060c",
          900: "#0a0d17",
          850: "#0d1120",
          800: "#111527",
          750: "#151a2e",
          700: "#1a2036",
          600: "#232a45",
          500: "#333c5c",
        },
        accent: {
          DEFAULT: "#7c5cff",
          soft: "#9b82ff",
          dim: "#4d3aa8",
        },
        cyan: {
          glow: "#3ee6de",
        },
        risk: {
          low: "#2fd67c",
          medium: "#f2c94c",
          high: "#ff9a3c",
          critical: "#ff4d5e",
        },
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(124,92,255,0.25), 0 8px 30px -8px rgba(124,92,255,0.35)",
        panel: "0 1px 0 rgba(255,255,255,0.04) inset, 0 12px 40px -18px rgba(0,0,0,0.6)",
      },
      backgroundImage: {
        "grid-fade":
          "radial-gradient(circle at 20% 20%, rgba(124,92,255,0.12), transparent 45%), radial-gradient(circle at 80% 0%, rgba(62,230,222,0.08), transparent 40%)",
      },
    },
  },
  plugins: [
    // `coarse:` applies on touch screens only (finger = coarse pointer) -> used to guarantee 44px tap targets without enlarging desktop controls
    ({ addVariant }) => {
      addVariant("coarse", "@media (pointer: coarse)");
    },
  ],
};
