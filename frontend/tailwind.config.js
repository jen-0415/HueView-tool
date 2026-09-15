/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        blush: { DEFAULT: "#FCE9ED", soft: "#FDEFF2" },
        line: { DEFAULT: "#F3C9D6", soft: "#F8DDE4" },
        ink: { DEFAULT: "#1B1416", soft: "#7A6168" },
        accent: { DEFAULT: "#C9145C", light: "#E8558D" },
        warm: { DEFAULT: "#F7EDC0", ink: "#7C6413" },
        cool: { DEFAULT: "#C6D6DD", ink: "#2F5261" },
        neutraltone: { DEFAULT: "#F1E4EA", ink: "#7A6168" },
        ok: { DEFAULT: "#1F7A52", bg: "#DFF3E8" },
      },
      fontFamily: {
        display: ["'Playfair Display'", "Georgia", "serif"],
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
