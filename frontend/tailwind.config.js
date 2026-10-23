/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        blush: { DEFAULT: "#FCE9ED", soft: "#FDEFF2" },
        line: { DEFAULT: "#F3C9D6", soft: "#F8DDE4" },
        // One step darker than line-soft: for table hairlines and panel edges
        // that have to stay visible against a white panel on a blush ground.
        rule: "#E9C4CF",
        // ink-mid is for 12-13px data labels. ink-soft clears AA on blush
        // (4.87:1) but stops being readable at the sizes the tables use it at.
        ink: { DEFAULT: "#1B1416", mid: "#5A4248", soft: "#7A6168" },
        accent: { DEFAULT: "#C9145C", light: "#E8558D" },
        warm: { DEFAULT: "#F7EDC0", ink: "#7C6413" },
        cool: { DEFAULT: "#C6D6DD", ink: "#2F5261" },
        neutraltone: { DEFAULT: "#F1E4EA", ink: "#7A6168" },
        ok: { DEFAULT: "#1F7A52", bg: "#DFF3E8" },
      },
      fontFamily: {
        display: ["Fraunces", "Georgia", "serif"],
        sans: ["'IBM Plex Sans'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
      fontSize: {
        // The scale the redesign works in, ~1.25 ratio. Nothing below 13px for
        // prose or 12px for numerals -- the old 9-11px mono labels were the
        // worst legibility problem in the tool.
        data: ["0.75rem", { lineHeight: "1rem" }],        // 12 -- mono labels
        fine: ["0.8125rem", { lineHeight: "1.25rem" }],   // 13 -- captions, notes
        base: ["0.9375rem", { lineHeight: "1.65" }],      // 15 -- body
        lead: ["1.0625rem", { lineHeight: "1.55" }],      // 17 -- panel titles
        sub: ["1.25rem", { lineHeight: "1.4" }],          // 20
        title: ["1.625rem", { lineHeight: "1.25" }],      // 26 -- readout values
        head: ["2.125rem", { lineHeight: "1.15" }],       // 34 -- stage headings
        hero: ["3rem", { lineHeight: "1.05" }],           // 48
      },
      spacing: {
        // 8px baseline grid, so panel padding stops being ad-hoc.
        gutter: "1.5rem",
      },
      borderRadius: {
        panel: "1rem",
        readout: "0.875rem",
      },
      boxShadow: {
        // Layered micro-shadow for the readout tier: enough to lift a measured
        // value off the panel without reading as a drop shadow.
        readout: "0 1px 1px rgba(27,20,22,.03), 0 2px 4px rgba(27,20,22,.03), 0 6px 12px rgba(27,20,22,.03)",
        panel: "0 1px 2px rgba(27,20,22,.025)",
      },
      transitionTimingFunction: {
        settle: "cubic-bezier(.2,.8,.2,1)",
      },
    },
  },
  plugins: [],
};
