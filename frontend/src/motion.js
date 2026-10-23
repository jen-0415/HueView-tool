// Motion helpers. No dependency -- CSS keyframes in index.css do the work,
// this just decides timing and whether to animate at all.
//
// The whole motion budget is spent in one place: the arrival of a result.
// Geometry (bar widths, strip heights, axis markers) animates from zero to the
// measured position. Numbers are always rendered as text straight away, so
// nothing on screen is ever a value that isn't the real one.

import { useEffect, useState } from "react";

const QUERY = "(prefers-reduced-motion: reduce)";

export function prefersReducedMotion() {
  return typeof window !== "undefined" && window.matchMedia?.(QUERY).matches;
}

export function useReducedMotion() {
  const [reduced, setReduced] = useState(prefersReducedMotion);

  useEffect(() => {
    const mq = window.matchMedia?.(QUERY);
    if (!mq) return;
    const onChange = (e) => setReduced(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  return reduced;
}

// 24ms between neighbours, capped so a six-region table never takes longer to
// settle than the eye takes to reach the bottom of it.
export const STAGGER_MS = 24;
export const STAGGER_CAP_MS = 320;

export function stagger(i = 0) {
  return { "--measure-delay": `${Math.min(i * STAGGER_MS, STAGGER_CAP_MS)}ms` };
}

// True once per analysis, for the result-arrival sequence. Stage switches must
// not re-run it, so the Results screen holds this at the top and passes it
// down rather than each panel asking for itself.
export function useArrival() {
  const reduced = useReducedMotion();
  const [elapsed, setElapsed] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setElapsed(true), 1000);
    return () => clearTimeout(t);
  }, []);

  return !reduced && !elapsed;
}

// Applied to a measured bar or strip. Returns nothing when the sequence is
// over or motion is reduced, so the element just renders at its value.
export function measure(active, i = 0) {
  return active ? { className: "measure-in", style: stagger(i) } : { className: "", style: undefined };
}
