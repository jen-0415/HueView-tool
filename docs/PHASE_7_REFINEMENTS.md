# Draft — Note to Adviser

Four places where the implementation and the manuscript disagree, found while
building Phases 7.4 and 7.5. Each needs a decision about which side changes.

Edit freely — this is a starting draft, and the tone should be yours.

---

Good [morning/afternoon] Ma'am/Sir,

While implementing Phases 7.4 and 7.5 of HueView, I found four points where
the manuscript and the code do not line up. Three need a small revision to
Chapter 3 or the appendices; one was a bug in our code that I have already
fixed. I would like your guidance on all four before we proceed to Phase 8.

**1. HSV skin filtering thresholds are not specified in Chapter 3**

Stage 3 of Feature Selection states that localized HSV filtering removes
eyebrows, eyelashes, hair, eyes and lips, but it does not give the numeric
thresholds. Since these values directly determine which pixels are counted as
skin, a panelist could reasonably ask where they came from.

I implemented the widely-cited rule of Kolkur et al. (2017), *Human Skin
Detection Using RGB, HSV and YCbCr Color Models*: hue between 0° and 50°, and
saturation between 0.23 and 0.68. I added one departure — a Value floor of
0.15, which Kolkur's HSV rule does not include. It is needed because the
region mask zeroes everything outside the polygon, and pixels near zero
brightness have numerically unstable hue; without a floor, some near-black
border pixels pass the filter at random. The floor is deliberately low so it
does not disadvantage darker skin tones, which matters given the study's aims.

Proposed: add a threshold table to Stage 3 and add Kolkur et al. (2017) to
the references. The values are frozen in `configs/hsv_skin_thresholds.json`
so the paper and the code cannot drift apart.

**2. The six region setups differ between the build plan and the manuscript**

Appendix 3 evaluates the five individual regions (Forehead, Left Cheek, Right
Cheek, Nose Bridge, Jawline) plus Full Face, and Tables 19–27 are built to
that shape.

Our build plan instead lists "Forehead only, Cheeks combined, Nose Bridge
only, Jawline only, All Regions Combined, Full Face" — which adds a "Cheeks
combined" setup, treats "All Regions Combined" and "Full Face" as separate,
and drops the individual cheek setups.

I implemented the manuscript's version, since the results tables and the
Cochran's Q test in Phase 12 both assume six setups in that arrangement.

Proposed: correct the build plan to match the manuscript.

**3. Table 3's jawline coverage does not match the region we extract**

Table 3 gives the jawline as "bottom 20–35% height, center 40–60% width." In
practice the region is built as the convex hull of the face-oval landmarks
below the jaw cutoff, which spans close to the full width of the face.

Visually the extracted region looks anatomically correct — it follows the
mandible and clears the lips. It is Table 3's stated width that seems too
narrow to describe a jawline.

Proposed: update Table 3's jawline coverage to match what the landmarks
actually produce. I can supply a figure showing the extracted region.

**4. Cheek region definitions were incorrect in our code (already fixed)**

This one was our bug, not a manuscript issue, but it affects results so I want
to record it.

The landmark index groups for the left and right cheeks each combined
face-oval points from one side of the image with eye points from the other.
MediaPipe names its landmark groups by the subject's own anatomy, so its
"left eye" group is on the image-right; Table 3 names regions by image
position ("left 15–45% width"). Because the region mask is a convex hull of
the listed points, each cheek stretched from one edge of the face across to
the opposite eye. Both cheek masks ended up covering most of the face and
overlapping each other almost entirely.

This would have affected SOP3 directly: the Left Cheek and Right Cheek
columns in Appendix 3 would have reported nearly the same region twice, which
would have looked like a valid result rather than an error.

I corrected the derivation so each cheek is bounded by the lower eyelid on the
same side, the nose wing medially, the lip line below, and an inset ring of
mesh points laterally (so the region no longer extends onto the ear). All five
regions are now verified non-overlapping, both at the landmark level and
pixel-by-pixel. The corrected regions also sit closer to Table 3's stated
coverage areas than the previous ones did.

The affected step is Phase 7.3, so the regional outputs will need to be
regenerated before Phase 8 training begins. No earlier phase is affected.

I have before-and-after figures for item 4 and can bring them to our next
meeting.

Thank you,
Joanne

---

## Notes for you, not for the email

- Items 1–3 are decisions for your adviser. Item 4 is informational, but
  raising it is the right call: it changes results, and it is much better
  coming from you now than surfacing during a defense.
- Item 4 is also worth a sentence in your Discussion or Limitations section —
  a documented, corrected error strengthens the methodology rather than
  weakening it.
- If asked how it was caught: the Phase 7.4 checkpoint measures 7.3's masks as
  a side effect of filtering them, and a region-overlap check is part of that
  checkpoint. That is a good answer.
- Do not send item 4 until the corrected regions are confirmed visually on a
  clean reference face and `--overlap-only` reports near-zero overlap on real
  images. Claiming the fix works before verifying it is the one way this
  backfires.
