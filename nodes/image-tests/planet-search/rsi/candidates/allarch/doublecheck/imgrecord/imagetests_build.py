"""Build doublecheck/imagetests.json: one friend-safe record of every image test of the candidate cards.

Run from A/doublecheck:  ~/anaconda3/bin/python -I imgrecord/imagetests_build.py   (add --fp to print entry fingerprints)

An image test = a pass that measured a card at its predicted place in archive images the search never used.
Passes read (in the order they ran; a missing <pass>.json is skipped, so the script can be re-run as passes finish):
  deepframes, openimg, openimg2, deepframes2, openimg3, largeimg1, largeimg2, openimg4, largeimg3,
  then (added by imgrecord2) chanceimg, largeimg4, otherarch, openimg5,
  then (added by imgrecord3) otherarch2, chanceimg2, otherarch3a, otherarch3b, otherarch3c,
  then (added by imgrecord5) mask7c (its file is mask7c_imagetests.json: PASS_FILE), otherarch3d, otherarch3e (read as
  otherarch3a), otherarch3f, otherarch3h (as otherarch3b), otherarch3g, otherarch3i (as otherarch3c): READER
A pass with a json but no hand-back file is provisional (status "read (pass still running: no hand-back yet)").
A pass's verdict words that start with no label are mapped by WORDS (imgrecord3: chanceimg2's "STANDS (seen again, combined, ...)"
is SEEN AGAIN (combined); "no new night" is UNSURE, as imgrecord2 decided); words still not recognised go to UNSURE with a note.
If a pass lists one card twice (openimg4's form: in its `combined` section and as a card entry), the card entry is used.
Other *.json of this folder were looked at and left out (see imgrecord.md): none of them measures images at a predicted place.

Output: {"about", "built", "passes": [...], "cards": {"C###": [{"pass", "verdict", "evidence", "nights_seen", "nights_missed"}]},
         "counts", "disagreements", "notes"}, written atomically (tmp file, then os.replace).
  verdict: SEEN AGAIN / SEEN AGAIN (combined) / NOT SEEN / UNSURE / NO POWER, mapped from the pass's own words. A pass's identity
           result (KNOWN <body>, PROBABLY <body>) goes in the evidence, never in the verdict.
  nights_seen: new nights (not used by the search) on which the body was seen (for SEEN AGAIN (combined): the nights combined).
  nights_missed: nights with power (expected S/N 5 or more, as the pass counted them) on which it was not seen.
  evidence: one or two plain sentences, years only. Every text is checked (no MJD, no 5-digit numbers, no day or month dates,
           no month names, no clock times, no coordinates, no file names or paths, no lead ids); a text that breaks the rule
           stops the script with an error, nothing is written.
Evidence for the entries that existed when this script was written (2026-11-23) is hand-written from each pass's numbers
(HAND below). Each hand row carries a fingerprint of the pass's entry; if the pass rewrites that entry, the row falls back
to the automatic builder (first clean sentence of the pass's own text, else a sentence built from its numbers) and a note
says so. Entries added later (largeimg2's remaining cards, openimg4, largeimg3) go through the automatic builder, except four
hand rows added by imgrecord2 after a row-by-row check (largeimg3 C### and C###, largeimg2 C###, openimg4 C###), and the five
entries of otherarch2 and chanceimg2 (hand rows by imgrecord3).
ERRATA (imgrecord4): a fault found in a pass after it ended, by a later pass, is corrected here (evidence only; verdict and nights
kept; the pass's json is never edited): otherarch2 C###, whose 2014 night was measured at a wrong place (found by ps1joint).
ps1joint itself gets no row (imgrecord4: its one new measurement, C###'s 2014 night at the right place, is a single night below
S/N 5 that would only be UNSURE, and it is quoted in the corrected otherarch2 row; C### got no new image measurement).
imgrecord5 (2026-11-23): mask7c's four rows are hand rows holding the pass's own evidence text (its automatic sentence would be
the verdict's bracket only, with no numbers); one more erratum: openimg3 C### loses its stale last sentence on the joint orbit.
imgrecord6 (2026-11-23): mask7d and ps1joint3 added (files mask7d_imagetests.json, ps1joint3_imagetests.json: PASS_FILE), all
five rows hand rows from the passes' own texts (ps1joint3 C### shortened to the length limit). ps1joint3's rows supersede the
Pan-STARRS1 at-place rows of C### (otherarch3c) and C### (otherarch3b); those stay in the table, corrected through ERRATA, which
may now also set the verdict and nights_missed: C### NO POWER (was NOT SEEN, 20 nights missed -> 0), C### a sentence added,
C### a sentence added (stays UNSURE), C### NO POWER (about 3 expected, under the S/N 5 bar of its pass's own rule).
imgrecord7 (2026-11-23): ps1wide, ps1wide2, ps1wide3, ps1wide4 (files <pass>_imagetests.json, ps1joint3's format) and otherarch3j
(otherarch3b's format) added. These are read ONLY once <pass>.handback.md exists (NEED_HANDBACK; until then the pass is skipped,
status "not read yet ..."), so the table never holds their half-written files. Their rows without a hand row take the pass's own
evidence text fitted to the rule by whole sentences (fit_text: the whole text, else its sentences merged into two, else the first
sentence with the last or another one, else the first sentence, else its leading clauses). `supersedes` now works by rule
(apply_supersedes), so the passes still running need no hand-written erratum: the superseded row stays; if the new row is SEEN AGAIN
(either form) and the old one NOT SEEN, the old one becomes NO POWER (nights_missed 0: a test of the wrong place had no power) and
gets one sentence: the at-place test measured the place of the stored orbit and the body is elsewhere (the new row's `why`, cut to
fit), see the <pass> row; in every other case the old verdict stays and the row gets "see the <pass> row". A hand-written ERRATA
entry that is applied to the same row wins over the rule. ps1wide's C###, C### and C### are hand rows (its texts are over the
length limit; shortened by hand, every number kept); its C### text fits whole.
imgrecord8 (2026-11-23): ps1wide5-ps1wide9 added (files <pass>_imagetests.json, ps1wide's format; read only once their hand-back
exists). Two later-pass corrections by rule, from the passes' own json (never typed by hand), each read only once its hand-back
exists: (1) duplicate warps (DUP_PASSES: ps1dup.json, then ps1dup2.json in the same format, a later file winning): a row whose
pass and card ps1dup audited, and whose at-place value it corrects, carries the corrected value and names the old one ("5.36
(5.95 before duplicate warps were dropped)"), in place where the old value is in the text, else in a clause; the copies count
when it changed and ps1dup's word for the row (DUP_WORDS). Verdict: FALLS turns a SEEN AGAIN row UNSURE (nights_seen 0); every
other word (STANDS, WEAKER BUT STANDS, CHANGES UP: the pass proposed no promotion, NEEDS A RE-RUN) keeps the verdict. Applied
right after the ERRATA step; a row an applied erratum wrote is left alone (the hand entry wins). (2) ps1night (NIGHT_PASS): for
every card with a failing Pan-STARRS1 year, one sentence on the row its year places came from (the joint pass's row, else
otherarch2's for the ps1joint files, else the card's newest Pan-STARRS1 row), after the supersedes rule: "Later (ps1night pass):
the Pan-STARRS1 year place(s) of YYYY were left out of the orbit (reason); the whole-arc sighting is not changed by this."
No verdict change. Both logs go to the top-level keys "duplicate_warps" and "ps1night".
imgrecord9 (2026-11-23): (1) ps1dup2.json is read as it is (never edited): its `cards` map a card to a LIST of entries whose
`before` / `after` give `copies` ("10 of 240 (copy max 252.21)") and `halves` ("8.35 (12) / 8.45 (6)") as strings; dup_norm
turns them into ps1dup's shape in memory, and an entry naming two passes ("otherarch2 + ps1joint") names the row of each. The
entry's verdict words are cut to their leading word (dup_word). Verdict by rule (dup_verdict): STANDS, WEAKER BUT STANDS and
every other word keep the row's verdict; FALLS takes the combined-statistic rule on the corrected numbers (SEEN AGAIN (combined)
needs fewer than 0.3 % of at least 200 shifted copies reaching the measured value and no single night above 60 % of it; else
UNSURE). Where ps1dup and ps1dup2 name the same row, ps1dup2 (the later re-measurement) wins (DUP_BOTH, top-level key
"duplicate_warps_both"). A FALLS row superseded by a SEEN AGAIN wide-search row gets "The wide search <pass> confirms the card
independently, so its stage does not change." in place of "see the <pass> row" (falls_sentence). (2) Orbit sentences by rule
from ps1dup2.json (orbit_sentences; years only): a card with a joint file in `joint_files_written` gets "Later (ps1dup2 pass):
the orbit now holds the <years> Pan-STARRS1 places; the <year> place was left out (one night carries <n> % of it)" (years from
the two files' `joint_with`, the share from `one_night_failures`), or, when no year was left out, "the <year> place moved <x>
arcsec after duplicate warps were dropped; orbit refit, all <n> years pass"; a card whose pass's file is `fit_to_install`
false gets "the <pass> orbit file was found not fit to install, and the installed orbit stands". On the row of the entry's
pass. (3) ps1wide10 added (read only once its hand-back exists). (4) Rows made by the builder (BUILDER_ROWS): where a wide
pass's own <pass>.json states a verdict for a card but the pass wrote no image row in <pass>_imagetests.json, the row is made
from that json (verdict, one sentence of its evidence, years only) and carries "made_by". Log: top-level key "builder_rows".
imgrecord10 (2026-11-24): (1) u20img and allarch20 added (read only once their hand-backs exist; FIT_TEXT). u20img's image rows
(u20img_imagetests.json) are per-card dicts in ps1wide's shape. allarch20's (allarch20_imagetests.json) are keyed "C###:<archive>"
(one per card and archive); entries_of joins a card's archives into one entry (archive_entry: the strongest label leads, every
other archive is joined to its last sentence as "<archive>: <label> (<its text>)"), so the table keeps one row per pass and card.
Its verdict words NOT CONFIRMED (a track that beats the copies but fails a rule, or one night only) are UNSURE (WORDS, as ps1wide10
itself wrote them in its image rows), with 0 nights seen (nights_seen counts SEEN AGAIN nights; the text says one night). (2) The
missing-row rule (builder_rows) reads a pass's per-card entries at the top level of <pass>.json or in its "cards" section
(pass_cards; ps1wide10, u20img and allarch20 keep them there), and a key that is not a card id is mapped through ALIAS, checked on
the card (u20img's "C###-split" is the card C###). For a verdict other than NO POWER the one sentence is the pass's concluding
"So ..." sentence, else its first sentence that is not the method preamble. Rows made from imgrecord10 on say so in "made_by".
(3) ERRATA for three ps1wide4 rows whose duplicate-warp text the length limit had cut (C###: halves 3.11 with 6 of 240 copies,
the halves pass only under the static-peak rule; C###: 3 of 240 copies, not 4; C###: the verdict run's later value 7.61 shown
beside the first statistic 9.96). orbit_sentences adds no sentence a row already carries.
imgrecord11 (2026-11-24): allarch21 and u20img2 added (both launched 04:33, allarch21 first; read only once their hand-backs
exist; FIT_TEXT, BUILDER_ROWS). allarch21_imagetests.json is in allarch20's shape ("C###:<archive>" keys, joined per card by
archive_entry); its "NOT SEEN (wide)" is NOT SEEN by label. u20img2_imagetests.json holds per-card dicts in u20img's shape (its
C### is keyed "C###", no alias). A later pass that re-measures a card and archive keeps its own row beside the earlier one; the
earlier row is superseded only where the later pass says so in its `supersedes`, which may now carry words after the pass name
("allarch20 UNSURE"): sup_names keeps the leading pass name (allarch21 C### and C### supersede allarch20's rows; every older
`supersedes` is a bare pass name and reads as before). Hand rows where the fitted text loses the result or the caveat
(allarch21 C###: the NOT SEEN (wide) family is the linear two-body family of a short arc carried far back, so it does not
refute the card).
imgrecord12 (2026-11-24): allarch22 added (launched 05:51; read only once its hand-back exists; FIT_TEXT, BUILDER_ROWS).
allarch22_imagetests.json is in allarch20's shape ("C###:<archive>" keys, joined per card by archive_entry, one row per pass and
card); it writes several parts for one card (each archive alone, then "combined ..."). Its verdict word CONFIRMED (in this shape
only: part_label) is SEEN AGAIN (combined) when the combined-statistic rule holds on the part's own numbers (confirmed_rule:
fewer than 0.3 % of at least 200 shifted copies reach the value, no night above 60 %, each half under 0.3 % of the copies, at
least 2 nights), else UNSURE; every CONFIRMED part read is logged in the notes. ps1wide*.json also use the word CONFIRMED, and
those files are left as they were (WORDS is unchanged, so their builder rows do not change). The other parts keep their own
verdicts (NOT CONFIRMED is UNSURE, as before) and are named in the row's text. Hand rows where the joined or fitted text loses
the result (allarch22 C###, C###).
imgrecord13 (2026-11-24): ps1night2 added (launched 17:08; read only once its hand-back exists; FIT_TEXT; not BUILDER_ROWS: the
per-card verdicts of ps1night2.json, STANDS / FALLS, judge the joint orbits, not images). ps1night2_imagetests.json holds
ps1wide6's fields under allarch20's "C###:<archive>" keys, one part per Pan-STARRS1 season ("Pan-STARRS1 3pi 2011 year stack"),
joined per card by archive_entry (one row per pass and card). Each part is the one-night test of one season inside an earlier
whole-arc sighting (SEASON_PASSES, season_label): a season that passes carries the pass's own verdict "SEEN AGAIN (its <year>
Pan-STARRS1 year stack on its own)", read as SEEN AGAIN (combined), because a year stack is several nights together, no night
alone, as every other Pan-STARRS1 year-stack row of the table is labelled (nights_seen 0: those nights were already in the
earlier sighting). A season that fails keeps the pass's own word, UNSURE, and is never read as NOT SEEN (a failed one-night test
does not show the body absent; ps1night's failing years changed no verdict either). The earlier whole-arc rows of these cards
(ps1joint3, ps1wide, ps1wide2) are left as they are: no ps1night-style sentence is added to them, since the ps1night2 row on the
same card names its failing years. Every season part read is logged in the notes. Hand rows for all four cards (the joined
text is over the length limit), every number from the pass's own text.
imgrecord14 (2026-11-24): a season part is SEEN AGAIN (combined) only when the pass's word is SEEN AGAIN AND the
combined-statistic rule holds on its peak numbers with the static-peak mask on, read from its own text (season_rule: fewer than
0.3 % of at least 200 shifted copies reach the peak, and no night above 60 % of it); else UNSURE. The pass's PASS is its
clear-peak entry test (at most 24 of 240 copies), looser than that rule, and a test that fails the pre-fixed rule is not rescued
by a looser one. The at-track reading is not used: the track is the old joint orbit, fitted to these same Pan-STARRS1 places, so
it has search freedom built in. The row is SEEN AGAIN (combined) if any season is, else UNSURE (archive_entry's order). C###'s
2013 season (1 of 240 copies, 0.42 %) is UNSURE, so its row is UNSURE; its hand row says why. A hand row of a season pass whose
verdict differs from the rule's label is not used (row_for; a note asks for a look by hand).
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # A/doublecheck
R = "$ALE_HOME/rsi/decam-big-tno"
OUT = os.path.join(HERE, "imagetests.json")

PASSES = [  # (name, what it did), in the order the passes started
    ("deepframes", "identity tracks of 6 cards measured in public DECam images no run of ours used"),
    ("openimg", "one-night and short-arc cards measured in public DECam images outside the card's search unit"),
    ("openimg2", "openimg continued, with mask value 7 counted as good pixels"),
    ("deepframes2", "deepframes continued: re-measured with mask value 7 kept, C### measured, joint orbits"),
    ("openimg3", "cards sent to a collaborator measured in public DECam images"),
    ("largeimg1", "large cards not yet tested in other images (even positions)"),
    ("largeimg2", "large cards not yet tested in other images (odd positions), then C###, C###, C###"),
    ("openimg4", "openimg3 continued: combined-statistic test of C### and C###, third epochs"),
    ("largeimg3", "largeimg1 continued: C### combined test, C###, C###, C###'s Subaru HSC night"),
    # added by pass imgrecord2 (2026-11-23): started 06:55, read when their files appear (missing: "not written yet";
    # json but no hand-back: provisional, "read (pass still running: no hand-back yet)"). Per-card entries with the keys of
    # largeimg3.json (chanceimg, largeimg4, otherarch) or openimg4.json (openimg5: refits / combined sections + per-card entries).
    ("chanceimg", "cards whose detections the whole-set chance test could not tell from a chance linkage, measured in archive images"),
    ("largeimg4", "largeimg2 and largeimg3 continued: C###'s seasons tied through its image sighting, C###, C###, C###, C###, C###"),
    ("otherarch", "cards with no DECam exposure that could test them, looked for in other telescopes' public archives"),
    ("openimg5", "openimg4 continued: C###'s own-year exposures, C### refitted through its 2018 pair, C###'s faint 2014 night"),
    # added by pass imgrecord3 (2026-11-23), in the order they started (otherarch2 07:50, chanceimg2 08:21, otherarch3a-c 09:24).
    # otherarch3a/b/c write per-card entries with the keys of otherarch2.json; read when their files appear, as above.
    # (c275joint and ps1joint write orbit files only: record rows, no image-test rows.)
    ("otherarch2", "otherarch continued: the Pan-STARRS1 chain checked on three catalogued bodies, C###'s y band and joint orbit, "
                   "then C### and C### in Pan-STARRS1"),
    ("chanceimg2", "chanceimg continued: tries to break C###'s combined sighting (split by year and band, orbit variants, stack "
                   "shape), C###'s 2018 core season"),
    ("otherarch3a", "otherarch2 continued: the Pan-STARRS1-testable cards, first third"),
    ("otherarch3b", "otherarch2 continued: the Pan-STARRS1-testable cards, second third"),
    ("otherarch3c", "otherarch2 continued: the Pan-STARRS1-testable cards, last third"),
    # added by pass imgrecord5 (2026-11-23), in the order they started (mask7c 10:44; otherarch3d-g 11:43; 3h, 3i later).
    # mask7c's image rows are in mask7c_imagetests.json (PASS_FILE; its mask7c.json classes 99 cards and is not an image
    # test file); the otherarch3 continuations are read with their source pass's reader (READER), when their files appear.
    ("mask7c", "negative image verdicts and list-based non-detections re-measured with the pipeline's transient flag "
               "(value 7) kept: C###, C###, C###, C###"),
    ("otherarch3d", "otherarch3a continued: the Pan-STARRS1-testable cards it did not reach"),
    ("otherarch3e", "otherarch3a continued: the Pan-STARRS1-testable cards it did not reach"),
    ("otherarch3f", "otherarch3b continued: the Pan-STARRS1-testable cards it did not reach"),
    ("otherarch3g", "otherarch3c continued: the Pan-STARRS1-testable cards it did not reach"),
    ("otherarch3h", "otherarch3b continued: the Pan-STARRS1-testable cards it did not reach"),
    ("otherarch3i", "otherarch3c continued: the Pan-STARRS1-testable cards it did not reach"),
    # added by pass imgrecord6 (2026-11-23), appended in the order they started (mask7d 11:50, ps1joint3 12:25; the otherarch3
    # continuations keep the places imgrecord5 gave them). Both write their image rows to <pass>_imagetests.json (PASS_FILE),
    # per-card dicts the generic reader takes as they are; ps1joint3's extra keys `supersedes` and `why` are not shown on the card
    # (the superseded rows are corrected through ERRATA below instead).
    ("mask7d", "mask7c continued with value 7 kept: C###'s and C###'s own-night extras, C###'s 2020 week on its own"),
    ("ps1joint3", "independent-halves test of ps1look's offset Pan-STARRS1 sightings of C### and C###"),
    # added by pass imgrecord7 (2026-11-23), in the order they started (ps1wide 13:35, ps1wide2 13:36, otherarch3j 13:42,
    # ps1wide3 13:51, ps1wide4 14:24); each is read only once its hand-back exists (NEED_HANDBACK)
    ("ps1wide", "wide Pan-STARRS1 search over the orbits each card's core seasons allow, expectation from the card's own "
                "detections, independent halves: C###, C###, C###, C###, C###, C###"),
    ("ps1wide2", "ps1wide continued on other cards (from otherarch3f and otherarch3g, then a tail of otherarch3e's)"),
    ("otherarch3j", "otherarch3b continued: the Pan-STARRS1-testable cards otherarch3e-3g did not reach"),
    ("ps1wide3", "ps1wide continued on other cards (from otherarch3d and otherarch3e)"),
    ("ps1wide4", "ps1wide continued on other cards (from otherarch3h and otherarch3i)"),
    # added by pass imgrecord8 (2026-11-23), in the order they started (ps1wide5 15:14, ps1wide6 15:18, ps1wide7-9 15:41; all
    # stopped at 15:47 and queued again); each is read only once its hand-back exists (NEED_HANDBACK)
    ("ps1wide5", "ps1wide continued on other cards (wide search, one warp per exposure)"),
    ("ps1wide6", "ps1wide continued on other cards (from otherarch3j): C###, C###, C###, C###, C###"),
    ("ps1wide7", "ps1wide continued on other cards (wide search, one warp per exposure)"),
    ("ps1wide8", "ps1wide continued on other cards (wide search, one warp per exposure)"),
    ("ps1wide9", "ps1wide continued on other cards (wide search, one warp per exposure)"),
    # added by pass imgrecord9 (2026-11-23; ps1wide10 started 21:55): read only once its hand-back exists (NEED_HANDBACK)
    ("ps1wide10", "ps1wide continued on eleven u20 cards (wide search, one warp per exposure)"),
    # added by pass imgrecord10 (2026-11-24; u20img launched 22:35, allarch20 23:28, both restarted 03:04 after the reboot):
    # read only once their hand-backs exist (NEED_HANDBACK)
    ("u20img", "DECam images of u20 cards, value 7 kept: second epochs of C###, C###, C###, C###, the season C### dropped (C###), "
               "and ps1wide9's and ps1wide8's candidate orbits of C### and C### as predictions"),
    ("allarch20", "other archives for the u20 cards (ESO raw, SDSS; CFHT MegaCam and Subaru only from cached listings), image "
                  "tests with a control; one row per card, its archives joined"),
    # added by pass imgrecord11 (2026-11-24; both launched 04:33, allarch21 first): read only once their hand-backs exist
    ("allarch21", "allarch20 continued with the complete CFHT listing: CFHT MegaCam tests and family searches, C###'s ESO "
                  "OmegaCAM family search, C###, C### and C### added; one row per card, its archives joined"),
    ("u20img2", "u20img continued in DECam images, value 7 kept: C###'s own 2014 nights and second epoch, C###'s 2020 and 2022 "
                "family search, C###'s copy test without fixed sources, C###'s remaining years, C###'s installed orbit in "
                "2023-2025"),
    # added by pass imgrecord12 (2026-11-24; allarch22 launched 05:51): read only once its hand-back exists
    ("allarch22", "allarch21 continued: C### in public ESO VIKING tiles and combined with KiDS and CFHT on one orbit family, "
                  "C### in ESO VIKING / VHS tiles and KiDS OmegaCAM i, C### and C### wide family searches in CFHT MegaCam; one "
                  "row per card, its archives joined"),
    # added by pass imgrecord13 (2026-11-24; ps1night2 launched 17:08; u20mpc24 and mpc24look, which ended before it, made
    # identity and same-night checks and wrote no image rows): read only once its hand-back exists
    ("ps1night2", "the one-night test ps1night could not do from disk: Pan-STARRS1 year stacks of C###, C###, C### and C###, "
                  "each season on its own at the installed joint orbit's track (warps fetched again, one per exposure); one "
                  "row per card, its seasons joined"),
]
# imgrecord5: a pass whose image rows are not in <pass>.json, and passes read with another pass's reader (entries_of)
PASS_FILE = {"mask7c": "mask7c_imagetests.json", "mask7d": "mask7d_imagetests.json",  # imgrecord6: mask7d, ps1joint3
             "ps1joint3": "ps1joint3_imagetests.json",
             "ps1wide": "ps1wide_imagetests.json", "ps1wide2": "ps1wide2_imagetests.json",  # imgrecord7
             "ps1wide3": "ps1wide3_imagetests.json", "ps1wide4": "ps1wide4_imagetests.json"}
PASS_FILE.update({p: p + "_imagetests.json" for p in ("ps1wide5", "ps1wide6", "ps1wide7", "ps1wide8", "ps1wide9")})  # imgrecord8
PASS_FILE["ps1wide10"] = "ps1wide10_imagetests.json"  # imgrecord9
PASS_FILE.update({"u20img": "u20img_imagetests.json", "allarch20": "allarch20_imagetests.json"})  # imgrecord10
PASS_FILE.update({"allarch21": "allarch21_imagetests.json", "u20img2": "u20img2_imagetests.json"})  # imgrecord11
PASS_FILE["allarch22"] = "allarch22_imagetests.json"  # imgrecord12
PASS_FILE["ps1night2"] = "ps1night2_imagetests.json"  # imgrecord13
READER = {"otherarch3d": "otherarch3a", "otherarch3e": "otherarch3a", "otherarch3f": "otherarch3b",
          "otherarch3h": "otherarch3b", "otherarch3g": "otherarch3c", "otherarch3i": "otherarch3c",
          "otherarch3j": "otherarch3b"}  # imgrecord7
# imgrecord7: passes read only once <pass>.handback.md exists (these agents write their json as they go)
NEED_HANDBACK = {"ps1wide", "ps1wide2", "ps1wide3", "ps1wide4", "otherarch3j",
                 "ps1wide5", "ps1wide6", "ps1wide7", "ps1wide8", "ps1wide9",  # imgrecord8: ps1wide5-9
                 "ps1wide10",  # imgrecord9
                 "u20img", "allarch20",  # imgrecord10
                 "allarch21", "u20img2",  # imgrecord11
                 "allarch22",  # imgrecord12
                 "ps1night2"}  # imgrecord13
# imgrecord7: passes whose rows (without a hand row) take the pass's own evidence text, fitted to the rule by fit_text
FIT_TEXT = {"ps1joint3", "ps1wide", "ps1wide2", "ps1wide3", "ps1wide4",
            "ps1wide5", "ps1wide6", "ps1wide7", "ps1wide8", "ps1wide9", "ps1wide10",  # imgrecord8, imgrecord9
            "u20img", "allarch20",  # imgrecord10
            "allarch21", "u20img2",  # imgrecord11
            "allarch22",  # imgrecord12
            "ps1night2"}  # imgrecord13
# imgrecord9: wide passes whose own <pass>.json may state a verdict for a card that has no row in <pass>_imagetests.json
BUILDER_ROWS = ("ps1wide", "ps1wide2", "ps1wide3", "ps1wide4", "ps1wide5", "ps1wide6", "ps1wide7", "ps1wide8", "ps1wide9",
                "ps1wide10", "u20img", "allarch20",  # imgrecord10: u20img, allarch20
                "allarch21", "u20img2",  # imgrecord11
                "allarch22")  # imgrecord12
# imgrecord10: the pass whose rule made a builder row (shown in "made_by"); passes not listed: imgrecord9
BUILDER_BY = {"ps1wide10": "imgrecord10", "u20img": "imgrecord10", "allarch20": "imgrecord10",
              "allarch21": "imgrecord11", "u20img2": "imgrecord11", "allarch22": "imgrecord12"}
# imgrecord10: a pass's key that is not a card id -> (card, a word the card's key must hold); checked on R/candidates/<card>
ALIAS = {"C###-split": ("C###", "split")}  # u20img: the 2021 season C### dropped became card C### (C/NOTES.md, 22:40)
# imgrecord8: later passes that correct rows by rule (read only once <pass>.handback.md exists)
DUP_PASSES = ["ps1dup", "ps1dup2"]  # duplicate warps; ps1dup2.json in ps1dup's format; a later file wins for the same row
NIGHT_PASS = "ps1night"  # one-night / clear-peak / fixed-source test of every Pan-STARRS1 year place of the joint files
# imgrecord13: passes whose "C###:<archive>" parts are one-season tests inside an earlier whole-arc sighting (season_label)
SEASON_PASSES = {"ps1night2"}
PS1_PASSES = ("otherarch2", "otherarch3", "ps1")  # pass-name prefixes of the Pan-STARRS1 rows


def pass_path(name):
    """the file a pass's image rows are read from (PASS_FILE, else <pass>.json in this folder)"""
    return os.path.join(HERE, PASS_FILE.get(name, name + ".json"))

LABELS = ["SEEN AGAIN (combined)", "SEEN AGAIN", "NOT SEEN", "UNSURE", "NO POWER"]
# a pass's own verdict words that do not start with a label -> the label of the same meaning (imgrecord3). Tried only when the
# text starts with no label (a text starting "NOT SEEN (2017 and the 2018 core season)" is NOT SEEN by label_of already).
WORDS = [
    (r"(?i)^\s*STANDS\s*\(seen again, combined", "SEEN AGAIN (combined)"),  # chanceimg2 C###: the combined sighting not broken
    (r"(?i)^\s*no new night\b", "UNSURE"),
    # imgrecord10: allarch20 (and ps1wide10's json): a track that beats the copies but fails a rule, or one night only;
    # ps1wide10 wrote these UNSURE in its own image rows
    (r"(?i)^\s*NOT CONFIRMED\b", "UNSURE"),  # largeimg3 / largeimg4 C###: no new night, earlier nights stand (imgrecord2's mapping)
]

# entries that are not image tests: refits of nights another pass measured, identity fits only
SKIP = {
    ("openimg2", "C###"): "refit of openimg's three nights, no new image measurement",
    ("deepframes2", "C###"): "identity fit against catalogued bodies, no image measurement",
    # ("largeimg3", "C###") was skipped here as a refit while that pass ran; its final entry judges the 2020 Subaru HSC night an
    # independent sighting (SEEN AGAIN, 2 nights), so imgrecord2 gives it a row (HAND below).
}
SECTIONS_SKIP = {"refits", "joint", "proposed", "pass", "about", "built", "_rule", "notes", "summary"}

# ---------------------------------------------------------------- hand rows: (verdict, nights_seen, nights_missed, evidence)
# None for verdict / nights = take them from the pass entry.
HAND = {}  # release: 115 per-candidate hand records removed (per-body results); the builder code below is unchanged

# ---------------------------------------------------------------- errata (imgrecord4)
# A fault found in a pass AFTER it ended, by a later pass. The pass's own json is never edited; the row keeps its verdict and nights
# and only its evidence is replaced by the text below. Applied only while the pass entry still has the fingerprint the erratum was
# written against (FP_ERRATA); if the pass rewrites that entry, the erratum is not applied and a note asks for a look by hand.
#   "sources": files (relative to this folder) holding every number of the corrected text (read by imgrecord2/check_rows.py).
ERRATA = {}  # release: 9 per-candidate hand records removed (per-body results); the builder code below is unchanged
FP_ERRATA = {}  # release: 9 per-candidate hand records removed (per-body results); the builder code below is unchanged

# fingerprints of the pass entries the HAND rows were written from (sha1 of the entry's JSON, first 12 hex)
FP = {}  # filled below from FP_TEXT

FP_TEXT = ""  # release: 117 per-candidate fingerprint rows removed (per-body results)

# ---------------------------------------------------------------- the evidence rule
MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
          "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
BAD = [
    (r"(?i)\bMJD\b", "MJD"),
    (r"(?i)\bJD\b", "JD"),
    (r"(?<![\d,.])\d{5,}(?!\d)", "5-digit number (day number, exposure number)"),
    (r"\b(19|20)\d\d[-/.](0?[1-9]|1[0-2])\b", "year-month or decimal-year date"),
    (r"\b\d{4}-\d{1,2}-\d{1,2}\b", "full date"),
    (r"\b(19|20)\d\d\.\d+", "decimal-year date"),
    (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "full date"),
    (r"\b(" + MONTHS + r")\b", "month name"),
    (r"(?i)\b(january|february|april|june|july|august|september|october|november|december)\b", "month name"),
    (r"\b\d{1,2}(st|nd|rd|th)\b", "day of month"),
    (r"\b\d{1,2}:\d{2}\b", "clock time or sexagesimal"),
    (r"\b\d{1,3}[hd°]\s?\d{1,2}[m'′]", "sexagesimal coordinate"),
    (r"°", "degree sign"),
    (r"\bRA\b|(?i:right ascension|declination|\bdecl\b)", "coordinate word"),
    (r"\b[1-9]\d{1,2}\.\d{3,}\b", "coordinate-like number (2-3 digits, 3+ decimals)"),
    (r"\b(?!0\.)\d{1,3}\.\d{4,}\b", "coordinate-like number (4+ decimals)"),
    (r"(?i)\b\d+(\.\d+)?\s*(deg|degrees)\b", "angle in degrees"),
    (r"(?i)\b(expnum|exposure (number|no\.?|#)|c4d_)", "exposure number or name"),
    (r"(?i)\b[\w-]+\.(fits|fz|json|jsonl|py|md|txt|csv|log|sh|gz)\b", "file name"),
    (r"(?:[\w.-]+/){2,}", "path"),
    (r"\bdbt-|\brank\d+|\bgen\d{3}\b|\bp[01]\.\d{5}\b", "lead id"),
]


def check_text(t):
    probs = []
    if not t or not t.strip():
        return ["empty"]
    for rx, what in BAD:
        m = re.search(rx, t)
        if m:
            probs.append("%s (%r)" % (what, m.group(0)))
    nsent = len(re.findall(r"(?<=[a-z0-9\)\"%])[.!?](\s+|$)", t))
    if nsent > 2:
        probs.append("more than two sentences (%d)" % nsent)
    if len(t) > 560:
        probs.append("too long (%d chars)" % len(t))
    return probs


# ---------------------------------------------------------------- helpers
FP_KEYS = ("verdict", "nights_seen", "nights_expected_not_seen", "n_nights_seen", "evidence", "H1", "H0")


def fp(entry):
    """fingerprint of the fields a row is written from (passes add other fields, e.g. tables, without changing these)"""
    part = {k: entry.get(k) for k in FP_KEYS} if "verdict" in entry else entry
    return hashlib.sha1(json.dumps(part, sort_keys=True).encode()).hexdigest()[:12]


def num(x, keys=("n",)):
    """nights field -> int: int, {'n': k}, or a list of nights."""
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, dict):
        for k in keys:
            if k in x and isinstance(x[k], (int, float)):
                return int(x[k])
        return None
    if isinstance(x, list):
        return len(x)
    return None


def label_of(text):
    t = (text or "").strip()
    for lab in LABELS:
        if t.upper().startswith(lab.upper()):
            return lab
    return None


def word_label(text):
    """label for a pass's own verdict words that start with no label (WORDS), else None"""
    for rx, lab in WORDS:
        if re.match(rx, text or ""):
            return lab
    return None


def verdict_label(text):
    """the label a row gets from the pass's verdict text (label_of, then WORDS); None = not recognised"""
    return label_of(text) or word_label(text)


def years_of(e):
    ys = None
    for k in ("nights_seen", "seen"):
        v = e.get(k)
        if isinstance(v, dict) and isinstance(v.get("years"), list):
            ys = v["years"]
    if ys is None and isinstance(e.get("years"), list):
        ys = e["years"]
    if not ys:
        return ""
    ys = sorted({int(y) for y in ys if str(y).isdigit() and 1990 < int(y) < 2100})
    return " (" + ", ".join(str(y) for y in ys) + ")" if ys else ""


def first_sentence(text):
    if not isinstance(text, str):
        return None
    t = text.strip()
    parts = re.split(r"(?<=[a-z0-9\)\"%])\.\s+(?=[A-Z(])", t)
    s = parts[0].strip()
    if not s.endswith("."):
        s += "."
    for lab, nice in (("SEEN AGAIN (combined)", "Seen again (combined)"), ("SEEN AGAIN", "Seen again"),
                      ("NOT SEEN", "Not seen"), ("NO POWER", "No power"), ("UNSURE", "Unsure")):
        if s.upper().startswith(lab.upper()):
            s = nice + s[len(lab):]
            break
    return s


def fmt(x):
    """a pass's number as the pass wrote it (up to 2 decimals above 1; imgrecord2: was 1 decimal, which turned 5.98 into 6.0)"""
    if isinstance(x, float):
        if abs(x) < 0.01:
            return "%.2g" % x
        return ("%.3f" % x).rstrip("0").rstrip(".") if abs(x) < 1 else ("%.2f" % x).rstrip("0").rstrip(".")
    return str(x)


def built_from_numbers(verdict, e, ns, nm):
    ys = years_of(e)
    pl = lambda n: "" if n == 1 else "s"
    if verdict == "SEEN AGAIN (combined)":
        cb, ex, ch = e.get("combined_body"), e.get("expected"), e.get("chance_one_sided")
        s = "Seen only in the combination of %d night%s%s" % (ns or 0, pl(ns or 0), ys)
        if isinstance(cb, (int, float)):
            s += ": combined forced S/N %s" % fmt(float(cb))
            if isinstance(ex, (int, float)):
                s += " where %s was expected" % fmt(float(ex))
        s += "."
        if isinstance(ch, (int, float)):
            s += " Chance from shifted copies of the track %s." % fmt(float(ch))
        return s
    if verdict == "SEEN AGAIN":
        s = "Seen on %d new night%s%s at the predicted place." % (ns or 0, pl(ns or 0), ys)
    elif verdict == "NOT SEEN":
        if nm:
            return "Not seen on %d night%s where S/N 5 or more was expected." % (nm, pl(nm))
        # imgrecord3: was "Not seen on 0 nights ..." when the verdict rests on the nights combined (chanceimg C###, C###)
        s = "Not seen where it was expected; no single night had S/N 5 or more expected, so the verdict rests on the nights combined."
        cb, ex = e.get("combined_body"), e.get("expected")
        if isinstance(cb, (int, float)) and isinstance(ex, (int, float)):
            s = "Not seen: combined forced S/N %s where %s was expected; no single night had S/N 5 or more expected." % (
                fmt(float(cb)), fmt(float(ex)))
        return s
    elif verdict == "NO POWER":
        return "No usable archive exposure where the prediction is tight enough to test the body."
    else:
        s = "Measured at the predicted place, but no night decides."
        cb, ex = e.get("combined_body"), e.get("expected")
        if isinstance(cb, (int, float)) and isinstance(ex, (int, float)):
            s = "Combined forced S/N %s where %s was expected; not decisive." % (fmt(float(cb)), fmt(float(ex)))
    if nm:
        s += " Not seen on %d night%s with expected S/N 5 or more." % (nm, pl(nm))
    return s


def auto_evidence(verdict, e, ns, nm):
    for src in (e.get("verdict"), e.get("evidence")):
        s = first_sentence(src)
        if s and 40 <= len(s) <= 320 and not check_text(s):
            return s, "first sentence of the pass's own text"
    return built_from_numbers(verdict, e, ns, nm), "built from the pass's numbers"


# ---------------------------------------------------------------- imgrecord7: whole texts fitted to the rule, supersedes by rule
SENT_SPLIT = r"(?<=[a-z0-9\)\"%])\.\s+(?=[A-Z(])"
LOWER_OK = {"The", "Two", "Three", "Four", "Five", "One", "A", "No", "With", "Its", "Each", "Both", "Every", "Nights", "Fakes",
            "Without", "All", "In", "On", "At", "This", "That", "These", "Those", "It", "Here", "There"}


def sentences(t):
    """a text's sentences, each ending with its period"""
    t = re.sub(r"\s+", " ", (t or "").strip())
    out = [s.strip() for s in re.split(SENT_SPLIT, t) if s.strip()]
    return [s if s.endswith((".", "!", "?")) else s + "." for s in out]


def as_clause(s):
    """a sentence turned into a clause to follow '; ' (first word lower-cased only if it is a common word)"""
    s = s.strip().rstrip(".")
    w = s.split()[0] if s.split() else ""
    return (s[0].lower() + s[1:]) if w in LOWER_OK else s


def clauses(s, sep):
    """prefixes of s cut at `sep` outside parentheses, longest first (each without the separator)"""
    depth, cuts = 0, []
    for i, ch in enumerate(s):
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and s.startswith(sep, i):
            cuts.append(i)
    return [s[:i] for i in reversed(cuts)]


def first_ok(cands):
    """the first candidate text that passes the evidence rule, else None"""
    for t in cands:
        if t and not check_text(t):
            return t
    return None


def fit_text(text):
    """the pass's own text fitted to the rule (two sentences, 560 characters) by whole sentences and clauses -> (text, how)"""
    ss = sentences(text)
    if not ss:
        return None, None
    s1, rest = ss[0], ss[1:]
    tries = [(" ".join(ss), "the pass's own text, whole")]
    if len(rest) >= 2:
        tries.append((s1 + " " + rest[0].rstrip(".") + "; " + "; ".join(as_clause(x) for x in rest[1:]) + ".",
                      "the pass's own text, its later sentences joined into one"))
        tries.append((s1.rstrip(".") + "; " + "; ".join(as_clause(x) for x in rest[:-1]) + ". " + rest[-1],
                      "the pass's own text, its middle sentences joined to the first"))
    if rest:
        tries.append((s1 + " " + rest[-1], "the pass's own first and last sentences"))
        for sep in ("; ", ", "):  # the last sentence (usually the orbit) cut at a clause end
            for c in clauses(rest[-1].rstrip("."), sep):
                if len(c) >= 40 and c.count("(") == c.count(")"):
                    tries.append((s1 + " " + c + ".", "the pass's own first sentence and the leading clauses of its last"))
        for x in sorted(rest[:-1], key=len, reverse=True):
            tries.append((s1 + " " + x, "the pass's own first sentence and one more"))
    tries.append((s1, "the pass's own first sentence"))
    for c in clauses(s1.rstrip("."), "; "):
        tries.append((c + ".", "the pass's own first sentence, its leading clauses"))
    for t, how in tries:
        if not check_text(t):
            return t, how
    return None, None


def rowref(names):
    return "see the %s row" % names[0] if len(names) == 1 else \
        "see the %s and %s rows" % (", ".join(names[:-1]), names[-1])


def why_parts(why):
    """the new row's `why` (first sentence, no final period), then shorter cuts of it at clause ends, longest first"""
    ss = sentences(why)
    if not ss:
        return []
    w = ss[0].rstrip(".")
    out = [w] + clauses(w, "; ") + [c for c in clauses(w, ", ") if len(c) >= 40]
    return [x for x in out if x.count("(") == x.count(")")]


def no_power_text(old, why, names):
    """a NOT SEEN row superseded by a SEEN AGAIN row: first read as not seen + the at-place sentence (why cut to fit)"""
    ss = sentences(old)
    s1 = ss[0]
    if re.match(r"(?i)not seen\b", s1):
        s1 = "First read as not seen" + s1[len("not seen"):]
    heads = [s1]
    if len(ss) > 1:
        heads.insert(0, s1.rstrip(".") + "; " + "; ".join(as_clause(x) for x in ss[1:]) + ".")
    base = "But that at-place test measured the place of the stored orbit and the body is elsewhere"
    tails = ["%s (%s); %s." % (base, w, rowref(names)) for w in why_parts(why)] + ["%s; %s." % (base, rowref(names))]
    return first_ok([h + " " + t for h in heads for t in tails])


def see_row_text(old, names):
    """any other superseded row: verdict kept, 'see the <pass> row' added (as a sentence, else as a clause)"""
    ss = sentences(old)
    ref = rowref(names)
    cands = [" ".join(ss) + " " + ref[0].upper() + ref[1:] + ".", " ".join(ss).rstrip(".") + "; " + ref + "."]
    if len(ss) > 1:
        cands.append(ss[0] + " " + ref[0].upper() + ref[1:] + ".")
    return first_ok(cands)


def sup_names(sup):
    """imgrecord11: a pass's `supersedes` (a string or a list) -> the pass names it supersedes. A string whose first word is a
    pass name and that goes on ("allarch20 UNSURE", "allarch20 UNSURE (family search not run)": allarch21) names that pass;
    any other string is kept whole (every older `supersedes` is a bare pass name)."""
    names = {n for n, _ in PASSES}
    out = []
    for s in ([sup] if isinstance(sup, str) else [x for x in sup if isinstance(x, str)] if isinstance(sup, list) else []):
        s = s.strip()
        w = re.match(r"[A-Za-z0-9_]+", s)
        out.append(w.group(0) if s not in names and w and w.group(0) in names else s)
    return out


def apply_supersedes(cards, notes):
    """imgrecord7: the rule for rows that name the row they supersede (`supersedes`: the pass, same card). The superseded row
    stays. NOT SEEN superseded by SEEN AGAIN (either form) -> NO POWER, nights_missed 0, the at-place sentence; every other case
    keeps its verdict and gets 'see the <pass> row'. A row an applied ERRATA entry wrote is left as it is (the hand entry wins).
    Returns the log [{card, pass, by, verdict_before, verdict_after, how}]."""
    targets = {}
    for c, rs in cards.items():
        for r in rs:
            for old in r.get("_sup") or []:
                targets.setdefault((c, old), []).append(r)
    log = []
    for (c, old), news in targets.items():
        tg = [r for r in cards[c] if r["pass"] == old]
        by = [r["pass"] for r in news]
        if not tg:
            notes.append("%s %s: supersedes %s, but that pass has no row for the card; nothing changed" % (by[0], c, old))
            continue
        r = tg[0]
        rec = {"card": c, "pass": old, "by": by, "verdict_before": r["verdict"]}
        if r.get("_erratum"):
            rec.update(verdict_after=r["verdict"], how="hand-written erratum applied to this row wins; rule not applied")
            notes.append("%s %s: superseded by %s; its hand-written erratum wins over the rule" % (old, c, ", ".join(by)))
            log.append(rec)
            continue
        seen = [n for n in news if n["verdict"].startswith("SEEN AGAIN")]
        if r["verdict"] == "NOT SEEN" and seen:
            ev = no_power_text(r["evidence"], seen[0].get("_why") or "", by)
            if ev:
                r["evidence"], r["verdict"], r["nights_missed"] = ev, "NO POWER", 0
                rec["how"] = "NOT SEEN superseded by SEEN AGAIN: NO POWER, nights_missed 0, at-place sentence added"
        else:
            ev = see_row_text(r["evidence"], by)
            if ev:
                r["evidence"] = ev
                rec["how"] = "verdict kept, '%s' added" % rowref(by)
        if not ev:
            rec["how"] = "NO TEXT FITS the evidence rule: row left unchanged (look by hand)"
        rec["verdict_after"] = r["verdict"]
        notes.append("%s %s: superseded by %s (rule): %s -> %s" % (old, c, ", ".join(by), rec["verdict_before"], r["verdict"]))
        log.append(rec)
    return log


# ---------------------------------------------------------------- imgrecord8: duplicate warps and ps1night, by rule
DUP_WORDS = {  # ps1dup's word for a row -> the clause the row gets (the verdict changes only for FALLS)
    "FALLS": "so the sighting no longer passes its rule",
    "CHANGES UP": "which now passes the copies rule, but the pass proposes no change (the wide-search rows decide)",
    "WEAKER BUT STANDS": "weaker, but the verdict stands",
    "STANDS": "the verdict stands",
    "NEEDS A RE-RUN": "the pass asks for a re-run",
    "UNSURE UNCHANGED": "duplicates do not change the verdict",  # imgrecord9: ps1dup2's word for the UNSURE rows
}


DUP_TAG = "before duplicate warps were dropped"


def handback_done(p):
    return os.path.exists(os.path.join(HERE, p + ".handback.md"))


def load_dup(notes):
    """(card, pass) -> (dup pass, entry) from DUP_PASSES whose hand-back exists (a later pass wins)"""
    m = {}
    del DUP_BOTH[:]
    for dp in DUP_PASSES:
        fn = os.path.join(HERE, dp + ".json")
        if not handback_done(dp) or not os.path.exists(fn):
            continue
        d = json.load(open(fn))
        for c, ents in (d.get("cards") or {}).items():
            for e in (ents if isinstance(ents, list) else []):
                if isinstance(e, dict) and isinstance(e.get("pass"), str):
                    for p in [x.strip() for x in e["pass"].split(" + ") if x.strip()]:  # imgrecord9: "otherarch2 + ps1joint"
                        if (c, p) in m and m[(c, p)][0] != dp:
                            DUP_BOTH.append({"card": c, "pass": p, "earlier": m[(c, p)][0], "wins": dp})
                        m[(c, p)] = (dp, dup_norm(e))  # imgrecord9: ps1dup2's strings read into ps1dup's shape
        notes.append("%s read: %d audited rows" % (dp, sum(1 for v in m.values() if v[0] == dp)))
    return m


def _isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _tok(t, v):
    """position of the number v (as fmt writes it) standing alone in t, else -1"""
    m = re.search(r"(?<![\d.])" + re.escape(fmt(float(v))) + r"(?![\d])", t)
    return m.start() if m else -1


def _span(t, v):
    """imgrecord9: (start, end) of the number v standing alone in t, also when printed with trailing zeros ("7.50"), else (-1, -1)"""
    s = fmt(float(v))
    m = re.search(r"(?<![\d.])" + re.escape(s) + (r"0*" if "." in s else r"(?:\.0+)?") + r"(?![\d])", t)
    return (m.start(), m.end()) if m else (-1, -1)


def dup_change(e):
    """ps1dup entry -> what it corrects: dict(bc, ac, kb, ka, halves) or None (no corrected value)"""
    b, a = e.get("before") or {}, e.get("after") or {}
    bc, ac = b.get("combined"), a.get("combined")
    if not (_isnum(bc) and _isnum(ac)):
        return None
    hb, ha = b.get("halves") or {}, a.get("halves") or {}
    hv = (hb.get("A"), hb.get("B"), ha.get("A"), ha.get("B"))
    ch = {"bc": bc, "ac": ac, "kb": b.get("copies"), "ka": a.get("copies"),
          "halves": hv if all(_isnum(x) for x in hv) else None}
    ch["n"] = a.get("copies_of") or 240  # imgrecord9
    ch["hc"] = (hb.get("A_copies"), hb.get("B_copies"), ha.get("A_copies"), ha.get("B_copies"))  # imgrecord9
    if fmt(float(bc)) == fmt(float(ac)) and dup_word(e.get("verdict")) not in ("FALLS", "CHANGES UP"):
        return None
    return ch


def dup_text(old, verdict_after, word, dp, ch):
    """the row's evidence with the corrected values (old ones named), or None when no text fits the rule"""
    t = re.sub(r"\s+", " ", old.strip())
    tag = "before duplicate warps were dropped"
    bcs, acs = fmt(float(ch["bc"])), fmt(float(ch["ac"]))
    inplace = False
    i, ie = _span(t, ch["bc"])  # imgrecord9: trailing zeros too
    if i >= 0:
        bcs = t[i:ie]
        t = t[:i] + "%s (%s %s)" % (acs, bcs, tag) + t[ie:]
        inplace = True
    hv = ch["halves"]
    if hv and fmt(float(hv[0])) != fmt(float(hv[2])) or hv and fmt(float(hv[1])) != fmt(float(hv[3])):
        (i, ie), (j, je) = _span(t, hv[0]), _span(t, hv[1])  # imgrecord9: trailing zeros too
        if 0 <= i < j:  # both halves in the text, in order: replaced, the old ones named after the second
            sa, sb = t[i:ie], t[j:je]
            hc = ch.get("hc") or ()  # imgrecord9: the halves' copies, when they changed
            extra = "; %d and %d of %d copies reach them now" % (hc[2], hc[3], ch.get("n", 240)) \
                if len(hc) == 4 and all(_isnum(x) for x in hc) and (hc[0], hc[1]) != (hc[2], hc[3]) else ""
            ins = "%s (%s and %s %s%s)" % (fmt(float(hv[3])), sa, sb, "before" if inplace else tag, extra)
            t = t[:i] + fmt(float(hv[2])) + t[i + len(sa):j] + ins + t[j + len(sb):]
            if extra:  # imgrecord9: a stale "(none of 240 copies)" right after the halves goes
                k = i + len(fmt(float(hv[2]))) + (j - i - len(sa)) + len(ins)
                m = re.compile(r"\s*\(none of 240 (?:shifted )?copies[^)]*\)").search(t, k, k + 40)
                if m:
                    t = t[:m.start()] + t[m.end():]
            inplace = True
    if verdict_after == "UNSURE" and word == "FALLS":
        t = re.sub(r"^(?i:seen again \(combined\)|seen again)", lambda m: "First read as " + m.group(0).lower(), t)
    head = "" if inplace else "with duplicate warps dropped (pass %s) the measured value is %s (%s %s)" % (dp, acs, bcs, tag)
    cop = ""
    if _isnum(ch["kb"]) and _isnum(ch["ka"]) and ch["kb"] != ch["ka"]:
        cop = "%d of %d shifted copies reach it (%d before)" % (ch["ka"], ch.get("n", 240), ch["kb"])
    wd = DUP_WORDS.get(word, "")
    must = word in ("FALLS", "CHANGES UP", "NEEDS A RE-RUN")
    tails = []
    for parts in ([head, cop, wd], [head, cop] + ([wd] if must else []), [head] + ([cop, wd] if must else []),
                  [head] + ([wd] if must else [])):
        x = ", ".join(p for p in parts if p)
        if x and not inplace and x.startswith("with"):
            pass
        elif x and inplace:
            x = "with duplicate warps dropped (pass %s): " % dp + x if (cop or wd) else ""
        if x not in tails:
            tails.append(x)
    if not inplace:  # compact forms of the same correction
        for parts in ([[cop]] if cop else []) + [[]]:
            x = ", ".join(["pass %s: %s (%s %s)" % (dp, acs, bcs, tag)] + parts + ([wd] if must else []))
            if x not in tails:
                tails.append(x)
    ss = sentences(t)
    cands = []
    for x in tails:
        if not x:
            cands.append(" ".join(ss))
            continue
        if len(ss) == 1:
            cands.append(ss[0] + " " + x[0].upper() + x[1:] + ".")
        cands.append(" ".join(ss[:-1] + [ss[-1].rstrip(".") + "; " + x + "."]))
    if len(ss) > 1:  # the first sentence's leading clauses, the later sentences, the correction joined to the last
        for x in tails:
            for sep in ("; ", ", "):
                for cl in clauses(ss[0].rstrip("."), sep):
                    if x and len(cl) >= 40 and cl.count("(") == cl.count(")"):
                        cands.append(" ".join([cl + "."] + ss[1:-1] + [ss[-1].rstrip(".") + "; " + x + "."]))
    last = [ss[0] + " " + x[0].upper() + x[1:] + "." for x in tails if x] if len(ss) > 1 else []
    if len(ss) > 1 and (must or not inplace):  # last resort: the first sentence and the correction (a later sentence dropped)
        cands += last
    if inplace:
        cands.append(" ".join(ss))
    got = first_ok([x for x in cands if tag in x])  # the corrected value with the old one named is never cut away
    if got:
        return got
    # imgrecord9: no candidate fits: each shortened at a clause end of one sentence (every sentence kept), last sentence
    # first; only then, for a word that is not a must, a later sentence dropped
    for c0 in cands + ([] if (must or not inplace) else last):
        s2 = sentences(c0)
        for k in range(len(s2) - 1, -1, -1):
            for sep in ("; ", ", "):
                for cl in clauses(s2[k].rstrip("."), sep):
                    if len(cl) >= 40 and cl.count("(") == cl.count(")"):
                        t2 = " ".join(s2[:k] + [cl + "."] + s2[k + 1:])
                        if tag in t2 and not check_text(t2):
                            return t2
    return None


def apply_dup(name, c, row, e, dup, notes, log):
    """the duplicate-warp correction of one row (after the ERRATA step; an applied erratum wins)"""
    if (c, name) not in dup:
        return
    dp, de = dup[(c, name)]
    word = dup_word(de.get("verdict"))  # imgrecord9: the leading word of the pass's verdict words
    ch = dup_change(de)
    rec = {"card": c, "pass": name, "by": dp, "word": word, "verdict_before": row["verdict"]}
    if row.get("_erratum"):
        rec.update(verdict_after=row["verdict"], how="hand-written erratum applied to this row wins; rule not applied")
        notes.append("%s %s: %s correction not applied, its erratum wins" % (name, c, dp))
        log.append(rec)
        return
    if ch is None:
        return
    v = dup_verdict(row["verdict"], de)  # imgrecord9: STANDS keeps the verdict, FALLS takes the combined-statistic rule
    ev = dup_text(row["evidence"], v, word, dp, ch)
    if not ev:
        rec.update(verdict_after=row["verdict"], how="NO TEXT FITS the evidence rule: row left unchanged (look by hand)")
        notes.append("%s %s: %s correction does not fit the evidence rule; row unchanged (look by hand)" % (name, c, dp))
        log.append(rec)
        return
    row["evidence"] = ev
    if v != row["verdict"]:
        row["verdict"], row["nights_seen"] = v, 0
    row["_dup"] = dp
    row["_dupword"] = word
    rec.update(verdict_after=v, value_before=ch["bc"], value_after=ch["ac"], copies_before=ch["kb"], copies_after=ch["ka"],
               how="value corrected" + ("; verdict %s -> %s (%s)" % (rec["verdict_before"], v, word) if v != rec["verdict_before"] else ""))
    notes.append("%s %s: duplicate warps (%s, %s): %s -> %s, verdict %s" % (name, c, dp, word, fmt(float(ch["bc"])),
                                                                           fmt(float(ch["ac"])), v))
    log.append(rec)


NIGHT_SHORT = (("one night", "one night above 60 %"), ("clear peak", "no clear peak"), ("fixed source", "fixed sources"))


def night_sentence(ent, short=False):
    """the ps1night sentence of one card (years only), or None when no year failed"""
    yf = ent.get("years_failing") or {}
    if not yf:
        return None
    ys = sorted(yf, key=lambda y: int(y))
    why = {}
    for y in ys:
        txt = yf[y] if isinstance(yf[y], str) else "; ".join(str(x) for x in (yf[y] or []))
        for part in txt.split("; "):
            for k, w in NIGHT_SHORT:
                if part.strip().lower().startswith(k):
                    why.setdefault(w, [])
                    if y not in why[w]:
                        why[w].append(y)
    rs = "; ".join("%s in %s" % (w, ", ".join(why[w])) if len(why) > 1 else w for w in why) or "it failed the tests"
    n = len(ys)
    s = "the Pan-STARRS1 year place%s of %s %s left out of the orbit" % ("s" if n > 1 else "", ", ".join(ys),
                                                                        "were" if n > 1 else "was")
    if not short:
        s += " (%s)" % rs
    if not (ent.get("years_kept") or {}):
        s += ", so no Pan-STARRS1 year place is left in the orbit"
    return "Later (ps1night pass): " + s + "; the whole-arc sighting is not changed by this."


def night_row(rs, ent):
    """the row the card's year places came from: the joint pass's row, else otherarch2's (ps1joint files), else the newest
    Pan-STARRS1 row; -> (row, how)"""
    jp = ent.get("joint_pass")
    for r in rs:
        if r["pass"] == jp:
            return r, "the joint pass's row"
    if jp == "ps1joint" or ent.get("at_place_pass") == "otherarch2":
        for r in rs:
            if r["pass"] == "otherarch2":
                return r, "otherarch2's row (the ps1joint file's year stacks)"
    ps = [r for r in rs if r["pass"].startswith(PS1_PASSES)]
    return (ps[-1], "the newest Pan-STARRS1 row (the joint pass %s has no row)" % jp) if ps else (None, None)


def apply_night(cards, notes):
    """imgrecord8: the ps1night sentence on each card with a failing year (no verdict change) -> log"""
    fn = os.path.join(HERE, NIGHT_PASS + ".json")
    if not handback_done(NIGHT_PASS) or not os.path.exists(fn):
        return []
    log = []
    for c, ent in sorted((json.load(open(fn)).get("cards") or {}).items()):
        if not isinstance(ent, dict) or not ent.get("years_failing"):
            continue
        r, how = night_row(cards.get(c, []), ent)
        if r is None:
            notes.append("ps1night %s: failing years but no Pan-STARRS1 row in the table; nothing added" % c)
            log.append({"card": c, "row": None, "how": "no Pan-STARRS1 row"})
            continue
        ss = sentences(r["evidence"])
        cands = []
        for short in (False, True):
            ns = night_sentence(ent, short)
            if len(ss) == 1:
                cands.append((ss[0] + " " + ns, "added as a sentence"))
            cands.append((" ".join(ss[:-1] + [ss[-1].rstrip(".") + "; l" + ns[1:]]), "joined to the last sentence"))
        for short in (False, True):
            ns = night_sentence(ent, short)
            if len(ss) > 1:
                cands.append((" ".join(ss[:-1]) + " " + ns, "replaces the row's last sentence"))
                cands.append((ss[0] + " " + ns, "replaces the row's later sentences"))
        ns = night_sentence(ent, True)  # last resort: the first sentence's leading clauses and the short sentence
        for sep in ("; ", ", "):
            for cl in clauses(ss[0].rstrip("."), sep):
                if len(cl) >= 40 and cl.count("(") == cl.count(")"):
                    cands.append((cl + ". " + ns, "the first sentence's leading clauses and the sentence"))
        for short in (False, True):  # a sentence without the duplicate-warp correction replaced, the correction kept
            ns = night_sentence(ent, short)
            for k in range(1, len(ss)):
                if DUP_TAG not in ss[k]:
                    cands.append((" ".join(ss[:k] + ss[k + 1:]) + " " + ns, "replaces a later sentence of the row"))
        for short in (False, True):  # last resort with a correction in a later sentence: the first sentence's leading clauses,
            ns = night_sentence(ent, short)  # then that sentence with the ps1night sentence joined to it
            for sep in ("; ", ", "):
                for cl in clauses(ss[0].rstrip("."), sep):
                    if len(cl) >= 40 and cl.count("(") == cl.count(")"):
                        for k in range(1, len(ss)):
                            if DUP_TAG in ss[k]:
                                cands.append((cl + ". " + ss[k].rstrip(".") + "; l" + ns[1:],
                                              "the first sentence's leading clauses, the correction and the sentence"))
        for k in range(1, len(ss)):  # a later sentence holding the correction cut down to the correction itself
            lo = ss[k].lower()
            i = lo.find("with duplicate warps dropped (pass")
            i = i if i >= 0 else lo.find("pass ps1dup")
            if DUP_TAG not in ss[k] or i < 0:
                continue
            cor = ss[k][i:].rstrip(".")
            cor = cor[0].upper() + cor[1:]
            for short in (False, True):
                ns = night_sentence(ent, short)
                heads = [ss[0]] + [cl + "." for sep in ("; ", ", ") for cl in clauses(ss[0].rstrip("."), sep)
                                   if len(cl) >= 40 and cl.count("(") == cl.count(")")]
                for h in heads:
                    cands.append((h + " " + cor + "; l" + ns[1:], "the correction kept, the ps1night sentence joined to it"))
        got = [(t, h) for t, h in cands if not check_text(t)]
        if DUP_TAG in r["evidence"]:  # imgrecord8: the duplicate-warp correction is never dropped for this sentence
            got = [(t, h) for t, h in got if DUP_TAG in t]
        if not got:
            notes.append("ps1night %s %s: NO TEXT FITS the evidence rule; row unchanged (look by hand)" % (c, r["pass"]))
            log.append({"card": c, "row": r["pass"], "how": "NO TEXT FITS"})
            continue
        r["evidence"] = got[0][0]
        r["_night"] = True
        log.append({"card": c, "row": r["pass"], "years": sorted(ent["years_failing"], key=int),
                    "no_year_left": not (ent.get("years_kept") or {}), "row_chosen": how, "how": got[0][1]})
        notes.append("ps1night %s: sentence on the %s row (%s; %s)" % (c, r["pass"], how, got[0][1]))
    return log


# ---------------------------------------------------------------- imgrecord9: ps1dup2's format, the verdict rule, orbit sentences
DUP_BOTH = []  # rows named by ps1dup and by ps1dup2 (the later re-measurement wins); filled by load_dup


def _copies(x):
    """copies field -> (k, n): an int (ps1dup: of 240) or "10 of 240 (copy max 252.21)" (ps1dup2)"""
    if _isnum(x):
        return int(x), 240
    m = re.match(r"\s*(\d+)\s+of\s+(\d+)", x) if isinstance(x, str) else None
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _halves(x):
    """halves field -> {"A", "A_copies", "B", "B_copies"}: a dict (ps1dup) or "8.35 (12) / 8.45 (6)" (ps1dup2)"""
    if isinstance(x, dict):
        return x
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*(?:\((\d+)[^)]*\))?\s*/\s*(-?\d+(?:\.\d+)?)\s*(?:\((\d+)[^)]*\))?", x) \
        if isinstance(x, str) else None
    if not m:
        return {}
    return {"A": float(m.group(1)), "A_copies": int(m.group(2)) if m.group(2) else None,
            "B": float(m.group(3)), "B_copies": int(m.group(4)) if m.group(4) else None}


def dup_norm(e):
    """a duplicate-warp entry in ps1dup's shape, in memory (ps1dup2 writes copies and halves as strings)"""
    e = dict(e)
    for side in ("before", "after"):
        s = e.get(side)
        if isinstance(s, dict):
            s = dict(s)
            s["copies"], s["copies_of"] = _copies(s.get("copies"))
            s["halves"] = _halves(s.get("halves"))
            e[side] = s
    return e


def dup_word(v):
    """the pass's verdict words -> their leading word: "WEAKER BUT STANDS (only with ...)" -> "WEAKER BUT STANDS" """
    u = (v or "").strip().upper()
    for k in sorted(DUP_WORDS, key=len, reverse=True):
        if u.startswith(k):
            return k
    return (v or "").strip()


def combined_rule(side):
    """the combined-statistic rule on corrected numbers: SEEN AGAIN (combined) needs fewer than 0.3 % of at least 200
    shifted copies reaching the measured value and no single night above 60 % of it; else UNSURE"""
    k, n, lf = side.get("copies"), side.get("copies_of"), side.get("largest_night_fraction")
    ok = _isnum(k) and _isnum(n) and n >= 200 and k < 0.003 * n and _isnum(lf) and lf <= 0.6
    return "SEEN AGAIN (combined)" if ok else "UNSURE"


def dup_verdict(row_verdict, de):
    """STANDS, WEAKER BUT STANDS and every word but FALLS keep the row's verdict; FALLS: the rule on the corrected numbers
    (a SEEN AGAIN row the rule no longer passes becomes UNSURE; a fall never promotes a row)"""
    if dup_word(de.get("verdict")) != "FALLS" or not row_verdict.startswith("SEEN AGAIN"):
        return row_verdict
    return row_verdict if combined_rule(de.get("after") or {}) == "SEEN AGAIN (combined)" else "UNSURE"


def add_sentence(ev, variants, keep=()):
    """ev with one more sentence (variants: the sentence, longest first), fitted to the rule; every tag of `keep` that is in
    ev stays -> (text, how) or (None, None)"""
    ss = sentences(ev)
    must = [k for k in keep if k in ev]
    low = lambda s: s[0].lower() + s[1:]
    ok_cl = lambda cl: len(cl) >= 40 and cl.count("(") == cl.count(")")
    cands = []
    for ns in variants:
        if len(ss) == 1:
            cands.append((ss[0] + " " + ns, "added as a sentence"))
        cands.append((" ".join(ss[:-1] + [ss[-1].rstrip(".") + "; " + low(ns)]), "joined to the last sentence"))
    cut_ok = lambda cl, full: not (cl.split("; ")[-1].startswith("with duplicate warps dropped (pass") and len(cl) < len(full))
    for ns in variants:  # the last sentence cut at a clause end (never inside the duplicate-warp clause)
        for sep in ("; ", ", "):
            for cl in clauses(ss[-1].rstrip("."), sep):
                if ok_cl(cl) and cut_ok(cl, ss[-1].rstrip(".")):
                    cands.append((" ".join(ss[:-1] + [cl + "; " + low(ns)]), "joined to the last sentence, cut at a clause end"))
    for ns in variants:  # a later sentence with no tag and no old value replaced
        for k in range(len(ss) - 1, 0, -1):
            if not any(x in ss[k] for x in KEEP_TAGS) and "before" not in ss[k]:
                cands.append((" ".join(ss[:k] + ss[k + 1:]) + " " + ns, "replaces a later sentence of the row"))
    for ns in variants:  # sentences cut at clause ends, every sentence kept
        for sep in ("; ", ", "):
            for k in range(len(ss)):
                for cl in clauses(ss[k].rstrip("."), sep):
                    if ok_cl(cl) and cut_ok(cl, ss[k].rstrip(".")):
                        t = ss[:k] + [cl + "."] + ss[k + 1:]
                        cands.append((" ".join(t[:-1] + [t[-1].rstrip(".") + "; " + low(ns)]),
                                      "joined to the last sentence, one sentence cut at a clause end"))
    for ns in variants:
        for k in range(len(ss) - 1, 0, -1):  # a later sentence replaced
            cands.append((" ".join(ss[:k] + ss[k + 1:]) + " " + ns, "replaces a later sentence of the row"))
        for k in range(len(ss) - 1, 0, -1):  # a later sentence with no tag dropped and another cut at a clause end
            if any(x in ss[k] for x in KEEP_TAGS) or "before" in ss[k]:
                continue
            rest = ss[:k] + ss[k + 1:]
            for j in range(len(rest)):
                for sep in ("; ", ", "):
                    for cl in clauses(rest[j].rstrip("."), sep):
                        if ok_cl(cl) and cut_ok(cl, rest[j].rstrip(".")):
                            cands.append((" ".join(rest[:j] + [cl + "."] + rest[j + 1:]) + " " + ns,
                                          "replaces a later sentence; another cut at a clause end"))
        for sep in ("; ", ", "):
            for cl in clauses(ss[0].rstrip("."), sep):
                if ok_cl(cl):
                    cands.append((cl + ". " + ns, "the first sentence's leading clauses and the sentence"))
    for t, how in cands:
        if not check_text(t) and all(k in t for k in must):
            return t, how
    return None, None


KEEP_TAGS = (DUP_TAG, "(ps1night pass)", "(ps1dup2 pass)")


def falls_sentence(cards, notes):
    """a SEEN AGAIN row turned UNSURE by a FALLS correction, superseded by a SEEN AGAIN wide-search row: that row confirms the
    card independently, so its stage does not change (in place of 'see the <pass> row') -> log"""
    log = []
    for c, rs in sorted(cards.items()):
        for r in rs:
            if r.get("_dupword") != "FALLS" or r["verdict"] != "UNSURE":
                continue
            by = [x["pass"] for x in rs if r["pass"] in (x.get("_sup") or []) and x["verdict"].startswith("SEEN AGAIN")
                  and x["pass"].startswith("ps1wide")]
            if not by:
                continue
            ref = rowref(by)
            ev = r["evidence"]
            for old in (" " + ref[0].upper() + ref[1:] + ".", "; " + ref + "."):
                if ev.endswith(old):
                    ev = ev[:-len(old)].rstrip(".") + "."
                    break
            s = "The wide search%s %s confirm%s the card independently, so its stage does not change." % (
                "es" if len(by) > 1 else "", " and ".join(by), "" if len(by) > 1 else "s")
            t, how = add_sentence(ev, [s], KEEP_TAGS)
            if not t:
                notes.append("%s %s: the FALLS sentence does not fit the evidence rule; row unchanged (look by hand)" % (r["pass"], c))
                continue
            r["evidence"] = t
            log.append({"card": c, "row": r["pass"], "confirmed_by": by, "how": how})
            notes.append("%s %s: FALLS row, confirmed by %s (stage unchanged sentence; %s)" % (r["pass"], c, ", ".join(by), how))
    return log


NUMWORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def _ps1_years(fn):
    """the Pan-STARRS1 place years a joint file holds (from its `joint_with` text), else None"""
    if not isinstance(fn, str) or not fn:
        return None
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        return None
    jw = json.load(open(p)).get("joint_with") or ""
    m = re.search(r"Pan-STARRS1 3pi (?:season|year-stack) places \(([\d, ]+)[;)]", jw)
    return [y.strip() for y in m.group(1).split(",") if y.strip()] if m else None


def _and(ys):
    return ys[0] if len(ys) == 1 else ", ".join(ys[:-1]) + " and " + ys[-1]


def orbit_sentences(cards, notes):
    """imgrecord9: one sentence by rule from ps1dup2.json (joint_files_written, one_night_failures, joint_file) on the row of
    the entry's pass (years only) -> log"""
    log = []
    for dp in DUP_PASSES[1:]:
        fn = os.path.join(HERE, dp + ".json")
        if not handback_done(dp) or not os.path.exists(fn):
            continue
        d = json.load(open(fn))
        written = [w for w in (d.get("joint_files_written") or []) if isinstance(w, str)]
        onf = d.get("one_night_failures") or {}
        for c, ents in sorted((d.get("cards") or {}).items()):
            for e in (ents if isinstance(ents, list) else []):
                jf = e.get("joint_file") if isinstance(e, dict) else None
                if not isinstance(jf, dict):
                    continue
                ps = [x.strip() for x in str(e.get("pass")).split(" + ")]
                r = next((x for p in ps for x in cards.get(c, []) if x["pass"] == p), None)
                mine = [w for w in written if os.path.basename(w).startswith(c + "_")]
                s = None
                if mine:
                    new, old = _ps1_years(mine[0]), _ps1_years(jf.get("file") if jf.get("file") != mine[0] else None)
                    gone = [y for y in (old or []) if y not in (new or [])]
                    share = next((re.search(r"(\d+) %", v).group(1) for k, v in onf.items()
                                  if k.split()[0] == c and ps[0] in k and re.search(r"(\d+) %", v)), None)
                    moved = {k: v for k, v in (e.get("year_places_moved_arcsec") or {}).items() if _isnum(v)}
                    if new and gone and share:
                        s = "the orbit now holds the %s Pan-STARRS1 places; the %s place%s left out (one night carries %s %% of it)" % (
                            _and(new), _and(gone), " was" if len(gone) == 1 else "s were", share)
                    elif new and moved:
                        y, x = max(moved.items(), key=lambda kv: kv[1])
                        allpass = str(e.get("one_night") or "").startswith("all ")
                        s = "the %s place moved %s arcsec after duplicate warps were dropped; orbit refit%s" % (
                            y, fmt(float(x)), ", all %s years pass" % NUMWORD.get(len(new), str(len(new))) if allpass else "")
                elif jf.get("fit_to_install") is False:
                    s = "the %s orbit file was found not fit to install, and the installed orbit stands" % ps[0]
                if not s:
                    continue
                s = "Later (%s pass): %s." % (dp, s)
                if r is not None and s in r["evidence"]:  # imgrecord10: the row (an erratum) already carries it
                    log.append({"card": c, "row": r["pass"], "sentence": s, "how": "already in the row"})
                    continue
                if r is None:
                    notes.append("%s %s: orbit sentence but no row of %s in the table; nothing added" % (dp, c, ps[0]))
                    log.append({"card": c, "row": None, "sentence": s})
                    continue
                t, how = add_sentence(r["evidence"], [s], KEEP_TAGS)
                if not t:
                    notes.append("%s %s %s: the orbit sentence does not fit the evidence rule; row unchanged (look by hand)" % (dp, c, r["pass"]))
                    log.append({"card": c, "row": r["pass"], "sentence": s, "how": "NO TEXT FITS"})
                    continue
                r["evidence"] = t
                log.append({"card": c, "row": r["pass"], "sentence": s, "how": how})
                notes.append("%s %s: orbit sentence on the %s row (%s)" % (dp, c, r["pass"], how))
    return log


PREAMBLE = re.compile(r"(?i)^images, value 7 counted as good pixels")  # imgrecord10: u20img's method sentence
PREAMBLE_S = re.compile(r"^Images, value 7 counted as good pixels\.\s*")  # imgrecord10: dropped before fit_text
_ALIAS_DONE = {}


def alias_card(k, notes):
    """imgrecord10: a pass's key that is not a card id -> the card (ALIAS), checked on R/candidates/<card>/card.json (its key
    must hold the word); None (with a note) when the check fails"""
    if k in _ALIAS_DONE:
        return _ALIAS_DONE[k]
    c, word = ALIAS[k]
    p = os.path.join(R, "candidates", c, "card.json")
    try:
        ok = word in str(json.load(open(p)).get("key") or "")
    except Exception:
        ok = False
    _ALIAS_DONE[k] = c if ok else None
    notes.append("alias %s -> %s %s" % (k, c, "checked on the card's key" if ok else "NOT confirmed by the card's key; not used"))
    return _ALIAS_DONE[k]


def pass_cards(name, notes=None):
    """imgrecord10: a pass's per-card entries from its <pass>.json, at the top level or in a "cards" section, keys mapped
    through ALIAS -> {card: entry}"""
    fn = os.path.join(HERE, name + ".json")
    if not os.path.exists(fn):
        return {}
    d = json.load(open(fn))
    out = {}
    secs = [d] + ([d["cards"]] if isinstance(d, dict) and isinstance(d.get("cards"), dict) else [])
    for sec in secs:
        for k, e in (sec.items() if isinstance(sec, dict) else []):
            c = alias_card(k, notes if notes is not None else []) if k in ALIAS else k
            if c and re.fullmatch(r"C\d{3}", c) and isinstance(e, dict):
                out.setdefault(c, e)
    return out


ARCH_RANK = {"SEEN AGAIN": 0, "SEEN AGAIN (combined)": 0, "NOT SEEN": 1, "UNSURE": 2, "NO POWER": 3}


CONF_RX = re.compile(r"^\s*CONFIRMED\b")  # imgrecord12: allarch22's word (never "NOT CONFIRMED", which starts otherwise)
CONF_LOG = {}  # imgrecord12: (card, archive) -> what confirmed_rule found; written to the notes by main


def confirmed_rule(p):
    """imgrecord12: the combined-statistic rule on a part's own evidence numbers -> (holds, what was read). SEEN AGAIN
    (combined) needs fewer than 0.3 % of at least 200 shifted copies reaching the value, no night above 60 % of it, each
    independent half reached by fewer than 0.3 % of its copies, and at least 2 nights; a number not found fails the rule."""
    ev = re.sub(r"\s+", " ", str(p.get("evidence") or ""))
    cp = re.search(r"\b(\d+) of (\d+) shifted copies reaching it", ev)
    lf = re.search(r"largest night (\d+(?:\.\d+)?) ?%", ev)
    hv = {h: re.search(r"half %s -?\d+(?:\.\d+)? \((\d+) of (\d+) copies\)" % h, ev) for h in ("A", "B")}
    nn = num(p.get("nights_seen")) or 0
    under = lambda m: bool(m) and int(m.group(2)) >= 200 and int(m.group(1)) < 0.003 * int(m.group(2))
    ok = under(cp) and bool(lf) and float(lf.group(1)) <= 60 and under(hv["A"]) and under(hv["B"]) and nn >= 2
    got = "copies %s, largest night %s %%, half A %s, half B %s, nights %d" % (
        "%s of %s" % cp.groups() if cp else "not found", lf.group(1) if lf else "not found",
        "%s of %s" % hv["A"].groups() if hv["A"] else "not found", "%s of %s" % hv["B"].groups() if hv["B"] else "not found", nn)
    CONF_LOG[(p.get("C"), p.get("archive"))] = "%s %s: CONFIRMED read as %s (%s)" % (
        p.get("C"), p.get("archive"), "SEEN AGAIN (combined)" if ok else "UNSURE: the combined-statistic rule fails", got)
    return ok, got


SEASON_LOG = {}  # imgrecord13: (card, season) -> how season_label read it; written to the notes by main


def season_rule(p):
    """imgrecord14: the combined-statistic rule on a season part's peak numbers with the static-peak mask on, from its own text
    -> (holds, what was read): fewer than 0.3 % of at least 200 shifted copies reach the peak, and no night above 60 % of it;
    a number not found (or found twice) fails the rule."""
    ev = re.sub(r"\s+", " ", str(p.get("evidence") or ""))
    cp = re.findall(r"\b(\d+) of (\d+) shifted copies reaching it with single-warp static peaks masked", ev)
    lf = re.findall(r"largest night (\d+(?:\.\d+)?) ?% of the value", ev)
    ok = (len(cp) == 1 and int(cp[0][1]) >= 200 and int(cp[0][0]) < 0.003 * int(cp[0][1]) and len(lf) == 1
          and float(lf[0]) <= 60)
    got = "copies %s, largest night %s" % ("%s of %s" % cp[0] if len(cp) == 1 else "not found",
                                          lf[0] + " %" if len(lf) == 1 else "not found")
    return ok, got


def season_label(p):
    """imgrecord13: one season part of a SEASON_PASSES pass. The pass's own verdict: SEEN AGAIN (a season that passes the
    clear-peak, one-night and fixed-source tests) is SEEN AGAIN (combined), a year stack being several nights together; UNSURE
    (a season that fails) stays UNSURE; a NOT SEEN would be read as UNSURE (a failed one-night test does not show the body
    absent); an unrecognised word is UNSURE."""
    v = p.get("verdict") if isinstance(p.get("verdict"), str) else ""
    own = verdict_label(v)
    lab = {"SEEN AGAIN": "SEEN AGAIN (combined)", "SEEN AGAIN (combined)": "SEEN AGAIN (combined)"}.get(own, "UNSURE")
    ok, got = season_rule(p)  # imgrecord14: the combined-statistic rule on the peak numbers, mask on
    if lab == "SEEN AGAIN (combined)" and not ok:
        lab = "UNSURE"
    SEASON_LOG[(p.get("C"), p.get("archive"))] = "%s %s: season part, the pass's word %r read as %s%s; combined-statistic rule %s (%s)" % (
        p.get("C"), p.get("archive"), v[:60], lab, "" if own in ("SEEN AGAIN", "SEEN AGAIN (combined)", "UNSURE")
        else " (its own label %s is not used for a season test)" % own, "holds" if ok else "fails", got)
    return lab


def part_label(p):
    """imgrecord12: the label of one part of an archive-shaped entry: CONFIRMED by confirmed_rule, else verdict_label
    (NOT CONFIRMED is UNSURE through WORDS); an unrecognised word is UNSURE, as before. imgrecord13: a season part
    (SEASON_PASSES) by season_label."""
    v = p.get("verdict") if isinstance(p.get("verdict"), str) else ""
    if p.get("_season"):
        return season_label(p)
    if CONF_RX.match(v):
        return "SEEN AGAIN (combined)" if confirmed_rule(p)[0] else "UNSURE"
    return verdict_label(v) or "UNSURE"


def archive_entry(parts):
    """imgrecord10: one entry for a card from a pass that writes one image row per card and archive ("C###:<archive>",
    allarch20). The part with the strongest label leads (SEEN AGAIN, then NOT SEEN, UNSURE, NO POWER; file order on ties); a
    text that does not name its archive gets "<archive>: " in front (not for "all archives"); every other archive is joined
    to the last sentence as "<archive>: <label> (<its text>)". nights_seen counts SEEN AGAIN nights only (a NOT CONFIRMED
    one-night sighting is UNSURE with 0 nights seen; its text says one night); nights_expected_not_seen is summed."""
    lab = part_label  # imgrecord12: CONFIRMED by the combined-statistic rule
    order = sorted(range(len(parts)), key=lambda i: (ARCH_RANK.get(lab(parts[i]), 2), i))
    main = parts[order[0]]

    def named(p):
        ev = re.sub(r"\s+", " ", str(p.get("evidence") or "").strip())
        a = str(p.get("archive") or "")
        if a and a != "all archives" and a.split()[0] not in ev[:60]:
            ev = a + ": " + ev
        return ev

    others = [parts[i] for i in order[1:]]
    tail = "; ".join("%s: %s (%s)" % (p.get("archive"), lab(p).lower(), re.sub(r"\s+", " ", str(p.get("evidence") or "")
                                                                                 .strip()).rstrip(".")) for p in others)
    cands = []
    for t in (named(main), (fit_text(named(main))[0] or "")):
        if t:
            cands.append(t.rstrip(".") + "; " + tail + "." if tail else t)
    ev = first_ok(cands) or named(main)
    e = dict(main)
    if CONF_RX.match(str(main.get("verdict") or "")):  # imgrecord12: the row's label, with the pass's own word kept
        e["verdict"] = lab(main) + " (the pass's word: %s; combined-statistic rule %s on its numbers)" % (
            main["verdict"], "holds" if lab(main).startswith("SEEN AGAIN") else "fails")
    elif main.get("_season"):  # imgrecord13: the row's label from season_label, with the pass's own word kept
        e["verdict"] = lab(main) + " (the pass's word: %s; a one-season year stack)" % main.get("verdict")
    e.update(evidence=ev, archives=[p.get("archive") for p in parts], parts=parts,
             nights_seen=(num(main.get("nights_seen")) or 0) if lab(main).startswith("SEEN AGAIN") else 0,
             nights_expected_not_seen=sum(num(p.get("nights_expected_not_seen")) or 0 for p in parts))
    return e


def builder_rows(name, have, notes):
    """imgrecord9: rows for the cards whose verdict the pass's own <pass>.json states but that have no row in its image-row
    file (made by the builder: verdict from the json, one sentence of its evidence, years only) -> [(card, row)]"""
    fn = os.path.join(HERE, name + ".json")
    if name not in BUILDER_ROWS or not os.path.exists(fn) or fn == pass_path(name):
        return []
    d = pass_cards(name, notes)  # imgrecord10: top level or "cards" section, ALIAS
    out = []
    for c, e in d.items():
        if c in have or not isinstance(e, dict) or not isinstance(e.get("verdict"), str):
            continue
        v = verdict_label(e["verdict"])
        if v is None:
            notes.append("%s %s: verdict in the pass's json not recognised (%r...); no row made" % (name, c, e["verdict"][:40]))
            continue
        ns = (num(e.get("nights_seen")) or 0) if v.startswith("SEEN AGAIN") else 0
        nm = (num(e.get("nights_expected_not_seen")) or 0) if v == "NOT SEEN" else 0
        evt = re.sub(r"\s+", " ", (e.get("evidence") or "").strip())
        evt = re.sub(r"\s*\(Dec -?\d+(?:\.\d+)? to -?\d+(?:\.\d+)?\)", "", evt)  # imgrecord10: no positions in the text
        evt = re.sub(r"\bDec -?\d+(?:\.\d+)?(?: \(the 3pi limit\))?", "the 3pi limit", evt)
        if re.match(r"(?i)(not searched|skipped|not tested|not measured)\b", evt):  # not an image test of this pass
            notes.append("%s %s: the pass's json says it did not search the card; no row made" % (name, c))
            continue
        ss = [x if x.endswith(".") else x + "." for x in re.split(r"(?<=[\w\)\"%])\.\s+(?=[A-Z(])", evt) if x]
        key = ss[0] if ss else ""
        if v != "NO POWER" and ss:  # imgrecord10: the pass's concluding "So ..." sentence, else its first after the preamble
            body = [x for x in ss if not PREAMBLE.match(x)] or ss
            key = next((x for x in reversed(body) if x.startswith("So ")), body[0])
            key = key[3].upper() + key[4:] if key.startswith("So ") else key
        if v == "NO POWER":  # the sentence that says why there is no power, else the first
            k = next((i for i, x in enumerate(ss) if re.search(r"(?i)\bpower\b|under 4\b", x)), 0)
            key = ss[k] if ss else ""
            if k > 0 and len(key) < 60:  # a short "Under 4: NO POWER." joined to the sentence before it
                key = ss[k - 1].rstrip(".") + "; " + key[0].lower() + key[1:]
        if re.match(r"(?i)no power\s*\(step \d\):\s*", key):
            key = re.sub(r"(?i)^no power\s*\(step \d\):\s*", "", key)
            key = key[0].upper() + key[1:]
        cands = [key] + [cl + "." for sep in ("; ", ", ") for cl in clauses(key.rstrip("."), sep)
                         if len(cl) >= 40 and cl.count("(") == cl.count(")")]
        ev = first_ok([x for x in cands if len(re.findall(r"(?<=[a-z0-9\)\"%])[.!?](\s+|$)", x)) <= 1])
        if not ev:
            notes.append("%s %s: no sentence of the pass's json fits the evidence rule; no row made (look by hand)" % (name, c))
            continue
        out.append((c, {"pass": name, "verdict": v, "evidence": ev, "nights_seen": ns, "nights_missed": nm,
                        "made_by": "builder (%s): the pass states this verdict in its json but wrote no image row"
                                   % BUILDER_BY.get(name, "imgrecord9")}))
        notes.append("%s %s: row made by the builder from the pass's json (%s)" % (name, c, v))
    return out


# ---------------------------------------------------------------- lead id -> card
_LEADMAP = None


def lead_to_card(key):
    global _LEADMAP
    if _LEADMAP is None:
        _LEADMAP = {}
        cdir = os.path.join(R, "candidates")
        for c in sorted(os.listdir(cdir)):
            p = os.path.join(cdir, c, "card.json")
            if not re.fullmatch(r"C\d{3}", c) or not os.path.exists(p):
                continue
            try:
                card = json.load(open(p))
            except Exception:
                continue
            ids = card.get("identities") or {}
            keys = [card.get("key"), ids.get("lead_id")]
            for a in ids.get("also_found_as") or []:
                keys.append(a.get("key") or a.get("lead_id") if isinstance(a, dict) else a)
            for k in keys:
                if isinstance(k, str) and k:
                    _LEADMAP.setdefault(k, c)
    return _LEADMAP.get(key)


# ---------------------------------------------------------------- per pass entries
def entries_of(name, d, notes):
    """-> list of (card, entry) in the pass's own order; entry = the dict the row is built from."""
    out = []
    arch = {}  # imgrecord10: allarch20's "C###:<archive>" entries, joined per card (archive_entry)
    if not isinstance(d, dict):
        notes.append("%s: not a dict, skipped" % name)
        return out
    if name == "deepframes2":
        seen = []
        for sec in ("remeasure",):
            for c in (d.get(sec) or {}):
                seen.append(c)
        for c in (d.get("joint") or {}):
            if c not in seen:
                seen.append(c)
        for k in d:
            if re.fullmatch(r"C\d{3}", k) and k not in seen:
                seen.append(k)
        for c in seen:
            ent = {"remeasure": (d.get("remeasure") or {}).get(c), "joint": (d.get("joint") or {}).get(c),
                   "entry": d.get(c), "proposed": (d.get("proposed") or {}).get(c)}
            out.append((c, {k: v for k, v in ent.items() if v is not None}))
        return out
    for k, v in d.items():
        if k in SECTIONS_SKIP:
            continue
        if k == "combined" and isinstance(v, dict):
            for c, e in v.items():
                if isinstance(e, dict):
                    out.append((c if re.fullmatch(r"C\d{3}", c) else lead_to_card(c), dict(e, _combined=True)))
            continue
        if not isinstance(v, dict):
            continue
        if re.fullmatch(r"C\d{3}", k):
            out.append((k, v))
        elif k.startswith("dbt-"):
            c = lead_to_card(k)
            if c:
                out.append((c, v))
            else:
                notes.append("%s: lead key not found on any card, skipped" % name)
        elif isinstance(v.get("card"), str) and re.fullmatch(r"C\d{3}", v["card"]):
            out.append((v["card"], v))
        elif re.fullmatch(r"C\d{3}:.+", k) and v.get("C", k[:4]) == k[:4]:  # imgrecord10
            if k[:4] not in arch:
                arch[k[:4]] = []
                out.append((k[:4], arch[k[:4]]))  # its place in the pass's order; joined below
            arch[k[:4]].append(dict(v, archive=v.get("archive") or k[5:]))
            if name in SEASON_PASSES:  # imgrecord13: a one-season test inside an earlier whole-arc sighting
                arch[k[:4]][-1].update(_season=True, C=v.get("C") or k[:4])
    out = [(c, archive_entry(e) if isinstance(e, list) else e) for c, e in out]  # imgrecord10
    # one row per pass and card: a card listed twice (combined section and card entry) keeps the card entry, in its first place
    pos, kept = {}, []
    for c, e in out:
        if c in pos:
            i = pos[c]
            if kept[i][1].get("_combined") and not e.get("_combined"):
                kept[i] = (c, e)
            notes.append("%s %s: listed twice in the pass; the card's own entry used" % (name, c))
            continue
        pos[c] = len(kept)
        kept.append((c, e))
    return kept


def row_for(name, c, e, notes):
    hand = HAND.get((name, c))
    if hand and name in SEASON_PASSES and hand[0] != verdict_label(str(e.get("verdict") or "")):  # imgrecord14
        notes.append("%s %s: hand row says %s but the season rule gives %s; hand row not used, row rebuilt automatically "
                     "(look by hand)" % (name, c, hand[0], verdict_label(str(e.get("verdict") or ""))))
        hand = None
    if hand and FP.get((name, c)) == fp(e):
        v, ns, nm, ev = hand
        if ns is None:
            ns = num(e.get("nights_seen")) or num(e.get("n_nights_seen")) or 0
        if nm is None:
            nm = num(e.get("nights_expected_not_seen")) or 0
        return {"pass": name, "verdict": v, "evidence": ev, "nights_seen": ns, "nights_missed": nm}
    if hand:
        notes.append("%s %s: the pass's entry changed after imgrecord wrote its evidence; row rebuilt automatically" % (name, c))
    if name in ("deepframes", "deepframes2"):
        # irregular structures: no automatic builder; keep the hand row if there is one
        if hand:
            v, ns, nm, ev = hand
            return {"pass": name, "verdict": v, "evidence": ev, "nights_seen": ns, "nights_missed": nm}
        notes.append("%s %s: new entry with no hand row, left out (look at it by hand)" % (name, c))
        return None
    vtext = e.get("verdict") if isinstance(e.get("verdict"), str) else ""
    v = label_of(vtext)
    if v is None and word_label(vtext):
        v = word_label(vtext)
        notes.append("%s %s: verdict words %r... mapped to %s" % (name, c, vtext[:40], v))
    ns = num(e.get("nights_seen"))
    if ns is None:
        ns = num(e.get("n_nights_seen"))
    nm = num(e.get("nights_expected_not_seen"))
    if nm is None:
        nm = num(e.get("nights_missed"))
    if v is None:
        if re.match(r"(?i)\s*(keeps\s+)?(KNOWN|PROBABLY)\b", vtext):
            v = "SEEN AGAIN" if (ns or 0) > 0 else "UNSURE"
            notes.append("%s %s: identity verdict; image verdict taken from its nights (%s)" % (name, c, v))
        else:
            v = "UNSURE"
            notes.append("%s %s: verdict words not recognised (%r...), mapped to UNSURE" % (name, c, vtext[:40]))
    if v == "SEEN AGAIN (combined)" and not ns:
        ns = num(e.get("nights")) or 0
    ns = ns or 0
    nm = nm or 0
    txt = PREAMBLE_S.sub("", e["evidence"]) if isinstance(e.get("evidence"), str) else e.get("evidence")  # imgrecord10
    ev, how = fit_text(txt) if name in FIT_TEXT else (None, None)  # imgrecord7
    if not ev:
        ev, how = auto_evidence(v, e, ns, nm)
    notes.append("%s %s: evidence %s" % (name, c, how))
    return {"pass": name, "verdict": v, "evidence": ev, "nights_seen": ns, "nights_missed": nm}


# ---------------------------------------------------------------- main
def main():
    for line in FP_TEXT.strip().splitlines():
        p, c, h = line.split()
        FP[(p, c)] = h
    notes, passes, cards = [], [], {}
    fps = {}
    dup, duplog = load_dup(notes), []  # imgrecord8
    buildlog = []  # imgrecord9
    for name, what in PASSES:
        path = pass_path(name)  # imgrecord5: PASS_FILE
        rec = {"pass": name, "what": what}
        if name in PASS_FILE:
            rec["file"] = PASS_FILE[name]
        if name in READER:
            rec["read_as"] = READER[name]
        if name in NEED_HANDBACK and not os.path.exists(os.path.join(HERE, name + ".handback.md")):
            rec["status"] = "not read yet (pass still running: no hand-back yet)"  # imgrecord7: its file is not read at all
            passes.append(rec)
            continue
        if not os.path.exists(path):
            rec["status"] = "not written yet"
            passes.append(rec)
            continue
        try:
            d = json.load(open(path))
        except Exception as exc:
            rec["status"] = "unreadable (%s), skipped" % type(exc).__name__
            passes.append(rec)
            continue
        hb = os.path.exists(os.path.join(HERE, name + ".handback.md"))
        rec["status"] = "read" if hb else "read (pass still running: no hand-back yet)"
        got, skipped = [], []
        for c, e in entries_of(READER.get(name, name), d, notes):  # imgrecord5: READER
            if c is None:
                continue
            fps[(name, c)] = fp(e)
            if "--fp" in sys.argv:
                continue
            meas =e.get("measured") if isinstance(e.get("measured"), str) else ""
            vt = e.get("verdict") if isinstance(e.get("verdict"), str) else ""
            if (name, c) in SKIP or meas.startswith("No new image measurement") or \
                    re.match(r"(?i)\s*(skipped|not tested|not measured)\b", vt):  # e.g. chanceimg skips known-body cards
                skipped.append(c)
                continue
            row = row_for(name, c, e, notes)
            if row is None:
                skipped.append(c)
                continue
            if (name, c) in ERRATA:  # imgrecord4: a later pass found a fault in this entry; evidence replaced, verdict and nights kept
                if FP_ERRATA.get((name, c)) == fp(e):
                    row["evidence"] = ERRATA[(name, c)]["evidence"]
                    for k in ("verdict", "nights_seen", "nights_missed"):  # imgrecord6: an erratum may also correct these
                        if k in ERRATA[(name, c)]:
                            row[k] = ERRATA[(name, c)][k]
                    row["_erratum"] = True  # imgrecord7: the supersedes rule leaves this row alone (internal key, not written)
                    notes.append("%s %s: evidence corrected by erratum (%s)" % (name, c, ERRATA[(name, c)]["why"]))
                else:
                    notes.append("%s %s: the pass's entry changed after the erratum was written; erratum NOT applied (look by hand)" % (name, c))
            apply_dup(name, c, row, e, dup, notes, duplog)  # imgrecord8: duplicate warps, after the ERRATA step
            probs = check_text(row["evidence"])
            if probs:
                sys.exit("EVIDENCE RULE BROKEN, nothing written: %s %s: %s\n  text: %s" % (name, c, "; ".join(probs), row["evidence"]))
            if row["verdict"] not in LABELS:
                sys.exit("bad verdict %r for %s %s" % (row["verdict"], name, c))
            sup = e.get("supersedes")  # imgrecord7: internal keys for apply_supersedes, removed before writing
            sup = sup_names(sup)  # imgrecord11: the leading pass name ("allarch20 UNSURE" -> allarch20)
            if sup:
                row["_sup"], row["_why"] = sup, e.get("why") if isinstance(e.get("why"), str) else ""
            cards.setdefault(c, []).append(row)
            got.append(c)
        made = []
        if "--fp" not in sys.argv:
            for c, row in builder_rows(name, set(got) | set(skipped), notes):  # imgrecord9
                if check_text(row["evidence"]):
                    sys.exit("EVIDENCE RULE BROKEN, nothing written: %s %s (builder row): %s" % (name, c, row["evidence"]))
                cards.setdefault(c, []).append(row)
                got.append(c)
                made.append(c)
                buildlog.append({"card": c, "pass": name, "verdict": row["verdict"]})
        rec["cards"] = got
        if made:
            rec["rows_made_by_builder"] = made
        if skipped:
            rec["not_image_tests"] = skipped
        passes.append(rec)
    if "--fp" in sys.argv:
        for (p, c), h in fps.items():
            print(p, c, h)
        return
    notes.extend(CONF_LOG[k] for k in sorted(CONF_LOG, key=str))  # imgrecord12: every CONFIRMED part read
    notes.extend(SEASON_LOG[k] for k in sorted(SEASON_LOG, key=str))  # imgrecord13: every season part read
    superseded = apply_supersedes(cards, notes)  # imgrecord7: supersedes by rule
    nightlog = apply_night(cards, notes)  # imgrecord8: ps1night, after the supersedes rule
    fallslog = falls_sentence(cards, notes)  # imgrecord9: a FALLS row confirmed by a wide search keeps the card's stage
    orbitlog = orbit_sentences(cards, notes)  # imgrecord9: ps1dup2's orbit files, by rule
    for rs in cards.values():
        for r in rs:
            for k in ("_sup", "_why", "_erratum", "_dup", "_night", "_dupword"):
                r.pop(k, None)
    # counts and disagreements
    rows = [r for rs in cards.values() for r in rs]
    cnt_rows = {lab: sum(1 for r in rows if r["verdict"] == lab) for lab in LABELS}
    cnt_last = {lab: sum(1 for rs in cards.values() if rs[-1]["verdict"] == lab) for lab in LABELS}
    dis = []
    for c in sorted(cards):
        rs = cards[c]
        vs = {r["verdict"] for r in rs}
        seen = [r["pass"] for r in rs if r["verdict"].startswith("SEEN AGAIN")]
        notseen = [r["pass"] for r in rs if r["verdict"] == "NOT SEEN"]
        missed = [r["pass"] for r in rs if r["nights_missed"] > 0]
        unsure = [r["pass"] for r in rs if r["verdict"] in ("UNSURE", "NO POWER")]
        if seen and notseen:
            kind = "seen in one pass, not seen in another"
        elif seen and missed:
            kind = "seen, but missed on nights with power"
        elif seen and unsure and len(rs) > 1:
            kind = "seen in one pass, unsure or no power in another"
        else:
            continue
        dis.append({"card": c, "kind": kind, "verdicts": [[r["pass"], r["verdict"], r["nights_seen"], r["nights_missed"]] for r in rs]})
    out = {
        "about": ("Image tests of the candidate cards (TRKSUB01 double check): each row is one pass that measured the card at its "
                  "predicted place in archive images the search never used (DECam; where a row cites another telescope, its "
                  "evidence names it), in the order the passes ran. verdict: SEEN AGAIN "
                  "(stacked S/N 5 or more on a new night, controls clean) / SEEN AGAIN (combined) (only the combination of nights is "
                  "significant) / NOT SEEN / UNSURE / NO POWER (no usable exposure). nights_seen: new nights on which the body was seen; "
                  "nights_missed: nights with expected S/N 5 or more on which it was not seen. An identity result (KNOWN or PROBABLY a "
                  "catalogued body) is in the evidence, not the verdict. Evidence gives years only. Built by the imgrecord "
                  "script of the double check; re-run as passes finish."),
        "built": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "passes": passes,
        "cards": {c: cards[c] for c in sorted(cards)},
        "counts": {"cards": len(cards), "rows": len(rows), "rows_by_verdict": cnt_rows, "cards_by_latest_verdict": cnt_last},
        "disagreements": dis,
        "superseded": superseded,  # imgrecord7: what the supersedes rule did, row by row
        "duplicate_warps": duplog,  # imgrecord8: what the duplicate-warp rule did, row by row
        "ps1night": nightlog,  # imgrecord8: the ps1night sentences, card by card
        "duplicate_warps_both": list(DUP_BOTH),  # imgrecord9: rows named by ps1dup and ps1dup2 (ps1dup2 wins)
        "falls_confirmed": fallslog,  # imgrecord9: FALLS rows whose card a wide search confirms
        "orbit_sentences": orbitlog,  # imgrecord9: ps1dup2's orbit files, by rule
        "builder_rows": buildlog,  # imgrecord9: rows made from a wide pass's json where it wrote no image row
        "notes": notes,
    }
    # last guard: no forbidden text anywhere in the rows
    for c, rs in out["cards"].items():
        for r in rs:
            assert not check_text(r["evidence"]), (c, r)
    tmp = OUT + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, OUT)
    print("wrote %s: %d cards, %d rows; rows %s; disagreements %d; notes %d" % (
        OUT, len(cards), len(rows), json.dumps(cnt_rows), len(dis), len(notes)))


if __name__ == "__main__":
    main()
