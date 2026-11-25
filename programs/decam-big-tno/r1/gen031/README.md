gen031: gen022's finder.py + "wide" repeats of the window passes (seeds down to S/N 4.5, spare time only). See NOTES.md.

- finder.py: the program (entry point). `FINDER_WSEED=0` gives gen022 back.
- dev_tools/plant_faint.py, dev_tools/cls.py: development tools (planter with the sky's magnitude slope, scorer of a
  planted run); not imported by finder.py.
