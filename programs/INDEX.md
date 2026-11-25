# Programs: every generation's submitted program

One folder per generation, exactly as submitted, minus the agent's notes (`NOTES*.md`), bytecode, `*.orig` copies and the
agent's own test outputs (`experiments/*.txt`, `experiments/*.json`). Files over 20 MB would be `<name>.omitted.txt` (size and
sha256); none were. Rewards are the public ones only (no per-scenario or per-body results).

- `planet-search/r1`, `planet-search/r2`: task 1 (`nodes/rsi-loop`), exported with `git archive` from each run's
  `programs.git` tag. Reward = weighted recall points over the 10 sealed fields (0 to 1).
- `decam-big-tno/r1`, `decam-big-tno/r2`: task 2 (`nodes/climb`), copied from each generation's Harbor artifact
  `artifacts/submission/`. r1 reward = weighted recall in the global top 2 x findable; r2 has two rewards, recall and
  GPU-hours over the four graded regions (lower is better), never mixed. Status `invalid` = scored 0 by the static scan
  (network code or a designation in the program) or no output.

| task | run | gen | model (agent, effort) | public reward | GPU-hours | status | files | bytes |
|---|---|---|---|---|---|---|---|---|
| planet-search | r1 | gen000 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0 |  | 2 of 10 scenarios failed or timed out | 1 | 8887 |
| planet-search | r1 | gen001 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0827 |  | ok | 10 | 60842 |
| planet-search | r1 | gen002a | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0827 |  | ok | 11 | 67742 |
| planet-search | r1 | gen002b | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0301 |  | 1 of 10 scenarios failed or timed out | 11 | 73401 |
| planet-search | r1 | gen002c | Claude Opus 5.5 (Claude Code, effort xhigh) |  |  |  | 11 | 69405 |
| planet-search | r1 | gen003a | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0827 |  | ok | 11 | 69381 |
| planet-search | r1 | gen003b | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0827 |  | ok | 11 | 68946 |
| planet-search | r1 | gen003c | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0827 |  | ok | 11 | 68501 |
| planet-search | r1 | gen004a | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1128 |  | ok | 11 | 69252 |
| planet-search | r1 | gen004b | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1128 |  | ok | 11 | 69796 |
| planet-search | r1 | gen004c | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0827 |  | ok | 11 | 70205 |
| planet-search | r1 | gen005a | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1128 |  | ok | 12 | 83675 |
| planet-search | r1 | gen005b | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1128 |  | ok | 11 | 72260 |
| planet-search | r1 | gen005c | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1128 |  | ok | 12 | 87140 |
| planet-search | r1 | gen006a | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1128 |  | ok | 13 | 100878 |
| planet-search | r1 | gen006b | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0902 |  | ok | 13 | 97638 |
| planet-search | r1 | gen006c | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1128 |  | ok | 12 | 88889 |
| planet-search | r1 | gen007a | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1128 |  | ok | 12 | 93386 |
| planet-search | r1 | gen007b | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0827 |  | ok | 12 | 90067 |
| planet-search | r1 | gen007c | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0827 |  | ok | 12 | 90499 |
| planet-search | r1 | gen008a | Claude Opus 5.5 (Claude Code, effort xhigh) |  |  |  | 13 | 103014 |
| planet-search | r1 | gen008b | Claude Opus 5.5 (Claude Code, effort xhigh) |  |  |  | 12 | 94691 |
| planet-search | r1 | gen008c | Claude Opus 5.5 (Claude Code, effort xhigh) |  |  |  | 14 | 107170 |
| planet-search | r1 | gen009a | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1128 |  | ok | 12 | 97006 |
| planet-search | r1 | gen009b | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1128 |  | ok | 12 | 94774 |
| planet-search | r1 | gen009c | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0 |  | 4 of 10 scenarios failed or timed out | 4 | 36565 |
| planet-search | r1 | gen009d | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0977 |  | ok | 4 | 42132 |
| planet-search | r1 | gen009e | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0451 |  | ok | 4 | 38090 |
| planet-search | r1 | gen010a | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 17 | 143379 |
| planet-search | r1 | gen010b | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0902 |  | ok | 17 | 142218 |
| planet-search | r1 | gen010c | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 135751 |
| planet-search | r1 | gen010d | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1128 |  | ok | 17 | 143161 |
| planet-search | r1 | gen010e | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 135557 |
| planet-search | r1 | gen011a | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 137576 |
| planet-search | r1 | gen011b | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 137410 |
| planet-search | r1 | gen011c | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 137653 |
| planet-search | r1 | gen011d | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 137516 |
| planet-search | r1 | gen011e | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 138275 |
| planet-search | r1 | gen012a | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 140002 |
| planet-search | r1 | gen012b | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 139568 |
| planet-search | r1 | gen012c | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 143717 |
| planet-search | r1 | gen012d | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 142402 |
| planet-search | r1 | gen012e | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 139538 |
| planet-search | r1 | gen013a | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 142495 |
| planet-search | r1 | gen013b | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 140671 |
| planet-search | r1 | gen013c | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 15 | 152401 |
| planet-search | r1 | gen013d | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 140712 |
| planet-search | r1 | gen013e | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 142449 |
| planet-search | r1 | gen014a | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 143623 |
| planet-search | r1 | gen014b | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 143419 |
| planet-search | r1 | gen014c | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 13 | 143299 |
| planet-search | r1 | gen015 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 16 | 216752 |
| planet-search | r1 | gen016 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 18 | 232620 |
| planet-search | r1 | gen017 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 18 | 232620 |
| planet-search | r1 | gen018 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 18 | 233969 |
| planet-search | r1 | gen019 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 18 | 239135 |
| planet-search | r1 | gen020 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1278 |  | ok | 20 | 243725 |
| planet-search | r1 | gen021 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1128 |  | ok | 20 | 250442 |
| planet-search | r1 | gen022 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1278 |  | ok | 20 | 243725 |
| planet-search | r1 | gen023 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1278 |  | ok | 20 | 244173 |
| planet-search | r1 | gen024 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1278 |  | ok | 20 | 244173 |
| planet-search | r1 | gen025 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1278 |  | ok | 20 | 244704 |
| planet-search | r1 | gen026 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 20 | 249513 |
| planet-search | r1 | gen027 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 21 | 250952 |
| planet-search | r1 | gen028 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1278 |  | ok | 20 | 245036 |
| planet-search | r1 | gen029 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1429 |  | ok | 20 | 245376 |
| planet-search | r1 | gen030 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1429 |  | ok | 20 | 257127 |
| planet-search | r1 | gen031 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1429 |  | ok | 20 | 260766 |
| planet-search | r1 | gen032 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1429 |  | ok | 20 | 261694 |
| planet-search | r1 | gen033 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1429 |  | ok | 20 | 266986 |
| planet-search | r1 | gen034 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1429 |  | ok | 20 | 271665 |
| planet-search | r1 | gen035 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1429 |  | ok | 20 | 273553 |
| planet-search | r1 | gen036 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1429 |  | ok | 20 | 275905 |
| planet-search | r1 | gen037 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1429 |  | ok | 21 | 279179 |
| planet-search | r1 | gen038 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1429 |  | ok | 22 | 281845 |
| planet-search | r1 | gen039 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1053 |  | ok | 23 | 284618 |
| planet-search | r1 | gen040 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1429 |  | ok | 26 | 294463 |
| planet-search | r1 | gen041 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 26 | 294482 |
| planet-search | r1 | gen042 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1203 |  | ok | 27 | 297002 |
| planet-search | r1 | gen043 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 26 | 294482 |
| planet-search | r1 | gen044 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1429 |  | ok | 26 | 295958 |
| planet-search | r1 | gen045 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 26 | 294520 |
| planet-search | r1 | gen046 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 26 | 296775 |
| planet-search | r1 | gen047 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1429 |  | ok | 26 | 299683 |
| planet-search | r1 | gen048 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 26 | 297331 |
| planet-search | r1 | gen049 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 26 | 297331 |
| planet-search | r1 | gen050 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 26 | 297331 |
| planet-search | r1 | gen051 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 27 | 302941 |
| planet-search | r1 | gen052 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 27 | 304103 |
| planet-search | r1 | gen053 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 27 | 304142 |
| planet-search | r1 | gen054 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 27 | 305135 |
| planet-search | r1 | gen055 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 27 | 305135 |
| planet-search | r1 | gen056 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 27 | 306036 |
| planet-search | r1 | gen057 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 27 | 306900 |
| planet-search | r1 | gen058 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 28 | 310663 |
| planet-search | r1 | gen059 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 28 | 312289 |
| planet-search | r1 | gen060 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 28 | 313679 |
| planet-search | r1 | gen061 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 28 | 313679 |
| planet-search | r1 | gen062 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 28 | 314498 |
| planet-search | r1 | gen063 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 28 | 324222 |
| planet-search | r1 | gen064 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 28 | 327167 |
| planet-search | r1 | gen065 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 28 | 328757 |
| planet-search | r1 | gen066 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 29 | 339190 |
| planet-search | r1 | gen067 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 30 | 367505 |
| planet-search | r1 | gen068 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 30 | 373886 |
| planet-search | r1 | gen069 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 30 | 374223 |
| planet-search | r1 | gen070 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1579 |  | ok | 30 | 374338 |
| planet-search | r1 | gen071 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.188 |  | ok | 30 | 377375 |
| planet-search | r1 | gen072 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1729 |  | ok | 30 | 379210 |
| planet-search | r1 | gen073 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.188 |  | ok | 30 | 378943 |
| planet-search | r1 | gen074 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.188 |  | ok | 30 | 379744 |
| planet-search | r1 | gen075 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.188 |  | ok | 30 | 382266 |
| planet-search | r1 | gen076 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.188 |  | ok | 30 | 383513 |
| planet-search | r1 | gen077 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.188 |  | ok | 30 | 385447 |
| planet-search | r1 | gen078 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.188 |  | ok | 31 | 386891 |
| planet-search | r1 | gen079 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.188 |  | ok | 31 | 388659 |
| planet-search | r1 | gen080 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.188 |  | ok | 32 | 402001 |
| planet-search | r1 | gen081 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.188 |  | ok | 34 | 406900 |
| planet-search | r1 | gen082 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.188 |  | ok | 35 | 417667 |
| planet-search | r1 | gen083 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.188 |  | ok | 35 | 421097 |
| planet-search | r1 | gen084 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 35 | 426270 |
| planet-search | r1 | gen085 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 35 | 428656 |
| planet-search | r1 | gen086 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 37 | 438373 |
| planet-search | r1 | gen087 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 43 | 474552 |
| planet-search | r1 | gen088 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 46 | 495317 |
| planet-search | r1 | gen089 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.188 |  | ok | 51 | 519175 |
| planet-search | r1 | gen090 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 49 | 527202 |
| planet-search | r1 | gen091 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 54 | 539903 |
| planet-search | r1 | gen092 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 55 | 542861 |
| planet-search | r1 | gen093 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 57 | 551918 |
| planet-search | r1 | gen094 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.203 |  | ok | 57 | 552836 |
| planet-search | r1 | gen095 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 57 | 551918 |
| planet-search | r1 | gen096 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 59 | 571054 |
| planet-search | r1 | gen097 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 59 | 571178 |
| planet-search | r1 | gen098 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 59 | 571178 |
| planet-search | r1 | gen099 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 59 | 571554 |
| planet-search | r1 | gen100 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 59 | 571554 |
| planet-search | r1 | gen101 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 59 | 571854 |
| planet-search | r1 | gen102 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 59 | 573648 |
| planet-search | r1 | gen103 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 60 | 592634 |
| planet-search | r1 | gen104 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.203 |  | ok | 70 | 668607 |
| planet-search | r1 | gen105 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 73 | 735406 |
| planet-search | r1 | gen106 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 77 | 770313 |
| planet-search | r1 | gen107 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 79 | 794018 |
| planet-search | r1 | gen108 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.218 |  | ok | 82 | 811803 |
| planet-search | r1 | gen109 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2331 |  | ok | 85 | 877052 |
| planet-search | r1 | gen110 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2331 |  | ok | 92 | 894433 |
| planet-search | r1 | gen111 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.203 |  | ok | 95 | 928478 |
| planet-search | r1 | gen112 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2331 |  | ok | 94 | 923275 |
| planet-search | r1 | gen113 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2331 |  | ok | 102 | 963406 |
| planet-search | r1 | gen114 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2331 |  | ok | 113 | 987072 |
| planet-search | r1 | gen115 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2331 |  | ok | 117 | 1006672 |
| planet-search | r1 | gen116 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2331 |  | ok | 118 | 1022237 |
| planet-search | r1 | gen117 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.203 |  | ok | 122 | 1074658 |
| planet-search | r1 | gen118 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2331 |  | ok | 123 | 1041160 |
| planet-search | r1 | gen119 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.1955 |  | ok | 124 | 1046232 |
| planet-search | r1 | gen120 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2331 |  | ok | 124 | 1044188 |
| planet-search | r1 | gen121 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2331 |  | ok | 128 | 1051959 |
| planet-search | r1 | gen122 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2331 |  | ok | 130 | 1060294 |
| planet-search | r1 | gen123 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2331 |  | ok | 132 | 1067334 |
| planet-search | r1 | gen124 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2406 |  | ok | 134 | 1078747 |
| planet-search | r1 | gen125 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2406 |  | ok | 136 | 1086673 |
| planet-search | r1 | gen126 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2331 |  | ok | 139 | 1098155 |
| planet-search | r1 | gen127 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2406 |  | ok | 142 | 1115895 |
| planet-search | r1 | gen128 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2406 |  | ok | 145 | 1131631 |
| planet-search | r1 | gen129 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2406 |  | ok | 151 | 1177557 |
| planet-search | r1 | gen130 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2105 |  | ok | 152 | 1184855 |
| planet-search | r1 | gen131 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.203 |  | ok | 155 | 1190632 |
| planet-search | r1 | gen132 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2406 |  | ok | 155 | 1200333 |
| planet-search | r1 | gen133 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2406 |  | ok | 162 | 1218004 |
| planet-search | r1 | gen134 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2406 |  | ok | 167 | 1235594 |
| planet-search | r1 | gen135 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2406 |  | ok | 168 | 1242454 |
| planet-search | r1 | gen136 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2707 |  | ok | 170 | 1264559 |
| planet-search | r1 | gen137 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2632 |  | ok | 172 | 1275639 |
| planet-search | r1 | gen138 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2707 |  | ok | 172 | 1279311 |
| planet-search | r1 | gen139 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2707 |  | ok | 175 | 1303259 |
| planet-search | r1 | gen140 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2707 |  | ok | 180 | 1334666 |
| planet-search | r1 | gen141 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2632 |  | ok | 184 | 1374392 |
| planet-search | r1 | gen142 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2707 |  | ok | 182 | 1340361 |
| planet-search | r1 | gen143 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2707 |  | ok | 191 | 1600328 |
| planet-search | r1 | gen144 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2632 |  | ok | 194 | 1604659 |
| planet-search | r1 | gen145 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2632 |  | ok | 195 | 1618217 |
| planet-search | r1 | gen146 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2707 |  | ok | 198 | 1629951 |
| planet-search | r1 | gen147 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.0376 |  | ok | 204 | 1649611 |
| planet-search | r1 | gen148 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2632 |  | ok | 206 | 1669026 |
| planet-search | r1 | gen149 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2632 |  | ok | 205 | 1662928 |
| planet-search | r1 | gen150 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2632 |  | ok | 209 | 1692802 |
| planet-search | r1 | gen151 | Claude Opus 5.5 (Claude Code, effort xhigh) | 0.2707 |  | ok | 204 | 1655565 |
| planet-search | r2 | gen001 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.1429 |  | ok | 1 | 32944 |
| planet-search | r2 | gen002 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.2105 |  | ok | 1 | 53793 |
| planet-search | r2 | gen003 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3008 |  | ok | 1 | 66008 |
| planet-search | r2 | gen004 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3008 |  | ok | 1 | 71084 |
| planet-search | r2 | gen005 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3308 |  | ok | 1 | 85532 |
| planet-search | r2 | gen006 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3609 |  | ok | 5 | 108588 |
| planet-search | r2 | gen007 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3684 |  | ok | 11 | 344973 |
| planet-search | r2 | gen008 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.391 |  | ok | 11 | 355906 |
| planet-search | r2 | gen009 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3684 |  | ok | 13 | 355034 |
| planet-search | r2 | gen010 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3684 |  | ok | 17 | 372199 |
| planet-search | r2 | gen011 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3684 |  | ok | 22 | 393330 |
| planet-search | r2 | gen012 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3684 |  | ok | 29 | 412176 |
| planet-search | r2 | gen013 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3684 |  | ok | 29 | 425395 |
| planet-search | r2 | gen014 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3684 |  | ok | 31 | 432980 |
| planet-search | r2 | gen015 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3985 |  | ok | 33 | 440944 |
| planet-search | r2 | gen016 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3985 |  | ok | 34 | 465837 |
| planet-search | r2 | gen017 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3985 |  | ok | 34 | 475456 |
| planet-search | r2 | gen018 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3985 |  | ok | 34 | 481573 |
| planet-search | r2 | gen019 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3985 |  | ok | 34 | 483184 |
| planet-search | r2 | gen020 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3985 |  | ok | 34 | 485194 |
| planet-search | r2 | gen021 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3684 |  | ok | 36 | 505771 |
| planet-search | r2 | gen022 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3985 |  | ok | 38 | 498696 |
| planet-search | r2 | gen023 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3985 |  | ok | 34 | 489022 |
| planet-search | r2 | gen024 | Claude Fable 5.2 (Claude Code, effort xhigh) | 0.3985 |  | ok | 36 | 511567 |
| decam-big-tno | r1 | gen000 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.0 |  | valid | 5 | 70942 |
| decam-big-tno | r1 | gen001 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.2828 |  | valid | 5 | 95867 |
| decam-big-tno | r1 | gen002 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.3838 |  | valid | 5 | 116660 |
| decam-big-tno | r1 | gen003 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.404 |  | valid | 10 | 152864 |
| decam-big-tno | r1 | gen004 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.404 |  | valid | 16 | 177788 |
| decam-big-tno | r1 | gen005 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.4444 |  | valid | 20 | 195198 |
| decam-big-tno | r1 | gen006 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.4444 |  | valid | 22 | 208158 |
| decam-big-tno | r1 | gen007 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.4545 |  | valid | 23 | 221552 |
| decam-big-tno | r1 | gen008 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.4646 |  | valid | 25 | 243927 |
| decam-big-tno | r1 | gen009 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.4747 |  | valid | 30 | 254513 |
| decam-big-tno | r1 | gen010 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5253 |  | valid | 38 | 293623 |
| decam-big-tno | r1 | gen011 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.4949 |  | valid | 37 | 295555 |
| decam-big-tno | r1 | gen012 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5657 |  | valid | 46 | 333858 |
| decam-big-tno | r1 | gen013 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5657 |  | valid | 48 | 336502 |
| decam-big-tno | r1 | gen014 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5657 |  | valid | 52 | 350893 |
| decam-big-tno | r1 | gen015 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5253 |  | valid | 39 | 306591 |
| decam-big-tno | r1 | gen016 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5455 |  | valid | 49 | 355080 |
| decam-big-tno | r1 | gen017 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5455 |  | valid | 51 | 362245 |
| decam-big-tno | r1 | gen018 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5657 |  | valid | 52 | 355258 |
| decam-big-tno | r1 | gen019 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5859 |  | valid | 49 | 356608 |
| decam-big-tno | r1 | gen020 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5758 |  | valid | 62 | 377882 |
| decam-big-tno | r1 | gen021 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5859 |  | valid | 60 | 387845 |
| decam-big-tno | r1 | gen022 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5859 |  | valid | 11 | 201920 |
| decam-big-tno | r1 | gen023 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5859 |  | valid | 9 | 215272 |
| decam-big-tno | r1 | gen024 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5859 |  | valid | 18 | 222517 |
| decam-big-tno | r1 | gen025 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5758 |  | valid | 16 | 243640 |
| decam-big-tno | r1 | gen026 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5859 |  | valid | 6 | 201788 |
| decam-big-tno | r1 | gen027 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5758 |  | valid | 2 | 216125 |
| decam-big-tno | r1 | gen028 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5859 |  | valid | 4 | 208862 |
| decam-big-tno | r1 | gen029 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5859 |  | valid | 16 | 233425 |
| decam-big-tno | r1 | gen030 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5859 |  | valid | 8 | 230295 |
| decam-big-tno | r1 | gen031 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5859 |  | valid | 4 | 194464 |
| decam-big-tno | r1 | gen032 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5758 |  | valid | 7 | 220069 |
| decam-big-tno | r1 | gen033 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5657 |  | valid | 7 | 229231 |
| decam-big-tno | r1 | gen034 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5657 |  | valid | 8 | 244272 |
| decam-big-tno | r1 | gen035 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5859 |  | valid | 8 | 233300 |
| decam-big-tno | r1 | gen036 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.6061 |  | valid | 11 | 255012 |
| decam-big-tno | r2 | gen000 | run r1's gen019 (Claude Fable 5.2), regraded under r2's rules | 0.479 | 1.9411 | valid | 49 | 356608 |
| decam-big-tno | r2 | gen001 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5389 | 1.9131 | valid | 2 | 239438 |
| decam-big-tno | r2 | gen002 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5689 | 1.9284 | valid | 2 | 251210 |
| decam-big-tno | r2 | gen003 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5868 | 1.9115 | valid | 2 | 275921 |
| decam-big-tno | r2 | gen004 | GPT 6 Astra (codex, effort xhigh) | 0.5928 | 1.9289 | valid | 2 | 271196 |
| decam-big-tno | r2 | gen005 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.5928 | 1.9263 | valid | 2 | 280423 |
| decam-big-tno | r2 | gen006 | GPT 6 Astra (codex, effort xhigh) | 0.6287 | 1.9253 | valid | 3 | 316444 |
| decam-big-tno | r2 | gen007 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.6587 | 1.9228 | valid | 3 | 324839 |
| decam-big-tno | r2 | gen008 | GPT 6 Astra (codex, effort xhigh) | 0.6647 | 1.9306 | valid | 4 | 337314 |
| decam-big-tno | r2 | gen009 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.6647 | 1.9299 | valid | 4 | 344586 |
| decam-big-tno | r2 | gen010 | GPT 6 Astra (codex, effort xhigh) | 0.6707 | 1.9264 | valid | 7 | 355601 |
| decam-big-tno | r2 | gen011 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.6826 | 1.9278 | valid | 7 | 367330 |
| decam-big-tno | r2 | gen012 | GPT 6 Astra (codex, effort xhigh) | 0.6826 | 1.926 | valid | 9 | 365667 |
| decam-big-tno | r2 | gen013 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.6826 | 1.9262 | valid | 10 | 375397 |
| decam-big-tno | r2 | gen014 | GPT 6 Astra (codex, effort xhigh) | 0.7066 | 1.9256 | valid | 11 | 392408 |
| decam-big-tno | r2 | gen015 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.7066 | 1.9266 | valid | 11 | 398288 |
| decam-big-tno | r2 | gen016 | GPT 6 Astra (codex, effort xhigh) | 0.7006 | 1.9245 | valid | 16 | 413869 |
| decam-big-tno | r2 | gen017 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.7066 | 1.5509 | valid | 11 | 396386 |
| decam-big-tno | r2 | gen018 | GPT 6 Astra (codex, effort xhigh) | 0.7066 | 1.9266 | valid | 15 | 415216 |
| decam-big-tno | r2 | gen019 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.7066 | 1.7761 | valid | 11 | 405023 |
| decam-big-tno | r2 | gen020 | GPT 6 Astra (codex, effort xhigh) | 0.7066 | 1.9267 | valid | 14 | 430377 |
| decam-big-tno | r2 | gen021 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.7066 | 1.5016 | valid | 10 | 396887 |
| decam-big-tno | r2 | gen022 | GPT 6 Astra (codex, effort xhigh) | 0.7186 | 1.8237 | valid | 17 | 425264 |
| decam-big-tno | r2 | gen023 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.7246 | 1.7766 | valid | 17 | 433352 |
| decam-big-tno | r2 | gen024 | GPT 6 Astra (codex, effort xhigh) | 0.7186 | 1.7764 | valid | 20 | 441945 |
| decam-big-tno | r2 | gen025 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.7305 | 1.8229 | valid | 17 | 449543 |
| decam-big-tno | r2 | gen026 | GPT 6 Astra (codex, effort xhigh) | 0.7305 | 1.8498 | valid | 21 | 454199 |
| decam-big-tno | r2 | gen027 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.7305 | 1.8777 | valid | 18 | 451689 |
| decam-big-tno | r2 | gen028 | GPT 6 Astra (codex, effort xhigh) | 0.7305 | 1.8179 | valid | 22 | 477020 |
| decam-big-tno | r2 | gen029 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.7305 | 1.8188 | valid | 23 | 481028 |
| decam-big-tno | r2 | gen030 | GPT 6 Astra (codex, effort xhigh) | 0.7066 | 1.8196 | valid | 24 | 485116 |
| decam-big-tno | r2 | gen031 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.7365 | 1.8202 | valid | 24 | 499128 |
| decam-big-tno | r2 | gen032 | GPT 6 Astra (codex, effort xhigh) | 0.7365 | 1.8135 | valid | 23 | 480426 |
| decam-big-tno | r2 | gen033 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.7425 | 1.8287 | valid | 25 | 510557 |
| decam-big-tno | r2 | gen034 | GPT 6 Astra (codex, effort xhigh) | 0.7605 | 1.8239 | valid | 27 | 509860 |
| decam-big-tno | r2 | gen035 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.7365 | 1.8236 | valid | 27 | 514470 |
| decam-big-tno | r2 | gen036 | GPT 6 Astra (codex, effort xhigh) | 0.7605 | 1.8327 | valid | 28 | 515196 |
| decam-big-tno | r2 | gen037 | Claude Fable 5.2 (claude-code, effort xhigh) | 0.7545 | 1.8187 | valid | 28 | 513888 |
| decam-big-tno | r2 | gen038 | GPT 6 Astra (codex, effort xhigh) | 0.7485 | 1.8297 | valid | 32 | 533388 |
| decam-big-tno | r2 | gen039 | GPT 6.2 Astra (codex, effort max) | 0.7844 | 1.8374 | valid | 30 | 564150 |
| decam-big-tno | r2 | gen040 | GPT 6.2 Astra (codex, effort max) | 0.7365 | 1.5009 | valid | 35 | 625148 |
| decam-big-tno | r2 | gen041 | GPT 6.2 Astra (codex, effort max) | 0.7904 | 1.8712 | valid | 40 | 672969 |
| decam-big-tno | r2 | gen042 | GPT 6.2 Astra (codex, effort max) | 0.8024 | 1.8811 | valid | 46 | 740044 |
| decam-big-tno | r2 | gen043 | GPT 6.2 Astra (codex, effort max) | 0.8084 | 1.9148 | valid | 55 | 821471 |
| decam-big-tno | r2 | gen044 | GPT 6.2 Astra (codex, effort max) | 0.8204 | 1.886 | valid | 63 | 893569 |
| decam-big-tno | r2 | gen045 | GPT 6.2 Astra (codex, effort max) | 0.7964 | 1.8929 | valid | 68 | 966744 |
| decam-big-tno | r2 | gen046 | GPT 6.2 Astra (codex, effort max) | 0.7904 | 1.9067 | valid | 71 | 996387 |
| decam-big-tno | r2 | gen047 | GPT 6.2 Astra (codex, effort max) | 0.8024 | 1.8994 | valid | 75 | 1034569 |
| decam-big-tno | r2 | gen048 | GPT 6.2 Astra (codex, effort max) | 0.8084 | 1.8891 | valid | 71 | 972856 |
| decam-big-tno | r2 | gen049 | GPT 6.2 Astra (codex, effort max) | 0.8263 | 1.9086 | valid | 76 | 1002011 |
| decam-big-tno | r2 | gen050 | GPT 6.2 Astra (codex, effort max) | 0.8084 | 1.8953 | valid | 77 | 1009901 |
| decam-big-tno | r2 | gen051 | GPT 6.2 Astra (codex, effort max) | 0.8144 | 1.8903 | valid | 78 | 1032214 |
| decam-big-tno | r2 | gen052 | GPT 6.2 Astra (codex, effort max) | 0.7964 | 1.9195 | valid | 78 | 1015892 |
| decam-big-tno | r2 | gen053 | GPT 6.2 Astra (codex, effort max) | 0.8263 | 1.9014 | valid | 78 | 1021982 |
| decam-big-tno | r2 | gen054 | GPT 6.2 Astra (codex, effort max) | 0.8263 |  | unchanged | 78 | 1021982 |
| decam-big-tno | r2 | gen055 | GPT 6.2 Astra (codex, effort max) | 0.8263 |  | unchanged | 78 | 1021982 |
| decam-big-tno | r2 | gen056 | GPT 6.2 Astra (codex, effort max) | 0.8263 | 1.8907 | valid | 79 | 1031655 |
| decam-big-tno | r2 | gen057 | GPT 6.2 Astra (codex, effort max) | 0.8144 | 1.9077 | valid | 88 | 1097081 |
| decam-big-tno | r2 | gen058 | GPT 6.2 Astra (codex, effort max) | 0.8263 | 1.8824 | valid | 90 | 1132440 |
| decam-big-tno | r2 | gen059 | GPT 6.2 Astra (codex, effort max) | 0.8323 | 1.8881 | valid | 96 | 1194398 |
| decam-big-tno | r2 | gen060 | GPT 6.2 Astra (codex, effort max) | 0.8144 | 1.8755 | valid | 105 | 1275575 |
| decam-big-tno | r2 | gen061 | GPT 6.2 Astra (codex, effort max) | 0.8204 | 1.8904 | valid | 110 | 1297659 |
| decam-big-tno | r2 | gen062 | GPT 6.2 Astra (codex, effort max) | 0.8144 | 1.9112 | valid | 107 | 1281210 |
| decam-big-tno | r2 | gen063 | GPT 6.2 Astra (codex, effort max) | 0.8323 | 1.8931 | valid | 103 | 1262601 |
| decam-big-tno | r2 | gen064 | GPT 6.2 Astra (codex, effort max) | 0.8263 | 1.8947 | valid | 108 | 1305086 |
| decam-big-tno | r2 | gen065 | GPT 6.2 Astra (codex, effort max) | 0.8323 | 1.8852 | valid | 103 | 1262494 |
| decam-big-tno | r2 | gen066 | GPT 6.2 Astra (codex, effort max) | 0.8323 | 1.8939 | valid | 110 | 1308339 |
| decam-big-tno | r2 | gen067 | GPT 6.2 Astra (codex, effort max) | 0.8383 | 1.8909 | valid | 117 | 1390359 |
| decam-big-tno | r2 | gen068 | GPT 6.2 Astra (codex, effort max) | 0.8323 | 1.8965 | valid | 129 | 1513515 |
| decam-big-tno | r2 | gen069 | GPT 6.2 Astra (codex, effort max) | 0.8263 | 1.8987 | valid | 130 | 1538158 |
| decam-big-tno | r2 | gen070 | GPT 6.2 Astra (codex, effort max) | 0.8263 | 1.9036 | valid | 138 | 1609520 |
| decam-big-tno | r2 | gen071 | GPT 6.2 Astra (codex, effort max) | 0.8323 | 1.8939 | valid | 124 | 1480215 |
| decam-big-tno | r2 | gen072 | GPT 6.2 Astra (codex, effort max) | 0.8263 | 1.9441 | valid | 133 | 1535908 |
| decam-big-tno | r2 | gen073 | GPT 6.2 Astra (codex, effort max) | 0.8323 | 1.9029 | valid | 132 | 1510430 |
| decam-big-tno | r2 | gen074 | GPT 6.2 Astra (codex, effort max) | 0.8263 | 1.9276 | valid | 141 | 1582989 |
| decam-big-tno | r2 | gen075 | GPT 6.2 Astra (codex, effort max) | 0.8323 | 1.8976 | valid | 127 | 1522702 |

Total: 325 programs, 169852361 bytes, 0 files omitted for size.
