# Placeholders

The code in this snapshot ran on our own machines with our own logins, allocations, paths and private pages. Every such
value was replaced by a placeholder (2026-11-24). This table says what each placeholder stands for. The real values are not in this
repository. To run a node, set each placeholder it uses to your own value. In shell scripts, `$NAME` placeholders are
read from the environment. In Python and JSON they are plain strings, so replace them in the code or read them from the
environment.

## Paths

| placeholder | meaning |
|---|---|
| `$ALE_HOME` | root of the project checkout on the workstation (the folder holding `planet-search/`, `rsi/decam-big-tno/`, `pages/`) |
| `$HOME` | the workstation user's home folder (for example `$HOME/anaconda3/bin/python`, `$HOME/work/projects/...`) |
| `$DATA_DRIVE` | the external data drive on the workstation (images, scenarios, cold copies); `$DATA_DRIVE/ale-astronomy` is a cold copy of `$ALE_HOME` |
| `$RSI_SANDBOX` | shared folder where task 1's agents and their runs worked (`$RSI_SANDBOX/r1`, `/r2`: per-generation work and tmp folders) |
| `$SHARED_DIR` | the workstation's shared folder that holds `$RSI_SANDBOX` |
| `$TMPDIR` | the agent harness's per-session temporary folder |
| `$NERSC_SCRATCH` | your scratch folder on NERSC Perlmutter |
| `$NERSC_HOME` | your home folder on NERSC |
| `$NERSC_CFS` | your project's community file system folder on NERSC |
| `$TACC_SCRATCH`, `$TACC_WORK`, `$TACC_HOME` | your scratch, work and home folders on TACC Vista / Lonestar6 |
| `$TACC_scratch_ROOT`, `$TACC_work_ROOT`, `$TACC_home1_ROOT` | the TACC folder one level up (the numeric user-id folder) |
| `$VM_WORK` | the work folder on the cloud VM that hosted Harbor trials |
| `$SSH_CM_DIR` | folder of the SSH ControlMaster sockets (one per site and login: `nersc-$NERSC_USER.sock`, `vista-$TACC_USER.sock`, `ls6-$TACC_USER.sock`) |
| `$PAGES_ROOT`, `$PAGES_OUT` | local source folder of the private progress pages and the folder of one page set |
| `$PAGES_PUBLISH` | the script that published those pages |
| `$PAGE_PUBLISHER` | the background job that published changed pages |
| `$DRIVE_FOLDER` | the cloud-drive folder that candidate packages were copied to |
| `$IMAP_CREDENTIALS_FILE`, `<pages-password-file>`, `<rubin-token-file>`, `$RUN_SECRETS_DIR`, `$RELAY_SETTINGS_FILE`, `<relay-settings-file>`, `$RELAY_KEY_DIR` | files that held credentials (never in this repository; the secret scanners in `campaign_sync.py` and `build_traces.py` read them only to redact their values from traces) |

## Accounts, hosts and ids

| placeholder | meaning |
|---|---|
| `$NERSC_USER` | login name at NERSC |
| `$TACC_USER` | login name at TACC (Vista, Lonestar6) |
| `$TACC_UID` | numeric TACC user-id folder |
| `$GITHUB_USER`, `$HF_USER`, `$EMAIL_USER`, `$GMAIL_USER` | the project lead's GitHub, Hugging Face, university-mail and Gmail user names |
| `TACC_USER\|GITHUB_USER\|...` (no `$`) | the same names inside a redaction regex (`build_traces.py`) |
| `$NERSC_GPU_ACCOUNT`, `$NERSC_CPU_ACCOUNT` | NERSC allocation (project) for GPU and CPU jobs |
| `$VISTA_ACCOUNT`, `$LS6_ACCOUNT`, `$EXPANSE_ACCOUNT` | allocations on TACC Vista, TACC Lonestar6 and SDSC Expanse |
| `$GCP_PROJECT` | the cloud project of the VM |
| `$VM_HOST` | SSH alias of the cloud VM that hosted Harbor trials and agents |
| `<login-node>`, `<mac-host>` | a cluster login node, the workstation's host name |
| `$GLOBUS_NERSC_COLLECTION`, `$GLOBUS_VISTA_COLLECTION`, `$GLOBUS_LS6_COLLECTION` | Globus collection ids of the three sites |
| `$GLOBUS_TASK_ID`, `<globus-task>` | one Globus transfer task |
| `<slurm-job-id>` | a Slurm job id quoted in a comment or log note |
| `$RELAY_ADDR`, `$RELAY_ADDR_2`, `$RELAY_PORT`, `$RELAY_PORT_2` | the local model relay (host:port) that held the model keys; agents' only network |
| `$RELAY_CLI` | the launcher that starts a headless agent through the relay (it reads the relay key; never in this repository) |
| `$HPC_TOOLS` | our cluster runbook's helper scripts (MFA proxy, SSH master keeper) and their logs |
| `$DOCKER_HOST_IP`, `$RELAY_FORWARD_PORT` | the VM's Docker bridge address and the port of its relay forwarder |
| `<coordinator-session>`, `<novelty-session>`, `<blog-session>` | names of the agent sessions that coordinated the work, checked leads and wrote the blog |
| `<pages-slug>`, `<publisher-job>` | the private page set's folder name and its publish job |
| `SITE_USER` | basic-auth user of the team's status site |

## People and pages

| placeholder | meaning |
|---|---|
| `PROJECT_LEAD_NAME`, `SENDER_NAME`, `SENDER_TITLE` | the project lead's name, as sender and ADES submitter/measurer, and the title in the email signature |
| `ALE Astronomy` | author of the Harbor tasks and of this repository |
| `OBSERVER_NAME`, `observer`, `observer@example.org` | an outside astronomer who received candidate batches (from pass 1) |
| `COLLEAGUE_NAME` | a colleague (from pass 1) |
| `account@example.org`, `sender@example.org`, `EPPN@access-ci.org` | email addresses (from pass 1) |
| `CLIENT_NAME`, `CLIENT_NAME_ZH`, `client_id` | a delivery client in forbidden-word guards (from pass 1) |
| `GMAIL_ID` | Gmail message and thread ids (from pass 1) |
| `<private-page-url>` | a private web page (progress trackers, write-ups) |
| `<team-site-url>`, `<team-site-domain>`, `team-site` | the team's password-gated status site; `nodes/pages/pages/team-site/` is its builder |
| `<blog-url>`, `<drive-folder-url>`, `<private-doc-url>`, `<hf-dataset>` | the blog, a cloud-drive folder, a private document, the private dataset of candidates |
| `<rubin-science-platform>` | the Rubin Science Platform (data access needs Rubin data rights) |
| `our data rights`, `approval` | wording of the Rubin data-rights guard, kept consistent between `verdicts.py` (writer) and the page builder (checker) |

## Data, grading and leads

| placeholder | meaning |
|---|---|
| `scen00` .. `scen14` | scenario ids (one per scenario, the same id everywhere): task 1's practice and sealed fields, task 2's regions |
| `REG_A`, `REG_B`, `REG_C` | the three regions held out in task 2's first run (graded regions); the other region ids are as they ran |
| `TRKSUB01` .. `TRKSUB03`, `TRKSUB`, `RUBIN_TRKSUB` | trkSub names of our filed bodies and of another survey's (from pass 1) |
| `KNOWN_BODY_F1`, `KNOWN_BODY_F3`, `KNOWN_BODY_F3B` | known bodies in task 1's sealed fields (from pass 1) |
| `C###` | a candidate card id |
| `<lead>`, `rXX`, `_pX` | a lead's field and distance tag |
| `LEAD01` .. `LEAD13` | lead ids (region, rank and score of a lead), one per lead, the same everywhere |
| `PLAN_DATES = []` | follow-up dates of an unannounced body, removed (`cand9_orbit.py`) |

Public facts were kept: the cluster login host names (`perlmutter.nersc.gov`, `vista.tacc.utexas.edu`, `ls6.tacc.utexas.edu`),
the NERSC Legacy Surveys staging folder (`/global/cfs/cdirs/cosmo/staging/decam`, readable by every NERSC user), MPC
observatory codes and their site coordinates, the NOIRLab Data Lab anonymous token (`nsc_fetch.py`, documented by
NOIRLab, no account), the Harbor canary GUID, and designations of public known bodies used as method examples.
