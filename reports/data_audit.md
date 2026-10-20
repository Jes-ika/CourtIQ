# CourtIQ Phase 1: Data Quality & Reconnaissance Audit

**Project:** CourtIQ: Pressure-Aware Shot Quality and Decision Audit  
**Phase:** Phase 1 — Dataset Reconnaissance, Data-Quality Auditing & Schema Discovery  
**Evaluation Target Date:** October 2026  

---

## 1. Upstream Dataset & Licensing Provenance

### Source & Retrieval
- **Upstream Repository:** [SkillCorner Open Data — Basketball](https://github.com/SkillCorner/opendata-basketball)
- **Data Retrieval Date:** October 20, 2026
- **Target Competition:** Spanish Liga ACB 2025–2026 Season
- **Provided Data Artifacts:**
  - `matches/`: 10 sample games with game metadata, dynamic event intelligence JSON, and 25fps raw tracking data.
  - `aggregates/`: Season-level offensive aggregate CSVs covering 293 ACB games.
  - `player_id_aliases.csv`: Canonical identifier mapping for 9 players with multiple IDs.

### Upstream License & Data-Use Terms
- **Upstream License:** MIT License (Copyright (c) 2026 SkillCorner. https://skillcorner.com).
- **License Terms:** Grants standard MIT permissions to use, copy, modify, merge, publish, distribute, sublicense, and sell, conditioned upon preserving the copyright notice and disclaimer in all copies or substantial portions.
- **Upstream Attribution Condition:** SkillCorner requests:
  > *"If you use the data, we kindly ask that you credit SkillCorner and notify us on Twitter (@skillcorner) so we can follow the work being done with it."*
- **CourtIQ Redistribution Decision:**
  - Even though the MIT license allows redistribution, **CourtIQ project policy strictly prohibits committing raw source tracking files, dynamic event bundles, or downloaded archives to Git**.
  - All raw source data is isolated in local workspace directories excluded by `.gitignore`.
  - Redistribution status: **Restricted by project governance**. Only derived schema models, pipeline code, and summary statistical audit artifacts are tracked.

---

## 2. Dataset Architecture & File Structure

### File Hierarchy
```
opendata-basketball/
├── data/
│   ├── matches.json                      # Structured game metadata list
│   ├── player_id_aliases.csv             # Alias mapping for 9 duplicate player IDs
│   ├── aggregates/
│   │   ├── acb_shotsaggregates_20252026.csv   (293 games, offense-only)
│   │   ├── acb_drivesaggregates_20252026.csv
│   │   └── acb_picksaggregates_20252026.csv
│   └── matches/
│       ├── 114086/
│       ├── 114099/
│       ├── 114169/
│       ├── 114234/
│       ├── 114243/
│       ├── 178442/
│       ├── 179612/
│       ├── 184439/
│       ├── 188630/
│       └── 191313/
│           ├── {gameId}_game_data.json         # Rosters, team IDs, UTC tip-off
│           ├── {gameId}_dynamic_events.json    # Relational JSON with 20 event families
│           └── {gameId}_tracking_data.jsonl.gz # 25 fps broadcast tracking (Git LFS)
└── docs/
    ├── PRIMER.md
    ├── KNOWN_ISSUES.md
    ├── aggregates_columns.md
    └── data_dictionary/
```

### Event Families in Dynamic Events
Every `{gameId}_dynamic_events.json` delivers 20 relational arrays:
`chances`, `possessions`, `shots`, `rebounds`, `turnovers`, `free_throws`, `fouls`, `timeouts`, `passes`, `touches`, `dribbles`, `picks`, `isolations`, `handoffs`, `drives`, `posts`, `off_ball_screens`, `closeouts`, `chance_players`, `matchups`.

---

## 3. Schema & Shot Field Verification

Verified directly from real data files using `courtiq inspect`:

| Field | Inferred Type | Non-Null Count | Null % | Description & Usage |
|---|---|---|---|---|
| `id` | `str` | 1,459 | 0.0% | Primary key (unique per shot, e.g. `shot-114243-1-0`) |
| `gameId` | `int` | 1,459 | 0.0% | SkillCorner Game ID (10 distinct games) |
| `period` | `int` | 1,459 | 0.0% | Quarter/period (1 to 4 + OT) |
| `startFrame` | `int` | 1,459 | 0.0% | Release frame (25 fps video index) |
| `startWallClock`| `int` | 1,459 | 0.0% | Elapsed video time in ms |
| `startGameClock`| `float` | 1,459 | 0.0% | Game clock in seconds remaining |
| `location` | `list[float]` | 1,433 | 1.78% | `[x, y]` coordinate of shooter in feet at release |
| `distance` | `float` | 1,433 | 1.78% | Distance from shooter to hoop in feet at release |
| `three` | `bool` | 1,459 | 0.0% | True for 3-point attempt, False for 2-pointer |
| `shotType` | `str` | 1,433 | 1.78% | Jumper, floater, layup, post, heave |
| `shotClock` | `float` | 1,452 | 0.48% | Shot clock seconds remaining |
| `releaseTime` | `float` | 1,377 | 5.62% | Seconds from touch start to shooter release |
| `closestDefDist`| `float`| 1,433 | 1.78% | Distance in feet to closest defender at release |
| `contestLevel` | `str` | 1,433 | 1.78% | Open, light, average, plus, blocked |
| `contested` | `bool` | 1,459 | 0.0% | True if contestLevel is average, plus, or blocked |
| `outcome` | `bool` | 1,459 | 0.0% | Target: True = Scored, False = Missed |
| `fouled` | `bool` | 1,459 | 0.0% | True if shooter was fouled |
| `shotQuality` | `float` | 1,421 | 2.60% | SkillCorner proprietary quality model (0-100) |

---

## 4. Key Uniqueness, Join Integrity & Alignment

### Primary Key Uniqueness
- **Shot ID (`id`)**: 1,459 shot events across 10 games. Duplicate count: **0 (100% unique)**.

### Relational Join Coverage
- Evaluated against rosters in `{gameId}_game_data.json`:
  - `shooterId` join rate: **100.0%** (all 1,459 shooter IDs match roster player IDs).
  - `closestDefId` join rate: **100.0%** of non-null defenders match roster player IDs.
  - Foreign keys `chanceId`, `possessionId`, and `touchId` map 1-to-1 with dynamic event tables.

### Tracking Frame Alignment Verification
- Mathematical alignment confirmed against uncompressed tracking frame data in game `114243`:
  - `shot['startFrame'] == tracking_frame['frameIdx']` (Exact match at 25 fps).
  - Euclidean distance computed from tracking player coordinates `[x, y]` for shooter and closest defender matched `shot['closestDefDist']` with **0.0000 ft error**.
  - Documented tolerance: Frame tolerance = 0 frames; spatial tolerance = 0.05 ft.

---

## 5. Temporal Partitioning & Leakage Prevention

To prevent target leakage, all features are strictly audited by their availability relative to shot release:

### Pre-Shot Predictors (Permissible for M0 / M1)
1. **Spatial Baseline (M0)**:
   - `distance`: Shooter-to-hoop distance in feet.
   - `location`: 2D court coordinates `[x, y]`.
   - `three`: 2-point vs 3-point attempt indicator.
2. **Context & Mechanics (M0)**:
   - `shotType`: jumper, layup, floater, post, heave.
   - `shotClock`: Seconds remaining on shot clock.
   - `releaseTime`: Touch duration before release.
   - `dribblesBefore`: Dribble count leading to attempt.
   - `catchAndShoot`: Catch-and-shoot indicator.
   - `createdFromPaint`: Direct paint origination indicator.
   - `period`: Game period.
3. **Defensive Pressure Context (M1)**:
   - `closestDefDist`: Distance in feet to nearest defender at release.
   - `contestLevel`: Categorical contest intensity (`open`, `light`, `average`, `plus`, `blocked`).
   - `contested`: Binary contest flag.

### Post-Shot Features (Strict Blacklist)
- `outcome` (Target label)
- `shotQuality` (Proprietary SkillCorner model output)
- `fouled`, `foulId` (Whistle/foul outcome)
- `blocked`, `blockerId` (Shot blockage outcome)
- `assisted`, `assistOpp`, `passerId` (Assist designation)
- `endFrame`, `endWallClock`, `endGameClock` (Shot conclusion timestamps)

---

## 6. Audit Categorization: Facts, Assumptions, and Questions

### Observed Facts
1. The 10 sample games provide exactly **1,459 raw shot events**.
2. **26 shots (1.78%)** lack spatial coordinates (`location = None`, `distance = NaN`, `closestDefDist = NaN`). These are unlocated camera-blind possessions.
3. **143 shots (9.8%)** involve a fouled shooter. Of these, **111 are fouled misses** and **32 are made and-one shots**.
4. In official FIBA box scores, fouled misses award free throws and are not counted as Field Goal Attempts (FGA). Excluding them yields exactly **1,322 valid field-goal attempts**.
5. 9 of the 10 raw tracking files in git clone are Git LFS pointer files (133 bytes) unless explicitly smudged or pulled. However, the derived dynamic events files are fully self-contained (3.5 MB per game) and contain the exact release frame index and pre-computed defender distance.

### Assumptions
1. We assume that the 10 ACB regular-season and playoff games in the SkillCorner release are representative of elite European professional basketball shooting environments.
2. We assume that FIBA shot accounting (excluding fouled misses from FGA) is the clinically correct population definition for shot-quality estimation.
3. We assume that computer-vision tracking error (mean `predError` ~1.5–1.9 ft) is distributed symmetrically across defenders and shooters without systematic bias towards either team.

### Unresolved Questions
1. Does the 10-game sample size provide sufficient power to detect subtle non-linear interactions (e.g., defender closing speed or angle of contest)?
2. Are defender contest categorizations (`open` vs `light` vs `average`) subject to human annotator variance or algorithmic threshold drift across venues?
3. How will models trained on Spanish Liga ACB data generalize to NBA or NCAA court geometries and three-point line distances (FIBA 6.75m vs NBA 7.24m)?
