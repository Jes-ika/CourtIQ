# CourtIQ Phase 1: Data Dictionary

**Project:** CourtIQ: Pressure-Aware Shot Quality and Decision Audit  
**Dataset:** SkillCorner Open Data — Basketball (Spanish Liga ACB 2025–2026)  
**Coordinate System:** Court origin `(0, 0)` at court center. Coordinates and distances in **feet**. Offensive hoop located at negative x-axis. 25 frames per second (fps).  

---

## 1. Shot Event Entity (`shots`)

Grain: One row per field-goal attempt. Primary Key: `id`.

| Field Name | Storage Type | Units / Format | Allowed Values | Nullable | Availability | Definition & Modeling Role |
|---|---|---|---|---|---|---|
| `id` | `string` | Identifier | `shot-{gameId}-{period}-{idx}` | No | Pre-shot | Unique primary key identifying the shot event. |
| `gameId` | `integer` | Identifier | 6-digit integer | No | Pre-shot | SkillCorner match ID. Used as group identifier in grouped validation. |
| `period` | `integer` | Ordinal | 1, 2, 3, 4, 5 (OT) | No | Pre-shot | Period of play. M0 categorical feature. |
| `startFrame` | `integer` | Frames | Positive integer | No | Pre-shot | Video frame (at 25 fps) at the exact moment of ball release. |
| `startWallClock` | `integer` | Milliseconds | Positive integer | No | Pre-shot | Elapsed video time from video start to release. |
| `startGameClock` | `float` | Seconds | 0.0 to 600.0 | No | Pre-shot | Game clock time remaining in period at release. |
| `location` | `list[float]` | Feet | `[x, y]` | Yes (1.78%) | Pre-shot | 2D court coordinates of shooter at release. Origin (0,0) at court center. |
| `distance` | `float` | Feet | 0.0 to 80.0+ | Yes (1.78%) | Pre-shot | Euclidean distance from shooter coordinates to hoop center at release. Core M0 feature. |
| `region` | `string` | Categorical | `left corner three`, `key`, `ra`, `middle two`, etc. | Yes (1.78%) | Pre-shot | Coarse discrete zone of shooter. |
| `shotClock` | `float` | Seconds | 0.0 to 24.0 | Yes (0.48%) | Pre-shot | Shot clock remaining at release. M0 numeric feature. |
| `shooterId` | `integer` | Identifier | Player ID | No | Pre-shot | ID of the shooter, joining to rosters in `game_data`. |
| `offTeamId` | `integer` | Identifier | Team ID | No | Pre-shot | ID of offensive team. |
| `defTeamId` | `integer` | Identifier | Team ID | No | Pre-shot | ID of defending team. |
| `three` | `boolean` | Flag | `True` (3pt), `False` (2pt) | No | Pre-shot | 3-point attempt indicator. Core M0 feature. |
| `shotType` | `string` | Categorical | `jumper`, `layup`, `floater`, `post`, `heave` | Yes (1.78%) | Pre-shot | Primary shot execution classification. M0 categorical feature. |
| `complexShotType` | `string` | Categorical | `dribblePullUp`, `catchAndShoot`, `stepback`, `postFadeaway`, etc. | Yes (1.78%) | Pre-shot | Fine-grained mechanics classification. |
| `catchAndShoot` | `boolean` | Flag | `True`, `False` | No | Pre-shot | True if releaseTime <= 1.5s or complexShotType is catchAndShoot. M0 feature. |
| `createdFromPaint`| `boolean` | Flag | `True`, `False` | No | Pre-shot | True if shot directly originated from a touch/pass in paint. M0 feature. |
| `dribblesBefore` | `integer` | Count | 0 to 20+ | No | Pre-shot | Number of dribbles taken by shooter prior to release. M0 feature. |
| `releaseTime` | `float` | Seconds | 0.0 to 10.0+ | Yes (5.62%) | Pre-shot | Seconds from shooter touch inception to release. M0 feature. |
| `closestDefId` | `integer` | Identifier | Player ID | Yes (1.78%) | Pre-shot | Player ID of closest defending player at release. |
| `closestDefDist`| `float` | Feet | 0.0 to 30.0+ | Yes (1.78%) | Pre-shot | Distance in feet to nearest defender at release. Core M1 feature. |
| `contestLevel` | `string` | Categorical | `open`, `light`, `average`, `plus`, `blocked` | Yes (1.78%) | Pre-shot | SkillCorner contest intensity classification. Core M1 feature. |
| `contested` | `boolean` | Flag | `True`, `False` | No | Pre-shot | True if contestLevel is average, plus, or blocked. M1 feature. |
| `outcome` | `boolean` | Binary | `True` (Made), `False` (Missed)| No | Post-shot | **Target Variable**. Encoded as 1 (Made) vs 0 (Missed). |
| `fouled` | `boolean` | Flag | `True`, `False` | No | Post-shot | **Leakage Blacklist**. Indicates shooter was fouled on attempt. |
| `blocked` | `boolean` | Flag | `True`, `False` | No | Post-shot | **Leakage Blacklist**. Indicates attempt was blocked. |
| `assisted` | `boolean` | Flag | `True`, `False` | No | Post-shot | **Leakage Blacklist**. Indicates attempt was assisted. |
| `shotQuality` | `float` | Score (0-100) | 0.0 to 100.0 | Yes (2.60%) | Post-shot | **Leakage Blacklist**. Proprietary SkillCorner model score. |

---

## 2. Tracking Frame Entity (`tracking_data`)

Grain: One row per video frame at 25 fps.

| Field Name | Type | Description |
|---|---|---|
| `frameIdx` | `integer` | Video frame index starting from 0. Lines up with `startFrame` in `shots`. |
| `wallClock` | `integer` | Elapsed video time in milliseconds. |
| `gameClock` | `float` | Period game clock in seconds. |
| `period` | `integer` | Quarter / period index. |
| `homePlayers` | `list[object]` | List of tracked home team players. |
| `awayPlayers` | `list[object]` | List of tracked away team players. |
| `ball` | `object` | Ball 3D position `xyz` and speed. |

### Tracking Player Sub-Object
| Field Name | Type | Description |
|---|---|---|
| `playerId` | `integer` | SkillCorner player identifier. |
| `xyz` | `list[float]` | 3D court position `[x, y, z]` in feet. z = 0.0 for players. |
| `speed` | `float` | Velocity magnitude in feet per second. |
| `isDetected` | `integer` | `1` if detected on screen by computer vision; `0` if extrapolated. |
| `predError` | `float` | Calibration 90th percentile positional error bound in feet. |

---

## 3. Facts, Assumptions, and Unresolved Questions

### Observed Facts
- 26 shot attempts have missing values across all spatial coordinates (`distance`, `location`, `closestDefDist`, `contestLevel`).
- The coordinate system is origin-centered at court center `(0, 0)`. The attacking basket is always normalized to the negative x axis.
- Tracking player coordinates are strictly in feet.

### Assumptions
- Shot release frame `startFrame` accurately reflects the ballistic departure of the ball from the shooter's hands within 1 frame (40ms).
- Missing `releaseTime` (5.62%) occurs on instantaneous put-backs and tip-ins where touch start was not tracked.

### Unresolved Questions
- Does `contestLevel = 'blocked'` imply that the contest evaluation was formed post-release or during the release window? (In our pipeline, `contestLevel` is evaluated only for pre-shot pressure; `blocked` as a post-shot outcome flag is strictly blacklisted).
