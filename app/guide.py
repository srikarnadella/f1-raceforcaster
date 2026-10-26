"""The "?" guide: what each page answers, how to read it, and what the terms mean.
Rendered by ui.help_button() in the top-right corner of every page."""

PAGES = {
    "": {
        "title": "Next race",
        "answers": "Who is likely to be fast at the next race, and why, before and during the weekend.",
        "read": [
            "**Team pace projection**: each team's expected one-lap gap to the fastest team at this circuit. "
            "Built from where each car is strong (straights, slow, medium and fast corners) multiplied by how much "
            "of this lap is made of each.",
            "**Why**: the same projection split by corner type. Bars left of zero gain time against the field; right lose it.",
            "**Pre-weekend win probabilities**: qualifying simulated hundreds of times from the projection, then each "
            "grid raced by the champion model. Replaced by the logged forecast once qualifying is done.",
            "**Strategy projection**: the fastest pit plans from a tyre-wear model of earlier visits. "
            "'S23 → M25' means 23 laps on softs then 25 on mediums.",
        ],
    },
    "weekend": {
        "title": "Weekend lab",
        "answers": "What happened in every session of one weekend: who was quick over one lap and over a long run.",
        "read": [
            "Pick a weekend at the top; the tabs split it into views.",
            "**Session pace**: best lap per driver per session, as % gap to the fastest. Ideal lap = sum of a driver's best sectors.",
            "**Long runs**: practice race simulations, corrected for tyre compound and tyre age. Fuel loads are secret, "
            "so gaps under about 0.3% are noise.",
            "**Tyres**: how much lap time each team loses per lap of tyre age, by compound. Green = gentle on tyres.",
            "**Run programmes**: practice runs grouped automatically (machine learning) into race sims, quali sims and short runs.",
            "**Telemetry**: speed traces of the fastest qualifying laps, and where on the lap each driver lost time to pole.",
            "**Track dominance**: the circuit drawn from GPS, each section coloured by the fastest team through it.",
        ],
    },
    "race": {
        "title": "Race story",
        "answers": "How one race unfolded: positions, gaps, strategies and true race pace.",
        "read": [
            "**Lap chart**: position after every lap. Pick drivers to highlight; everyone else is grey.",
            "**Gap to leader**: seconds behind the leader each lap. Jumps are pit stops; bunching is a safety car.",
            "**Tyre strategy**: one bar per driver, coloured by compound (red soft, yellow medium, grey hard, green inter, blue wet).",
            "**Race pace**: each driver's pace in clean air, corrected for fuel burn, compound and tyre age, so it "
            "shows car + driver speed rather than where they happened to be.",
        ],
    },
    "next": {
        "title": "Logged forecast",
        "answers": "The official pre-race forecast, written to the permanent log after qualifying and before the start.",
        "read": [
            "Win, podium and top-10 probabilities and expected finishing position for every driver.",
            "All numbers come from one simulation of 10,000 races, so they agree with each other.",
            "Only forecasts logged before the race count towards the model's real accuracy.",
        ],
    },
    "last": {
        "title": "Post-race review",
        "answers": "How the last forecast did, and what the weekend taught the model.",
        "read": ["Pace summary by team and driver, the biggest misses with hedged explanations, and takeaways for the model."],
    },
    "driver": {
        "title": "Drivers",
        "answers": "How one driver is doing in a season, and how they compare with the teammate in the same car.",
        "read": [
            "Pick a season, a circuit type (to see only, say, street circuits) and a driver.",
            "**Head-to-heads** count races where the driver beat the teammate in qualifying and in the result.",
            "**Grid to flag**: hollow dot = starting slot, filled dot = finish; the line is places gained or lost.",
            "**Qualifying gap to teammate**: below zero = faster than the teammate that weekend.",
            "**Driver profile**: one-lap pace, race pace, tyre management, consistency, starts, attacking and "
            "defending, each against the teammate and then compared with the grid; the archetype is found by clustering.",
            "**Where the lap is won**: time vs the teammate by corner type, from qualifying telemetry. Same car, "
            "so the difference is the driver (and set-up).",
        ],
    },
    "constructor": {
        "title": "Constructors",
        "answers": "How one team's car performs through a season, by circuit type and by corner type.",
        "read": [
            "**Pace through the season**: gap to the best team each round, in qualifying or race pace.",
            "**By circuit type**: average gap at each kind of track (see Circuit types).",
            "**Car strengths**: time vs the field median by corner type; below zero = a strength.",
            "**Tyre wear**: seconds lost per lap of tyre age vs the field; negative = kinder to tyres.",
        ],
    },
    "track": {
        "title": "Tracks",
        "answers": "What a circuit rewards, who has won there, and how races there are usually run.",
        "read": [
            "**Winners and pole sitters** since 2014, and where winners started (a tall bar at 1 = pole matters).",
            "**Overtakes per lap** counts real on-track passes, not pit-stop shuffles.",
            "**Qualifying order by year** and the **track map** of the fastest team through each section.",
        ],
    },
    "teams": {
        "title": "Team DNA",
        "answers": "Where each car is fast and slow: straights, fast, medium and slow corners.",
        "read": [
            "**Time view**: % time lost or gained per zone vs the field median, from each team's fastest qualifying lap. "
            "Green cells are strengths.",
            "**Speed view**: apex speed in corners and top speed on straights vs the median; a purer grip and drag signal.",
            "**Saturday vs Sunday**: below the dashed line = relatively better in the race than in qualifying.",
            "**Car character map**: cars close together gain and lose time in similar places.",
        ],
    },
    "drivers": {
        "title": "Driver DNA",
        "answers": "Driver traits with the car taken out: every number is against the teammate.",
        "read": [
            "Negative % = faster than the teammate. 'Race minus quali' positive = relatively stronger on Sundays.",
            "Lap 1 gain = places gained from the grid to the end of lap 1. Overtakes = green-flag passes.",
        ],
    },
    "circuits": {
        "title": "Circuit types",
        "answers": "Which circuits behave alike, found by machine learning rather than by hand.",
        "read": [
            "Each dot is a circuit, placed by its profile (time on straights and in each corner type, speed, braking, "
            "tyre wear, safety cars, passing). Nearby dots are similar tracks; colours are the groups found.",
        ],
    },
    "unsupervised": {
        "title": "Unsupervised lab",
        "answers": "Patterns found without labels: drivers at each track, driver and car types, odd weekends.",
        "read": [
            "**Track affinity**: a driver's qualifying gap to the teammate at each track minus their usual gap. "
            "Left = better than their own norm there. Coloured bars are statistically notable; grey ones are within noise.",
            "**Driver × track type**: the same by circuit type, the level where history does predict the next visit a little.",
            "**Latent map**: matrix factorisation puts drivers and tracks in one space; near = tends to do well there.",
            "**Driver archetypes**: clustering of driver traits (pace, tyres, starts, racecraft).",
            "**Car character** and **Anomalies**: car types by where they gain time, and the strangest team weekends.",
        ],
    },
    "season": {
        "title": "2026: predicting winners",
        "answers": "Which pre-race signals found this season's winners, and when a prediction can be trusted.",
        "read": [
            "**Which signal finds the winner**: how often each signal's top pick won (qualifying, practice, profiles, models).",
            "**How sure can we be?**: a confidence model predicts whether the favourite will win; bars show how often "
            "it was right at each confidence level.",
            "**What makes a favourite safe**: factors that raise (right) or lower (left) the favourite's chances.",
            "**Race by race**: every 2026 favourite, the model's and the confidence model's numbers, and what happened.",
        ],
    },
    "eras": {
        "title": "Regulation eras",
        "answers": "How much the current rules change each track, and what to expect where they have not raced yet.",
        "read": [
            "Every era (2017–21, 2022–25, 2026) is measured separately with the same rules for counting.",
            "**Era factors**: new rules ÷ old rules at the same tracks. 1.15× passes = 15% more overtaking per track.",
            "**Old vs new**: each track's passes per lap under both rule sets.",
            "**What makes a track good for passing**: a model of passes per lap from the track's layout, so tracks the "
            "new rules have not visited yet still get an estimate.",
            "**Every track**: the estimate used by projections and strategy; 'How' says whether it comes from races "
            "under these rules, the previous era scaled, or the layout model.",
        ],
    },
    "championship": {
        "title": "Championship",
        "answers": "Title chances for drivers and constructors, updated after every race.",
        "read": ["Probabilities come from simulating the remaining races thousands of times with current line-ups."],
    },
    "ml": {
        "title": "ML lab",
        "answers": "Do the models and measures actually work on races they never saw?",
        "read": [
            "**Why race winners are missed**: the champion's misses split into causes, and how often a favourite "
            "given X% actually wins (calibration).",
            "**Race simulator**: driver profiles raced lap by lap thousands of times; compared with the champion, a "
            "blend of both and the grid on seasons it was not tuned on.",
            "**Ways of measuring pace**: which measure of race, qualifying and practice pace best matches results "
            "('agrees') and holds from one race to the next ('repeats').",
            "**Race-winner models**: the champion against the neural net and practice-informed challengers, "
            "each race predicted using only earlier races (walk-forward).",
            "**Race-pace projection**, **team pace projection** and **strategy** checks compare each tool with a simple baseline.",
            "**Anomalies**: team weekends far from that team's usual form.",
        ],
    },
    "scorecard": {
        "title": "Scorecard",
        "answers": "The model's real record: forecasts logged before races, scored after them.",
        "read": ["Compared with simple baselines (pole sitter wins, finish in grid order). With few races the numbers are noisy."],
    },
    "models": {
        "title": "Model log",
        "answers": "Which model is in charge, what was tried, and what needs a human decision.",
        "read": ["Champion and shadow versions, every experiment with its result, and pending proposals."],
    },
}

GLOSSARY = [
    ("Gap %", "How much slower than the fastest, as a percentage of lap time. 0.1% ≈ 0.08 s on an 80 s lap."),
    ("Ideal lap", "A driver's three best sectors added together, even if set on different laps."),
    ("Long run", "Five or more laps at race pace on one set of tyres in practice: a race simulation."),
    ("Clean air", "A lap started more than 1.5 s behind the car ahead, so it is not slowed by traffic."),
    ("Corrected race pace", "Lap time with fuel burn, tyre compound and tyre age taken out by a statistical fit."),
    ("Tyre wear (deg)", "Seconds lost per lap as a tyre ages. 0.05 s/lap over 20 laps = 1 s slower at the end."),
    ("Compounds", "S soft (fastest, wears quickest), M medium, H hard, I intermediate, W wet."),
    ("Corner types", "Slow: apex under 130 km/h. Medium: 130–210. Fast: over 210. Straights: full throttle, no brake."),
    ("Team DNA", "A team's strength in each corner type, averaged over recent weekends with recent ones weighted more."),
    ("Teammate-relative", "Compared with the other driver in the same car, so the car's speed cancels out."),
    ("SC / VSC", "Safety car / virtual safety car: the field slows, gaps shrink and pit stops get cheaper."),
    ("Pit-stop loss", "Time lost making a stop compared with staying out, measured from real stops."),
    ("Passing ease", "Overtakes per lap scaled so 100% ≈ Monza or Spa; low means track position is king."),
    ("Champion / challenger", "The champion makes the official forecast; challengers are logged next to it and can "
                              "replace it only by passing the promotion test."),
    ("Walk-forward", "Testing that predicts each race using only races before it, as in real life."),
    ("Log-loss / Brier", "Scores for probability forecasts; lower is better. Spearman: rank agreement, higher is better."),
    ("Regulation era", "A period under one set of technical rules (2017–21, 2022–25, 2026). Form and track behaviour "
                       "are measured per era and scaled between eras."),
    ("Era factor", "How a measure changes at the same tracks from one era to the next, e.g. 1.15× passes per lap."),
    ("Neural net", "A small network that scores the whole field at once (listwise ranking), running as a challenger."),
]
