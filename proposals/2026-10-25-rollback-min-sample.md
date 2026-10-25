# Proposal: minimum sample before shadow-mode auto-rollback

**Status:** needs a human decision. **Touches:** section 8 rule 5 (protected). Implemented literally for now.

Section 8 says: roll back if the new champion's live rolling composite is worse than the old one's by more than 2%
during the 6-race shadow window. `src/f1/improve.py::check_rollback` applies this after **every** scored race,
including the first. On one race, winner log-loss alone swings by ±50%, so a single unlucky race will often trigger a
rollback of a change that passed a 169-race backtest gate.

Suggested change: evaluate the rollback rule only from the 3rd shadow race onward (or require a paired-bootstrap
probability ≥ 80% that the old champion is better). Not applied; awaiting approval.
