# UX Test Notes (Local Only)

Use this file to track A/B results while testing locally.

- Branch: `main`
- Push status: `DO NOT PUSH YET`
- Test mode: `UX_LAB_MODE=1`
- Date: `2026-06-29`

## Local Run Command

```bash
UX_LAB_MODE=1 streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

## Test Scenarios

1. Missing image flow
2. Normal generation (Stability)
3. Normal generation (Tripo)
4. Invalid vertex count
5. Filename with special characters
6. Apply last successful settings
7. Post-generation checklist behavior

## A/B Session Log

| Scenario | A: Current UX Result | B: Enhanced UX Result | Better Variant | Notes |
|---|---|---|---|---|
| Missing image flow |  |  |  |  |
| Normal generation (Stability) |  |  |  |  |
| Normal generation (Tripo) |  |  |  |  |
| Invalid vertex count |  |  |  |  |
| Filename special characters |  |  |  |  |
| Apply last successful settings |  |  |  |  |
| Post-generation checklist |  |  |  |  |

## Metrics Snapshot (Optional)

- Total runs in A: 
- Total runs in B: 
- Success rate A: 
- Success rate B: 
- Avg time-to-first-success A: 
- Avg time-to-first-success B: 

## Decision Draft

- Recommended variant to keep: 
- Reasons:
  - 
  - 
- Issues to fix before deploy:
  - 
  - 

## Next Actions (No Push Yet)

- [ ] Finalize findings
- [ ] Confirm enhancement list
- [ ] Approve commit plan
- [ ] Push only after explicit go-ahead
