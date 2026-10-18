# Relevance proxy validation

- Labels produced by: **LLM judge**
- Pairs exported: **150**
- Labelled: **150** (22 unsure, excluded)
- Raw agreement: **80%** of 128 decisive pairs
- Cohen's kappa: **0.62**

## Where they disagree

| | LLM judge: relevant | LLM judge: not relevant |
|---|---|---|
| **Proxy: relevant** | 56 | 22 |
| **Proxy: not relevant** | 3 | 47 |

- **22** pairs the tag overlap called relevant and llm judge did not — chunks on the right topic that would not help this student. This is the proxy's expected failure mode.
- **3** pairs llm judge called relevant and the tags missed — usually a chunk addressing the student's confusion whose `findings` tags do not name the confirmed category.

## By aspect

| Aspect | Agreement |
|---|---|
| `differential` | 83% (24/29) |
| `management` | 50% (1/2) |
| `pitfall` | 78% (31/40) |
| `radiographic_sign` | 87% (39/45) |
| `technique` | 67% (8/12) |

Raw agreement is inflated when one label dominates, which it does here: most retrieved chunks on a five-finding corpus are topically related, so a proxy answering "relevant" to everything would still score highly. Quote kappa if the two figures disagree.

