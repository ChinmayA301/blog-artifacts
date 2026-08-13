# Approach B — persona-conditioned cascade probabilities

## Methodological target

The original experiment holds the Independent Cascade activation probability
constant. That makes the topology real but the behavior homogeneous. Approach
B asks whether seed choice changes when activation becomes conditional on both
the receiving persona and the content topic.

For content type `c`, the prototype defines:

```text
logit p(u → v | c)
  = logit p₀
  + β₁ affinity(v, c)
  + β₂ share_propensity(v)
  + β₃ affinity_similarity(u, v, c)
```

The probability is clipped to `[0.005, 0.60]`. The coefficients are synthetic
behavioral parameters and belong in the sensitivity analysis; they are not
estimated from Higgs retweet events.

Two initial content profiles exercise the mechanism:

- `ml_paper`: machine-learning familiarity, technology/science interest,
  technical literacy, need for cognition, and reading preference;
- `sports_clip`: sports interest/exposure, video preference, excitement
  seeking, and social-media attitude.

BO searches four interpretable seed-rule weights: topology, topic affinity,
sharing propensity, and neighborhood redundancy. This keeps the search space
small enough for a Gaussian-process surrogate while allowing the winning seed
set to change by topic.

## Conditioning sanity check

A random global persona sample is not a model of Twitter during the July 2012
Higgs event. The implementation therefore separates two decisions:

1. **Persona-pool conditioning.** Exclude records below the platform age floor,
   then softly weight the remaining rows by a declared platform-fit score.
2. **Persona-to-node coupling.** Map sampled platform-fit rank to observed graph
   activity rank through a Gaussian-copula prior.

The coupling strength is unidentified. The required grid is:

| Coupling | Meaning |
|---:|---|
| `0.00` | random persona-to-node assignment |
| `0.35` | weak activity/platform-fit alignment |
| `0.70` | strong alignment stress test |

This is not a parameter-tuning opportunity. A result that appears only at one
coupling is assumption-sensitive and must be reported that way.

## Gate before a full Higgs run

The public 999-row sample is sufficient to validate schema compatibility and
exercise the method. It is **not** sufficient to claim that the synthetic
population resembles the 2012 Higgs Twitter population. The audit therefore
returns:

- **GO:** synthetic sensitivity experiment;
- **NO-GO:** empirical behavioral claim about Higgs users.

A stronger empirical study would need platform-era calibration targets,
observed content-topic labels, or repeated edge-level retweet opportunities
from which heterogeneous probabilities could be estimated.

## Comparisons to report

For each topic and coupling:

1. top-degree seeds under the conditional cascade;
2. persona-conditioned BO seeds under the same cascade;
3. mean spread with independent high-precision Monte Carlo evaluation;
4. ML-vs-sports seed-set overlap;
5. stability of rank and lift across coupling values.

Do not promote the largest lift across the grid as the headline. The robust
finding, if any, is the effect that survives the entire plausible grid.
