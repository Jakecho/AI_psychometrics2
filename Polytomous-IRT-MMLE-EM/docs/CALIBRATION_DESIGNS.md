# Calibration designs

| Design | Data | Fixed items | What fixes the metric | Estimated |
|---|---|---|---|---|
| Single-group | one group | none | reference distribution (mean; SD for GPCM/GRM) | all items; the Rasch SD |
| Concurrent | 2+ groups or forms | none | group 1 = reference | all items jointly; focal distributions |
| FIPC | one new group | bank anchors | anchor parameters | new items; the group distribution (MEM) |
| MG-FIPC | 2+ groups or forms | bank anchors | anchor parameters | free items jointly; every group distribution |

**Shared items.** One item column means one parameter vector. The item M-step sums the expected counts of every
group that answered the item, so common items cannot drift apart across groups. A test checks that a pooled
concurrent fit equals a single-group fit of the stacked data.

**Pooled vs group-specific priors.** With a pooled prior, one π is used for every group and updated from Σ_g n_gq,
which weights groups by sample size. With group-specific priors, each π_g is updated from its own n_gq only.

**FIPC methods** (Kim, 2006; irtQ `fipc.method`):
- **NWU–OEM**: the E-step uses the anchors only, under the initial prior, followed by one M-step for the new items.
  The run then stops. The prior is not updated; the distribution implied by the posteriors is reported (as irtQ
  does) but was not used for estimation. This is never described as converged.
- **MWU–MEM**: cycle 1 is the same as OEM. Later cycles use every item and update the group distribution in each
  cycle until convergence.

In both methods, anchor parameters are returned exactly unchanged (tested bit for bit), while their response
probabilities enter every posterior.

**Drift (page 11).** Free concurrent calibration absorbs anchor drift into the parameters and the group means.
Fixed-anchor calibration attributes undetected drift to the group's ability. Leave-one-out displacement frees each
anchor in turn, with the others fixed, and compares its estimated location with its bank value. Changes in item
parameters and changes in the population can only be separated if enough anchors are stable.
