# Snurt (Non‑Ad Topic Memory) — Markdown Spec

> **Purpose (one line):** Snurt is your per‑topic **interest meter** that **goes up** on meaningful interactions and **fades** with time. You use it to rank which topics the user cares about *now*.

---

## 1) What Snurt Examines (Signals)

Each interaction event on a topic produces a quality‑weighted “bump” $S_{\text{reg}}$ built from cleaned, capped signals:

* **Dwell time** $d$: time engaged with the topic (capped; ignores tiny dwells).
* **Session frequency** $f_{\text{session}}$: repeat interactions within a session/day.
* **Lifetime relevance memory** $R$: slow memory that accumulates past interest with its own half‑life.
* **Prompt length** $L$: longer inputs imply effort/engagement.
* **Search/click utility**: reward reasonable use of suggestions (only if suggestions were provided).
* **Gratitude/affect** $S_{\text{grateful}}$: tiny positive/negative nudge in $[-1,1]$.
* **Verbosity multiplier** $v$: small user‑level factor $\in [0.8,1.2]$ to avoid over‑rewarding wordiness.

All raw signals are **cleaned** via gating, capping, log scaling, robust standardization, and a sigmoid to yield comparable pieces in $[0,1]$.

---

## 2) Reasoning Behind the Weights

* **External vs. Internal (≈ 30% / 70%)**

  * *External* (dwell, session frequency, lifetime R) reflect observable behavior and return patterns.
  * *Internal* (prompt length, click utility, gratitude) capture intention/quality of engagement.
  * The 30/70 split weights intent/quality a bit more than raw time—preventing a single long idle page from dominating.
* **Click utility cap (≤ \~20% of Internal)**

  * Clicks are easy to game and noisy; we cap their contribution so they *help* but don’t dominate.
* **Half‑life choices**

  * Snurt fades on a practical horizon (e.g., 14–30 days): fast enough to track current interests, slow enough to avoid jitter.
  * Lifetime memory $R$ fades more slowly (e.g., 30–90 days) to reflect durable interests.
* **Robust stats**

  * We use median/MAD when topics are few to neutralize outliers; standard $\mu/\sigma$ when topics are many.

You can later learn or personalize these weights from outcome signals (e.g., “did the user return to this topic tomorrow?”), but these defaults make Snurt useful immediately.

---

## 3) Overall Architecture

**Per topic $t$ we store:**

* Raw Snurt memory: $S_t$
* Last update timestamp: $\text{last\_updated}_t$
* Lifetime memory: $R_t$ (for the $s_{\text{rel}}$ signal)

**Units & Decay:**

* Time unit $U =$ **days** (use **fractions** for sub‑day intervals; e.g., 1 hour = $1/24$ day).
* Snurt half‑life $h_{\text{nurt}}$; decay rate $\lambda = \ln 2 / h_{\text{nurt}}$.
* Lifetime memory half‑life $h_{\text{rel}}$; $\lambda_{\text{rel}} = \ln 2 / h_{\text{rel}}$.

**High‑level flow per event:**

1. **Decay** $S_t$ from its last timestamp to now.
2. **Compute** a cleaned, capped event bump $S_{\text{reg}}\in[0,1]$ from signals.
3. **Add** $\eta\,S_{\text{reg}}$ ($\eta$ is a small learning rate).
4. **Update** timestamps.

**Display time:**

* Decay $S_t$ to “now”, normalize across topics via **z‑score** (or robust z), then apply a **sigmoid** to map to $[0,1]$ for UI ranking.

**Separation from ads:**

* Snurt tracks **non‑ad** interest only. Ad memory (Saurt) is separate and combined later in the ad path; keeping them separate prevents bias.

---

## 4) Formulas — Building Each Sub‑Item

### 4.1 Helper transforms

* **Gate tiny dwells:** only count if above a small threshold $T_{\min}$ (e.g., 5–10s).
  $\mathbf{1}[d \ge T_{\min}]$
* **Cap dwell:**
  $d_{\cap} = \min(d,\; \tau_{\text{cap}}), \quad \tau_{\text{cap}}\in[120,180]\;\text{s}$
* **Log scale:**
  $\ell(x) = \log(1 + x)$
* **Standardize (mean/std or robust median/MAD):**
  $z(x) = \frac{x - \mu}{\sigma + \varepsilon} \quad \text{or} \quad z_{\text{robust}}(x) = \frac{x - \text{median}}{1.4826\cdot \text{MAD} + \varepsilon}$
* **Sigmoid to $[0,1])$:**
  $\sigma(x) = \frac{1}{1 + e^{-x}}$
* **Clip:**
  $\operatorname{clip}(x,0,1) = \min(\max(x,0),\,1)$

### 4.2 Per‑signal scores (all end in $[0,1])$

* **Dwell:**
  $s_{\text{dur}} = \sigma\!\big( z\big(\ell(d_{\cap})\big) \big)\; \cdot\; \mathbf{1}[d \ge T_{\min}]$

* **Session frequency:**
  $s_{\text{freq}} = \sigma\!\big( z\big(\ell(f_{\text{session}})\big) \big)$

* **Lifetime relevance memory:** maintain $R$ with its own half‑life $h_{\text{rel}}$:
  $R \leftarrow R\cdot e^{-\lambda_{\text{rel}}\,\Delta} + r_k, \qquad \lambda_{\text{rel}} = \frac{\ln 2}{h_{\text{rel}}}$
  $s_{\text{rel}} = \sigma\!\big( z\big(\ell(R)\big) \big)$

* **Prompt length:**
  $s_{\text{plen}} = \sigma\!\big( z\big(\ell(L)\big) \big)$

* **Search/click utility** (only if suggestions were provided, $\text{provided}>0$):
  $r_{\text{click}} = \frac{\text{clicked}}{\max(\text{provided},1)} \in [0,1]$
  $f_{\text{click}} = \operatorname{clip}\!\big(\ell(p_{\text{searches}})\cdot r_{\text{click}},\;0,\;1\big)$
  $s_{\text{click}} = \sigma\!\big( z(f_{\text{click}}) \big)$
  Otherwise, set $s_{\text{click}}=0$.

* **Gratitude/affect:** small signal $S_{\text{grateful}}\in[-1,1]$ used as‑is (or lightly scaled).

* **Verbosity multiplier:** per‑user $v\in[0.8,1.2]$ to avoid over‑rewarding wordiness.

### 4.3 Combine signals into one event bump $S_{\text{reg}}$

* **External block:**
  $\text{External} = w_{12}\,s_{\text{dur}} + w_{13}\,s_{\text{freq}} + w_{14}\,s_{\text{rel}}$

* **Internal block (cap click so it cannot dominate):**
  $\text{InternalBase} = w_{15}\,s_{\text{plen}} + w_{16}\,\min\!\Big( s_{\text{click}},\; 0.2\cdot ( s_{\text{plen}} + |S_{\text{grateful}}| + \varepsilon ) \Big) + w_{17}\,S_{\text{grateful}}$
  $\text{Internal} = v\cdot \text{InternalBase}$

* **Final event bump:**
  $S_{\text{reg}} = w_{\text{ext}}\cdot \text{External} + w_{\text{int}}\cdot \text{Internal}, \qquad w_{\text{ext}} + w_{\text{int}} = 1$

**Good defaults:** $w_{\text{ext}}\approx 0.3$, $w_{\text{int}}\approx 0.7$. Keep $h_{\text{rel}} > h_{\text{nurt}}$.

---

## 5) Putting It Together — Updates & Display

### 5.1 Event‑driven constant‑time update (what you implement)

Let event time be $t_k$. With time unit $U =$ days, use fractional days for sub‑day gaps.

* **Decay step:**
  $\Delta_k = \frac{t_k - \text{last\_updated}_t}{U} \qquad (\text{e.g., 1 hour} = 1/24\;\text{day})$
  $S_t \leftarrow S_t\cdot e^{-\lambda\,\Delta_k}, \qquad \lambda = \frac{\ln 2}{h_{\text{nurt}}}$

* **Add the event bump:**
  $S_t \leftarrow S_t + \eta\,S_{\text{reg}}$

* **Update timestamp:**
  $\text{last\_updated}_t \leftarrow t_k$

* **First event init:** if $\text{last\_updated}_t$ is unset, take $S_t\gets 0$ and set $\text{last\_updated}_t\gets t_k$ before applying the update.

* **Lifetime memory $R_t$** uses its own decay $\lambda_{\text{rel}}$ with the same $\Delta_k$ notion.

### 5.2 Read‑time decay (no event)

When merely reading the score at time $T$:

$\Delta_{\text{now}} = \frac{T - \text{last\_updated}_t}{U}, \qquad S_{t,\text{now}} = S_t\cdot e^{-\lambda\,\Delta_{\text{now}}}$

### 5.3 Cross‑topic normalization and UI score

Compute a z‑score across topics using either standard or robust stats:

* **If many topics ($N\ge 10$)**:
  $z_t = \frac{S_{t,\text{now}} - \mu}{\sigma + \varepsilon}$

* **If few topics** (use robust median/MAD):
  $z_t = \frac{S_{t,\text{now}} - \text{median}}{1.4826\cdot \text{MAD} + \varepsilon}$

Map to $[0,1]$ for display with a slope parameter $k\in[1,2]$:

$\text{Snurt}[t] = \sigma\!\big( k\,z_t \big) = \frac{1}{1 + e^{-k z_t}}$

---

## 6) Continuous‑Time (Calculus) View (optional but canonical)

Represent events as impulses to justify the constant‑time update:

$\frac{dS_t}{dt} = -\lambda S_t + \eta\, r_t(t), \qquad r_t(t) = \sum_k S_{\text{reg},k}\,\delta(t - t_k)$

Closed‑form solution for $t\ge t_0$:

$S_t(t) = e^{-\lambda (t - t_0)} S_t(t_0) + \eta \sum_{t_k \le t} S_{\text{reg},k}\, e^{-\lambda (t - t_k)}$

This is equivalent to the event‑driven rule: you only need the current $S_t$ and last timestamp; past events are already exponentially down‑weighted.

---

## 7) Recommended Defaults

* **Time unit:** days (use fractions for hours/minutes).
* **Snurt half‑life:** $h_{\text{nurt}} \in [14,30]$ days.
* **Lifetime $R$ half‑life:** $h_{\text{rel}} \in [30,90]$ days.
* **Dwell cap:** $\tau_{\text{cap}} \in [120,180]$ s.
* **Tiny dwell gate:** $T_{\min} \in [5,10]$ s $\Rightarrow$ ignore below this.
* **External/Internal weights:** $w_{\text{ext}}\approx 0.3$, $w_{\text{int}}\approx 0.7$.
* **Click cap:** contribution $\le 20\%$ of Internal.
* **UI slope:** $k \in [1,2]$.
* **Learning rate:** small $\eta$ so single events don’t spike the score.

---

## 8) Minimal Pseudocode (Reference)

```python
# State per topic t
S_t = 0.0               # raw Snurt memory
last_updated = None     # timestamp
R_t = 0.0               # lifetime relevance memory

# constants
day = 86400.0
lambda_ = math.log(2) / h_nurt_days
lambda_rel = math.log(2) / h_rel_days

# on event at time t_k (unix seconds)
delta_days = 0.0 if last_updated is None else (t_k - last_updated) / day
S_t *= math.exp(-lambda_ * delta_days)

# update lifetime memory R_t using same delta notion
R_t *= math.exp(-lambda_rel * delta_days)
R_t += r_k  # your small per-event increment for lifetime

# compute S_reg from cleaned signals (see formulas)
S_t += eta * S_reg
last_updated = t_k

# read-time scoring at T
Delta_now = 0.0 if last_updated is None else (T - last_updated) / day
S_now = S_t * math.exp(-lambda_ * Delta_now)
# then z-score across topics -> sigmoid for UI
```

---

**That’s the full Snurt spec in Markdown with \$\$ math.** Drop this file into your repo (e.g., `docs/Snurt.md`) and it should render on GitHub with math enabled. If your local preview doesn’t show formulas, view it on GitHub or enable a Markdown math plugin.
