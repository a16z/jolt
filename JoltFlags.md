## Primary sumcheck from Lasso:
$$
\sum_{x \in \{0,1\}^{log(m)}}{\widetilde{eq}(r,x) \cdot g(E_1(x), ... E_\alpha(x))}
$$

## Jolt primary sumcheck:
- $F$: number of instructions ($f$ for flags)
$$
\sum_{x \in \{0,1\}^{log(m)}} \sum_{f \in \{0,1\}^{log(F)}} {\widetilde{eq}(r,x) \cdot \widetilde{flags}(x,f) \cdot g_f(\text{terms}_f)}
$$
Where:
- $g_f$: the relevant collation function of instruction $f$
- $\text{terms}_f$: relevant memory evaluation polynomials: $E_1(x), ... E_\alpha(x)$ – Some of these terms will be shared across multiple $g_f$ 
- $\widetilde{flags}$: $log(m) + log(F)$ variate or of length $m*F$ 


## Expand Flags
Assume $F=2$:
$$
\sum_{x \in \{0,1\}^{log(m)}} {\widetilde{eq}(r,x) \cdot [\widetilde{flags}(x,0) \cdot g_0(\text{terms}_0) + \widetilde{flags}(x,1) \cdot g_1(\text{terms}_1)]}
$$

Sumcheck runs for $log(m)$ rounds. After each round of sumcheck, bind a single variable in:
- $\widetilde{eq}(r,x)$ 
- $\widetilde{flags}(x,0),...\widetilde{flags}(x,F)$ 
- All $E$ polynomials in the set $\text{terms}_0,\cup \,... \,\cup \,\,\text{terms}_F$        

Verifier must evaluate:
	- $\widetilde{eq}(r,r')$
	- $\widetilde{flags}(r',0),\widetilde{flags}(r',1), ..., \widetilde{flags}(r',F)$ 


Batching
- Evaluate single polynomial at $F$ points, pass degree $F-1$ polynomial through $F$ points.
- Highly structured: Same $r'$ first bits. Differ in final $log(F)$ bits.
- Thaler Book 4.5.2 – Subroutine reducing multiple polynomial evaluations to 1.
- Prover sends claimed values for $\widetilde{flags}(r',0), ..., \widetilde{flags}(r',F)$ and an opening proof as described in 4.5.2, and can check with a single opening proof that these are valid.
- Flags should be a single concatenated MLE, spaced to powers of 2, filled with 0s in the gaps.
- TODO: Check max size of flags commitment fits in RAM.