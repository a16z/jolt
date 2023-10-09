## Primary sumcheck from Lasso:
$$
\sum_{x \in \{0,1\}^{log(m)}}{\widetilde{eq}(r,x) \cdot g(E_1(x), ... E_\alpha(x))}
$$

## Jolt primary sumcheck:
- $F$: number of instructions ($f$ for flags)
$$\sum_{x \in \{0,1\}^{log(m)}} \sum_{f \in \{0,1\}^{log(F)}} {\widetilde{eq}(r,x) \cdot \widetilde{flags}(x,f) \cdot g_f(\text{terms}_f)}$$
Where:
- $g_f$: the relevant collation function of instruction $f$
- $terms_f$: relevant memory evaluation polynomials: $E_1(x), ... E_\alpha(x)$ – Some of these terms will be shared across multiple $g_f$ 
- $\widetilde{flags}$: $log(m) + log(F)$ variate or of length $m*F$ 


## Expand Flags
Assume $F=2$:
$$\sum_{x \in \{0,1\}^{log(m)}} {\widetilde{eq}(r,x) \cdot [\widetilde{flags}(x,0) \cdot g_0(\text{terms}_0) + \widetilde{flags}(x,1) \cdot g_1(\text{terms}_1)]}$$

Sumcheck runs for $log(m)$ rounds. After each round of sumcheck, bind a single variable in:
- $\widetilde{eq}(r,x)$ 
- $\widetilde{flags}(x,0),...\widetilde{flags}(x,F)$ 
- All $E$ polynomials in the set $\text{terms}_0\cup ... \cup \text{terms}_F$        

Verifier must evaluate:
	- $\widetilde{eq}(r,r')$
	- $\widetilde{flags}(r',0),\widetilde{flags}(r',1), ..., \widetilde{flags}(r',F)$ 


Batching
- Evaluate single polynomial at $F$ points, pass degree $F-1$ polynomial through $F$ points.
- Highly structured: Same $r'$ first bits. Differ in final $log(F)$ bits.
- Thaler Book 4.5.2 – Subroutine reducing multiple polynomial evaluations to 1.
- Prover sends claimed values for $\widetilde{flags}(r',0), ..., \widetilde{flags}(r',F)$ and an opening proof as described in 4.5.2, and can check with a single opening proof that these are valid.
- Flags should be a single concatenated MLE, spaced to powers of 2, filled with 0s in the gaps.



## Memory checking
Surge steps 4-7 check that $E_i$ is well formed via memory checking techniques.

We're trying to prove the correct computation of:
$$\sum_{i \in \{0,1\}^{log(m)}}{\tilde{eq}(i, r) \cdot T[nz[i]]}$$
Where, if T is SOS: $T[r] = g(T_1[r_1], ... T_\alpha[r_c])$ 
The verifier can't compute $T_i[r_j]$ directly, so instead we commit to $\alpha$ polynomials $E_1, ... E_\alpha$ purported to specify values of $m$ reads into $T_i$ 

The verifier then passes $E_1(r), ... E_\alpha(r)$ along with opening proofs to the verifier. This allows the verifier to check the primary sumcheck.

We then check that $E_i$ was "well-formed". Meaning $E_i(j) = T_i(dim_i(j)) \forall \{0,1\}^{log(m)}$ via memory checking.

Once per $T_i$ we must run memory checking. For each $T_i$ we'll compute a multi-set hash of 4 different sets:
- init
- read
- write
- final
If the prover is honest `H(init) * H(write) = H(read) * H(final)`.
The multi-set hash is of the following form:
$$H(a,v,t) = \prod_{i=0}{h_{\gamma, \tau}(a[i],v[i],t[i])}$$
Where $h_{\gamma, \tau} = t(i) \cdot \gamma^2 + v(i) \cdot \gamma + a(i) - \tau$. *Note the linear combination differs slightly from the papers.*

Each of the entries `(a,v,t)` correspond to:
- `a`: the address of the tuple
- `v`: the value stored in that address
- `t`: the count of accesses to that address at the time of operation

We'll compute these multi-set hashes via a sumcheck-based Grand Product Argument (an optimized GKR protocol), which will reduce the multi-set hashes from products over `MEMORY_SIZE` or `NUM_OPERATIONS` terms to a single point $H(a(r), v(r), t(r))$. The verifier can then check against an evaluations of commitments to the underlying terms $h(a(r), v(r), t(r))$ 

Naively, we could commit to $a_{init}, a_{read}, a_{write}, a_{final}, v_{init}, v_{read}, v_{write}, v_{final}, t_{init}, t_{read}, t_{write}, t_{final}$. 

But depending on the context in which we're running memory checking, there would be a lot of wasted commitments and opening proofs. Beyond reuse, some of these items can be computed directly, or derived from other commitments

## Surge memory checking
- Address
    - $a_{read} = a_{write} = dim_i$
    - $a_{init} = a_{final} = [0...\text{MEMORY_SIZE}]$
- Value
    - $v_{read} = v_{write} = E$
    - $v_{init} = v_{final} = \tilde{T}_i$ 
- Counter
    - $t_{write} = t_{read} + 1$ 
    - $t_{init} = 0$
    - $t_{final}$ 

Required commitments:
- $dim_i$
- $E$
- $t_{read}$
- $t_{final}$

## General read only memory checking
$C=1$ `NUM_MEMORIES=1`
`TODO`

## Flags
`TODO: Goal is to cancel out unused (a,v,t) entries in read / write.`

## Read-write
`TODO`