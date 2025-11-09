# Torus-based compression

We implement a torus-based compression method to compress an output of the pairing computation for the BN254 curve (viewed as an element in a degree 12 extension $\mathbb{F}_{q^{12}}$ over the base prime field $\mathbb{F}_q$) to two elements in a degree 2 sub-extension over the same prime field, thus achieving a threefold compression ratio with no information loss. In other words, the decompressed value recovers exactly the pairing value computed without compression.

Recall that the pairing computation follows two steps - the Miller loop and the final exponentiation. The compression method requires only making changes to the final exponentiation step. The compression overhead turns out to be insignificant for applications in Jolt.

# Methodology 
The pairing output has the form $f^{\frac{q^{12}-1}{r}}$, where $f \in \mathbb{F}_{q^{12}}$ is the output from the Miller loop, and $r$ is an integer such that the pairing inputs are $r$-torsion points on the BN254 curve defined over some finite extension of $\mathbb{F}_q$ - in other words, the $r^{\text{th}}$ power vanishes. We can write

$$
f^{\frac{q^{12}-1}{r}} = \Psi_6(q^2)\frac{\Phi_6(q^2)}{r}, 
$$ 
where 

$$
\Phi_6(x) = x^2 - x + 1
$$
is the $6^{\text{th}}$ cyclotomic polynomial and 

$$
\Psi_6(x) = \frac{x^6 - 1}{\Phi_6(x)} = (x^3 + 1)(x + 1).
$$

(In practice, a common optimization is that instead of exponentiating by $\Psi_6(x)$, one raises to a multiple of $\Psi_6(x)$ and an integer coprime to the order of the multiplicative group of the target field. This of course is a 1-to-1 map of the underlying field and has the advantage that one may use linear-algebraic techniques to compute exponentiation by the multiple more efficiently (e.g. see how this is applied to the BN254 curve in [Faster Hashing to $\mathbb{G}_2$](https://cacr.uwaterloo.ca/techreports/2011/cacr2011-26.pdf)). This optimization does not affect the compression method, so we consider only the case of raising to a $\Psi_6(x)$-power for the rest of the discussion.)

Let $\xi \in \mathbb{F}_{q^2}$ be a sextic non-residue and identify 
$$\mathbb{F}_{q^6} = \mathbb{F}_{q^2}(\xi^{\frac{1}{3}}) = \mathbb{F}_{q^2}(\tau)$$
and 
$$\mathbb{F}_{q^{12}} = \mathbb{F}_{q^6}(\xi^{\frac{1}{2}}) = \mathbb{F}_{q^6}(\sigma),$$
where $\tau = \xi^{\frac{1}{3}}$ and $\sigma = \xi^{\frac{1}{2}}$. Through this notation, we emphasise that the sets $\{1, \tau\}$ and $\{1, \sigma\}$ form $\mathbb{F}_{q^2}$-linear and $\mathbb{F}_{q^6}$-linear bases of the fields $\mathbb{F}_{q^6}$ and $\mathbb{F}_{q^{12}}$ viewed as vector spaces, respectively. 

It turns out that for each element $f \in \mathbb{F}_{q^{12}}$, the power $f^{\Psi_6(q^2)}$ can be written as 

$$
f^{\Psi_6(q^2)} = \frac{b + \sigma}{b - \sigma},
$$
where $b = c_0 + c_1\tau + c_2\tau^2 \in \mathbb{F}_6$, $c_i \in \mathbb{F}_2$, and we can recover $c_2$ from $c_0$ and $c_1$ alone. 

Hence we can represent $f^{\Psi_6(q^2)}$ using the pair $(c_0, c_1)$ achieving a compression ratio of three, where the compression takes two steps in which we compress to the field $\mathbb{F}_{q^6}$ and $\mathbb{F}_{q^2}$, respectively.

## Compression to $\mathbb{F}_{q^6}$ 

We can compute $f^{\Psi_6(q^2)}$ as 

$$
f^{\Psi_6(q^2)} = f^{(q^6 - 1)(q^2 + 1)} = (f^{q^6 - 1})^{q^2 + 1}.
$$

Write $f = a_0 + a_1\sigma$, where $a_i \in \mathbb{F}_{q^6}$, we have 

$$
f^{q^6 - 1} = \frac{(a_0 + a_1\sigma)^{q^6}}{a_0 + a_1\sigma} = \frac{(a_0 - a_1\sigma)}{a_0 + a_1\sigma} = \frac{\tilde{a} - \sigma}{\tilde{a} + \sigma}, 
$$

where $\tilde{a} = \frac{a_0}{a_1}$ and the second equality follows since the $q^2$-power map generates the Galois group of the quadratic extension $\mathbb{F}_{q^{2}}(\sigma)/\mathbb{F}_{q^2}$ inside $\mathbb{F}_{q^{12}}$, so in particular $\sigma^{q^6} = -\sigma$. Hence

$$
(f^{q^6 - 1})^{q^2 + 1} = \frac{\tilde{a} - \sigma}{\tilde{a} + \sigma}(\frac{\tilde{a} - \sigma}{\tilde{a} + \sigma})^{q^2} = \frac{\tilde{a} - \sigma}{\tilde{a}^{q^2} + \sigma}\cdot\frac{\tilde{a} + \sigma}{\tilde{a}^{q^2} - \sigma} = \frac{\tilde{a} - \sigma}{\tilde{a} + \sigma}\cdot\frac{-\tilde{a}^{q^2} - \sigma}{-\tilde{a}^{q^2} + \sigma},
$$
which simplifies to 
$$
(f^{q^6 - 1})^{q^2 + 1} = \frac{\tilde{\beta} - \sigma}{\tilde{\beta} + \sigma},
$$
where 
$$
\tilde{\beta} = \frac{-\tilde{a}^{q+1} + \xi}{-\tilde{a}^q + \tilde{a}} \in \mathbb{F}_{q^6}.
$$


## Compression to two elements in $\mathbb{F}_{q^2}$

We can write $\tilde{\beta} = c_0 + c_1\tau + c_2\tau^2$, where recall $\tau = \xi^{\frac{1}{3}}$, then we have

$$
c_2 = \frac{3c_0^2 + \xi}{3c_1\xi},
$$
so we can drop $c_2$ to only use $c_0$ and $c_1$ to represent $\tilde{\beta}$.

## Compression and decompression
For compressing a pairing value $a^{\frac{q^{12} - 1}{r}}$, first compute $f = a^{\Phi_6(q^2)}$, then compress $f^{\Psi_6(q^2)}$ to two $\mathbb{F}_{q^2}$ elements as in the previous section. 

For decompression, first compute $\tilde{\beta} \in \mathbb{F}_{q^6}$ from two coefficients $c_0$ and $c_1$ in $\mathbb{F}_{q^2}$, where $\tilde{\beta} = c_0 + c_1\tau + c_2\tau^2$ as in the previous section. Then, compute  
$$
a^{\frac{q^{12} - 1}{r}} = \frac{\tilde{\beta} - \sigma}{\tilde{\beta} + \sigma}
$$to recover the original pairing value.

# Implementation Detail

## Basis choice
Different choices of basis vectors for an extension field affect the complexity of field operations within the extension field. In Arkworks, for field arithmetic optimization, the field $\mathbb{F}_{q^{12}}$ is represented as an extension $\mathbb{F}_{q^2}(\tau, \sigma^{\frac{1}{3}})$, where $\{1, \sigma\}$ is an $\mathbb{F}_{q^6}$-basis over the sub-extension $\mathbb{F}_{q^2}(\tau) = \mathbb{F}_{q^6}$. (Recall that $\tau = \xi^{\frac{1}{3}}$ and $\sigma = \xi^{\frac{1}{2}}$, where $\xi \in \mathbb{F}_{q^2}$ is a sextic non-residue.)

The $q^2$-power map no longer maps $\sigma^{\frac{1}{3}}$ to $-\sigma^{\frac{1}{3}}$, so naively applying the formula in the previous section on the generator $\sigma^{\frac{1}{3}}$ will cause problems. Fortunately, we can fix this easily by doing a change of basis before compression and after decompression. 

Using the identity $\tau = \sigma^{\frac{2}{3}}$, we have

$$
a + b\sigma = a + (b\sigma^{\frac{2}{3}})\sigma^{\frac{1}{3}} = a + (b\tau)\sigma^{\frac{1}{3}}
$$

and hence 

$$
a + b\sigma^{\frac{1}{3}} = a + (b\tau^{-1})\sigma,
$$

where $a$, $b$, and $\tau$ all are elements of $\mathbb{F}_{q^6} = \mathbb{F}_{q^2}(\tau)$. To convert between elements in $\mathbb{F}_{q^{12}}$ written in $\mathbb{F}_{q^6}$-bases $\{1, \sigma\}$ and $\{1, \sigma^{\frac{1}{3}}\}$, it suffices to do a multiplication or division by $\tau$, and we shall see this can be implemented entirely using arithmetics in $\mathbb{F}_{q^2}$.

Indeed, write $b = c_0 + c_1\tau + c_2\tau^2$, where $c_i \in \mathbb{F}_{q^2}$, we have

$$
b\tau = c_2\xi + c_0\tau + c_1\tau^2,
$$

and

$$
b\tau^{-1} = c_1 + c_2\tau + c_0\xi^{-1}\tau^2.
$$
The conversion formulae are provided in the following code snippet.

```rust
#[inline]
pub fn fq12_to_compressible_fq12(value: Fq12) -> CompressibleFq12 {
    // Divide by the generator of Fq6
    let new_c1 = Fq6 {
        c0: value.c1.c1,
        c1: value.c1.c2,
        c2: value.c1.c0 * Fq6Config::NONRESIDUE.inverse().unwrap(),
    };

    CompressibleFq12 {
        c0: value.c0,
        c1: new_c1,
    }
}

#[inline]
pub fn compressible_fq12_to_fq12(value: CompressibleFq12) -> Fq12 {
    // Multiply by the generator of Fq6
    let new_c1 = Fq6 {
        c0: value.c1.c2 * Fq6Config::NONRESIDUE,
        c1: value.c1.c0,
        c2: value.c1.c1,
    };

    Fq12 {
        c0: value.c0,
        c1: new_c1,
    }
}
```

## Switching order of exponentiation

In the case of compression, one needs to first raise to a power of $\Phi_6(q^2)$ before then raising to a power of $\Psi_6(q^2)$ (recall that the compression method amounts to using two elements in $\mathbb{F}_{q^2}$ to represent a $\Psi_6(q^2)$-power). However, the final exponentiation is significantly faster if we switch the order and first raise to the power of $\Psi_6(q^2)$ instead, as in the uncompressed case. This is because elements of $\Psi_6(q^2)$-powers have the special properties that they are in a cyclotomic subgroup, so optimized squaring and inversion are much cheaper. Fortunately, the overhead in practice is insignificant for multi-pairing computation, which is the use case for Dory, because the overhead in the final exponentiation step is amortized across the multi-Miller loop computations.  
