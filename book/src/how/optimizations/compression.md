# Torus-based compression

We implement a torus-based compression method to compress an output of the pairing computation for the BN254 curve (viewed as an element in a degree 12 extension $\mathbb{F}_{q^{12}}$ over the base prime field $\mathbb{F}_q$) to two elements in a degree 2 field extension over the same prime field, thus achieving a threefold compression ratio with no information loss. The compression only affects the final exponentiation step of the pairing computation.  

# Methodology 
The pairing output has the form $f^{\frac{q^{12}-1}{r}}$, where $f \in \mathbb{F}_{q^{12}}$ is the output from the Miller loop, and $r$ is an integer with the property that the $r$-th power of pairing inputs vanish. We can write

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

Let $\xi \in \mathbb{F}_{q^2}$ be a sextic non-residue and identify 
$$\mathbb{F}_{q^6} = \mathbb{F}_{q^2}(\xi^{\frac{1}{3}}) = \mathbb{F}_{q^2}(\tau)$$
and 
$$\mathbb{F}_{q^{12}} = \mathbb{F}_{q^6}(\xi^{\frac{1}{2}}) = \mathbb{F}_{q^6}(\sigma),$$
where $\tau = \xi^{\frac{1}{3}}$ and $\sigma = \xi^{\frac{1}{2}}$.

It turns out that for each element $f \in \mathbb{F}_{q^{12}}$, the power $f^{\Psi_6(q^2)}$ can be written as 

$$
f^{\Psi_6(q^2)} = \frac{b + \sigma}{b - \sigma},
$$
where $b = c_0 + c_1\tau + c_2\tau^2 \in \mathbb{F}_6$, where $c_i \in \mathbb{F}_2$, and we can recover $c_2$ from $c_0$ and $c_1$ alone. Hence we can represent $f^{\Psi_6(q^2)}$ using the pair $(c_0, c_1)$ achieving a compression ratio of 3, where the compression takes two steps in which we compress to the field $\mathbb{F}_{q^6}$ and $\mathbb{F}_{q^2}$, respectively.

## Compression to $\mathbb{F}_{q^6}$ 

We can compute $f^{\Psi_6(q^2)}$ as 

$$
f^{\Psi_6(q^2)} = f^{(q^6 - 1)(q^2 + 1)} = (f^{q^6 - 1})^{q^2 + 1}.
$$

Write $f = a_0 + a_1\sigma$, where $a_i \in \mathbb{F}_{q^6}$, we have 

$$
f^{q^6 - 1} = \frac{(a_0 + a_1\sigma)^{q^6}}{a_0 + a_1\sigma} = \frac{(a_0 - a_1\sigma)}{a_0 + a_1\sigma} = \frac{\tilde{a} - \sigma}{\tilde{a} + \sigma}, 
$$

where $\tilde{a} = \frac{a_0}{a_1}$ and the second equality follows since the $q^2$-power map generates the Galois group of the quadratic extension $\mathbb{F}_{q^{2}}(\sigma)/\mathbb{F}_{q^2}$ inside $\mathbb{F}_{q^{12}}$, so in particular $\sigma^{q^2} = -\sigma$. Hence

$$
(f^{q^6 - 1})^{q^2 + 1} = \frac{\tilde{a} - \sigma}{\tilde{a} + \sigma}(\frac{\tilde{a} - \sigma}{\tilde{a} + \sigma})^{q^2} = \frac{\tilde{a} - \sigma}{\tilde{a}^{q^2} + \sigma}(\frac{\tilde{a}^{q^2} + \sigma}{\tilde{a} - \sigma}) = \frac{\tilde{a} - \sigma}{\tilde{a}^{q^2} + \sigma}\frac{-\tilde{a}^{q^2} - \sigma}{-\tilde{a} + \sigma},
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

## Compression
For compressing a pairing value $a^{\frac{q^{12} - 1}{r}}$, first compute $f = a^{\Phi_6(q^2)}$, then compress $f$ to two $\mathbb{F}_{q^2}$ elements as in the previous section. 


## Decompression

For decompression, first compute $\tilde{\beta}$ as in the previous section on compressing to $\mathbb{F}_{q^2}$, then compute  
$$
f^{\Psi_6(q^2)} = \frac{\tilde{\beta} - \sigma}{\tilde{\beta} + \sigma}.
$$
This recovers the original pairing value.

# Implementation Detail

## Basis choice
In Arkworks, for optimizing field operations, the field $\mathbb{F}_{q^{12}}$ is represented as the field extension $\mathbb{F}_{q^2}(\tau, \sigma^{\frac{1}{3}}) = \mathbb{F}_{q^2}(\sigma^{\frac{1}{3}})$ instead of $\mathbb{F}_{q^2}(\tau, \sigma)$, where recall that $\tau = \xi^{\frac{1}{3}}$ and $\sigma = \xi^{\frac{1}{2}}$, where $\xi \in \mathbb{F}_{q^2}$ is a sextic non-residue. The $q^2$-power map no longer maps $\sigma^{\frac{1}{3}}$ to $-\sigma^{\frac{1}{3}}$, so naively applying the formula in the previous section on the generator $\sigma^{\frac{1}{3}}$ will cause problems. Fortunately, we can fix this easily by doing a change of basis before compression and after decompression. 

Consider 

$$
a + b\sigma = a + (b\sigma^{\frac{2}{3}})\sigma^{\frac{1}{3}} = a + (b\tau)\sigma^{\frac{1}{3}}
$$

and hence 

$$
a + b\sigma^{\frac{1}{3}} = a + (b\tau^{-1})\sigma,
$$

where $a$, $b$, and $\tau$ all are elements of $\mathbb{F}_{q^6} = $\mathbb{F}_{q^2}(\tau)$. To convert between elements in $\mathbb{F}_{q^{12}}$ written in $\mathbb{F}_{q^6}$-bases $(1, \sigma)$ and $(1, \sigma^{\frac{1}{3}})$, it suffices to do a multiplication or a division by $\tau$, and we shall see this can be implemented entirely in $\mathbb{F}_{q^2}$ arithmetic.

Indeed, write $b = c_0 + c_1\tau + c_2\tau^2$, where $c_i \in \mathbb{F}_{q^2}$, we have $

$$
b\tau = c_2\xi + c_0\tau + c_1\tau^2,
$$

and

$$
b\tau^{-1} = c_1 + c_2\tau + c_0\xi^{-1}\tau^2,
$$
which gives the coressponding conversion formulae.


