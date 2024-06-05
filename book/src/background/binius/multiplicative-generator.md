# Multiplicative Generator
# Finding a Multiplicative Generator for $\text{GF}(2^k)$
To find a generator of the multiplicative group $\text{GF}(2^k)^*$, which includes all nonzero elements of the field $\text{GF}(2^k)$, a probabilistic method is used. This method iteratively tests random elements to determine if they are generators of the group. The process is summarized below.


## Probabilistic Method Algo
```python
while true:
    # sample random nonzero element x
    for each maximal proper divisor d of |𝔽*|:
        if xᵈ == 1: continue
    return x
```
This algorithm ensures that the element $x$ does not reside in any proper subgroup of $\text{GF}(2^k)^*$. The condition checked within the loop, $x^d \neq 1$, ensures that $x$ has the maximal possible order, which is necessary for it to be a generator of the group.


## Background:
- **Group Structure**: $\text{GF}(2^k)^*$ is the multiplicative group of the field $\text{GF}(2^k)$, containing all field elements except 0. Its order is $2^k - 1$, which is prime to 2.
- **Maximal Proper Divisors**: These divisors are crucial as testing powers for these values ensures that $x$ is not a member of any smaller cyclic subgroup. The divisors of $2^k - 1$ are used to verify the generator property.
- **Probabilistic Success Rate**: The likelihood of a random element being a generator is significant, given the structure of the group. This can be supported by the [[chinese-remainder-theorem]], which suggests that the intersection of several smaller groups (subgroups corresponding to divisors) is likely non-trivial only when all subgroup conditions are simultaneously satisfied.
## Example Divisors:
- For $\text{GF}(2^{16})^*$, $|\mathbb{F}^*| = 2^{16} - 1 = 3 \times 5 \times 17 \times 257$
- For $\text{GF}(2^{32})^*$, $|\mathbb{F}^*| = 2^{32} - 1 = 3 \times 5 \times 17 \times 257 \times 65537$
- For $\text{GF}(2^{64})^*$, $|\mathbb{F}^*| = 2^{64} - 1 = 3 \times 5 \times 17 \times 257 \times 65537 \times 6700417$


# Maximal Proper Divisors
A maximal proper divisor of a number $n$ is a divisor $d$ of $n$ which is neither $1$ nor $n$ itself, and there are no other divisors of $n$ that divide $d$ except $1$ and $d$. Essentially, $d$ is a divisor that is not a multiple of any smaller divisor other than $1$ and itself, making it 'maximal' under the set of proper divisors.
## Algorithm for Finding
The algorithm to find the maximal proper divisors of a given number $n$ involves identifying all divisors of $n$ and then selecting those which do not have other divisors besides $1$ and themselves. The steps are as follows:
1. **Find All Divisors**: First, list all divisors of $n$ by checking for every integer $i$ from $1$ to $\sqrt{n}$ if $i$ divides $n$. If $i$ divides $n$, then both $i$ and $n/i$ are divisors.
2. **Filter Maximal Divisors**: From this list of divisors, exclude $1$ and $n$. For each remaining divisor, check if it can be expressed as a multiple of any other divisor from the list (other than $1$ and itself). If it cannot, then it is a maximal proper divisor.

```python
function findMaximalProperDivisors(n):
    divisors = []
    for i from 1 to sqrt(n):
        if n % i == 0:
            divisors.append(i)
            if i != n / i:
                divisors.append(n / i)
    divisors.remove(1)
    divisors.remove(n)

    maximal_proper_divisors = []
    for d in divisors:
        is_maximal = true
        for e in divisors:
            if d != e and e != 1 and d % e == 0:
                is_maximal = false
                break
        if is_maximal:
            maximal_proper_divisors.append(d)

    return maximal_proper_divisors