## Read Write Memory (VM RAM)

In contrast to our standard procedures for offline memory checking, the registers and RAM within this context are considered *writable* memory. This distinction introduces additional verification requirements:

- The multiset equality typically expressed as $I \cdot W = R \cdot F$ is not adequate for ensuring the accuracy of read values. It is essential to also verify that each read operation retrieves a value that was written in a previous step.

- To formalize this, we assert that the timestamp of each read operation, denoted as $\text{read\_timestamp}$, must not exceed the global timestamp at that particular step. The global timestamp is a monotonically increasing sequence starting from 0 and ending at $\text{TRACE\_LENGTH}$.

- The verification of $\text{read\_timestamp} \leq \text{global\_timestamp}$ is equivalent to confirming that $\text{read\_timestamp}$ falls within the range $[0, \text{TRACE\_LENGTH}]$ and that the difference $(\text{global\_timestamp} - \text{read\_timestamp})$ is also within the same range.

- The process of ensuring that both $\text{read\_timestamp}$ and $(\text{global\_timestamp} - \text{read\_timestamp})$ lie within the specified range is known as range-checking. This is the procedure implemented in `timestamp_range_check.rs`.
