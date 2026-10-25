/* Include the production body in a translation unit with subsequent code.
 * A leaked .note.GNU-stack section makes the probe non-executable, matching
 * the failure mode of later global_asm blocks in the Rust guest.
 */
.text
#include "../src/word_memory.S"
.globl word_memory_section_probe
.type word_memory_section_probe,@function
word_memory_section_probe:
    li a0, 291
    ret
.size word_memory_section_probe, .-word_memory_section_probe
