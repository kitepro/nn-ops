# nn-ops
 Ops to be compiled using Clang-AVX2

Every Op uses:
<ul>
<li>Multithreading (Some ops have #define THREADS to limit number of threads in some cases, change that to your needs.)</li>
<li>SIMD AVX2 (Do rewrite for AVX512 if you are lucky enough to experience 512bit vectors)</li>
<li>Inline Assembly</li>
</ul><br>
&nbsp&nbsp&nbsp to make things as fast as possible.
