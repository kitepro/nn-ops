# nn-ops
 Ops to be compiled using Clang-AVX2

Every Op uses:
<ul>
<li>Multithreading (Some ops have #define THREADS to limit number of threads in some cases, change that to your needs.)</li>
<li>SIMD AVX2 (Do rewrite for AVX512 if you are lucky enough to experience 512bit vectors)</li>
<li>FMA instructions (So dont forget -fma while compiling)</li>
</ul><br>
&nbsp&nbsp&nbsp to make things as fast as possible.
<br><br>
<b>I'm compiling from VS2019 with this .bat file: (%1 is the file name to be compiled)</b><br>

```ruby
@echo Building %1
clang.exe -c %1.cpp -fopenmp -msse -msse2 -mavx -mavx2 -mfma -O3 -o %1.o
clang.exe -shared -v -fopenmp -msse -msse2 -mavx -mavx2 -mfma -O3 -o %1.dll %1.o
@echo DONE
```
