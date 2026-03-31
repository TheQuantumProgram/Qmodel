# IQFT Small Compare Summary

- Family: `IQFTCompare`
- Models: `n = 10..20`, all with `window_size = 5`
- Purpose: compare abstract model-checking cost against full-execution cost on instances where exact concrete execution is still feasible

## Headline

- All `11/11` models were `satisfied` in both abstract and concrete execution.
- Smallest abstract/full concrete time ratio: `n=20`.
- Largest abstract/full concrete time ratio: `n=10`.
- Space ratios are reported against both the actual full-execution statevector footprint and the theoretical full density-matrix footprint.

## Table

| n | abstract s | full concrete s | time ratio a/full | abs bytes | full sv bytes | abs/full sv | full dm bytes | abs/full dm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.057861 | 0.018102 | 3.196348 | 65792 | 16384 | 4.015625 | 16777216 | 0.003921508789 |
| 11 | 0.060620 | 0.025558 | 2.371884 | 65856 | 32768 | 2.009766 | 67108864 | 0.000981330872 |
| 12 | 0.078516 | 0.051997 | 1.510002 | 65920 | 65536 | 1.005859 | 268435456 | 0.000245571136 |
| 13 | 0.085081 | 0.111353 | 0.764063 | 65984 | 131072 | 0.503418 | 1073741824 | 0.000061452389 |
| 14 | 0.088052 | 0.229334 | 0.383947 | 66048 | 262144 | 0.251953 | 4294967296 | 0.000015377998 |
| 15 | 0.133178 | 0.519211 | 0.256500 | 66112 | 524288 | 0.126099 | 17179869184 | 0.000003848225 |
| 16 | 0.143482 | 1.102322 | 0.130163 | 66176 | 1048576 | 0.063110 | 68719476736 | 0.000000962988 |
| 17 | 0.126762 | 2.244004 | 0.056489 | 66240 | 2097152 | 0.031586 | 274877906944 | 0.000000240980 |
| 18 | 0.160540 | 5.980853 | 0.026842 | 66304 | 4194304 | 0.015808 | 1099511627776 | 0.000000060303 |
| 19 | 0.164599 | 11.270417 | 0.014605 | 66368 | 8388608 | 0.007912 | 4398046511104 | 0.000000015090 |
| 20 | 0.171936 | 22.427221 | 0.007666 | 66432 | 16777216 | 0.003960 | 17592186044416 | 0.000000003776 |
