/=  sp  /common/stark/prover
/=  np  /common/nock-prover
/=  *  /common/zeke
|%
++  check-target
  |=  [proof-hash-atom=tip5-hash-atom target-bn=bignum:bignum]
  ^-  ?
  =/  target-atom=@  (merge:bignum target-bn)
  ?>  (lte proof-hash-atom max-tip5-atom:tip5)
  ::  Optimized: early return for common cases
  ?:  =(target-atom max-tip5-atom:tip5)  %.y
  (lte proof-hash-atom target-atom)
::
::  +check-target-fast: optimized target checking with branch prediction hints
::
++  check-target-fast
  ~/  %check-target-fast
  |=  [proof-hash-atom=tip5-hash-atom target-bn=bignum:bignum]
  ^-  ?
  =/  target-atom=@  (merge:bignum target-bn)
  ?>  (lte proof-hash-atom max-tip5-atom:tip5)
  ::  Optimized: handle common cases first for better branch prediction
  ?:  =(target-atom max-tip5-atom:tip5)  %.y
  ?:  =(proof-hash-atom 0)  %.y
  (lte proof-hash-atom target-atom)
::
++  prove-block  (cury prove-block-inner pow-len)
::
::  +prove-block-inner
++  prove-block-inner
  |=  prover-input:sp
  ^-  [proof:sp tip5-hash-atom]
  =/  =prove-result:sp
    ?-  version
      %0  (prove:np version header nonce pow-len)
      %1  (prove:np version header nonce pow-len)
      %2  (prove:np version header nonce pow-len)
    ==
  ?>  ?=(%& -.prove-result)
  =/  =proof:sp  p.prove-result
  =/  proof-hash=tip5-hash-atom  (proof-to-pow proof)
  [proof proof-hash]
--
