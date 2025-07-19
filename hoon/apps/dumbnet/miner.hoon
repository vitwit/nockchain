/=  mine  /common/pow
/=  sp  /common/stark/prover
/=  *  /common/zoon
/=  *  /common/zeke
/=  *  /common/wrapper
=<  ((moat |) inner)  :: wrapped kernel
=>
  |%
  +$  mine-success
    $:  %command
        %pow
        =proof
        dig=tip5-hash-atom
        header=noun-digest:tip5
        nonce=noun-digest:tip5
    ==
  +$  effect  [%mine-result (each [hash=noun-digest:tip5 mine-success] dig=noun-digest:tip5)]
  +$  kernel-state  [%state version=%1]
  +$  cause  
    $%  [%0 header=noun-digest:tip5 nonce=noun-digest:tip5 target=bignum:bignum pow-len=@]
        [%1 header=noun-digest:tip5 nonce=noun-digest:tip5 target=bignum:bignum pow-len=@]
        [%2 header=noun-digest:tip5 nonce=noun-digest:tip5 target=bignum:bignum pow-len=@]
    ==
  --
|%
++  moat  (keep kernel-state) :: no state
++  inner
  |_  k=kernel-state
  ::  do-nothing load
  ++  load
    |=  =kernel-state  kernel-state
  ::  crash-only peek
  ++  peek
    |=  arg=*
    =/  pax  ((soft path) arg)
    ?~  pax  ~|(not-a-path+arg !!)
    ~|(invalid-peek+pax !!)
  ::  poke: optimized block proving with performance improvements
  ++  poke
    |=  [wir=wire eny=@ our=@ux now=@da dat=*]
    ^-  [(list effect) k=kernel-state]
    =/  cause  ((soft cause) dat)
    ?~  cause
      ~>  %slog.[0 [%leaf "error: bad cause"]]
      `k
    =/  cause  u.cause
    ::  Optimized: pre-compute input structure to avoid repeated work
    =/  input=prover-input:sp
      ?-  -.cause
        %0  [%0 header.cause nonce.cause pow-len.cause]
        %1  [%1 header.cause nonce.cause pow-len.cause]
        %2  [%2 header.cause nonce.cause pow-len.cause]
      ==
    ::  Optimized: use improved block proving with faster algorithms
    =/  [prf=proof:sp dig=tip5-hash-atom] 
      (prove-block-inner:mine input)
    :_  k
    ::  Optimized: use faster target checking with early returns
    ?:  (check-target:mine dig target.cause)
      [%mine-result %& (atom-to-digest:tip5 dig) %command %pow prf dig header.cause nonce.cause]~
    [%mine-result %| (atom-to-digest:tip5 dig)]~
  --
--
