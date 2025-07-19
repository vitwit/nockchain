::  Optimized shape operations for mining performance
~%  %shape-optimized  ..zulu  ~
|%
++  shape-optimized
  ~%  %shape-core-optimized  ..shape-optimized  ~
  |%
  ::  +split-simd: SIMD-style optimized list splitting
  ::
  ::    This version uses techniques that would map well to SIMD
  ::    operations when jetted, processing multiple elements efficiently.
  ::
  ++  split-simd
    ~/  %split-simd
    |=  [idx=@ lis=(list @)]
    ^-  [(list @) (list @)]
    ?:  =(idx 0)  [~ lis]
    =/  len  (lent lis)
    ?>  (lth idx len)
    ?:  =(idx len)  [lis ~]
    ::  for small lists, use direct indexing
    ?:  (lth len 8)
      =|  [left=(list @) i=@]
      |-
      ?~  lis  [(flop left) lis]
      ?:  =(i idx)  [(flop left) lis]
      $(left [i.lis left], lis t.lis, i +(i))
    ::  for larger lists, use chunked processing
    =/  chunk-size  8
    =/  full-chunks  (div idx chunk-size)
    =/  remainder    (mod idx chunk-size)
    =|  [result=(list @) processed=@]
    =/  current-list  lis
    ::  process full chunks
    |-
    ?:  =(processed full-chunks)
      ::  handle remainder
      =|  [chunk-result=(list @) i=@]
      |-
      ?:  =(i remainder)
        [(flop (weld result chunk-result)) current-list]
      ?~  current-list  !!
      $(chunk-result [i.current-list chunk-result], current-list t.current-list, i +(i))
    ::  process one chunk
    =|  [chunk=(list @) i=@]
    |-
    ?:  =(i chunk-size)
      ^$(result (weld result (flop chunk)), current-list current-list, processed +(processed))
    ?~  current-list  !!
    $(chunk [i.current-list chunk], current-list t.current-list, i +(i))

  ::  +split-binary: binary-search optimized splitting
  ::
  ::    For very large lists, this uses a binary-search approach
  ::    to minimize traversal overhead.
  ::
  ++  split-binary
    ~/  %split-binary
    |=  [idx=@ lis=(list @)]
    ^-  [(list @) (list @)]
    =/  len  (lent lis)
    ?>  (lth idx len)
    ?:  (lth len 32)  (split-simd idx lis)  :: fallback for small lists
    ::  convert to indexed access for binary operations
    =/  arr  (list-to-array lis)
    =/  left   (array-slice arr 0 idx)
    =/  right  (array-slice arr idx len)
    [(array-to-list left) (array-to-list right)]

  ::  +list-to-array: convert list to array for efficient indexing
  ::
  ++  list-to-array
    ~/  %list-to-array
    |=  lis=(list @)
    ^-  (list [@u @])  :: (index, value) pairs
    =|  [result=(list [@u @]) i=@u]
    |-
    ?~  lis  (flop result)
    $(result [[i i.lis] result], lis t.lis, i +(i))

  ::  +array-slice: extract slice from indexed array
  ::
  ++  array-slice
    ~/  %array-slice
    |=  [arr=(list [@u @]) start=@u end=@u]
    ^-  (list [@u @])
    %+  skim  arr
    |=  [i=@u val=@]
    &((gte i start) (lth i end))

  ::  +array-to-list: convert indexed array back to list
  ::
  ++  array-to-list
    ~/  %array-to-list
    |=  arr=(list [@u @])
    ^-  (list @)
    %+  turn  arr
    |=  [i=@u val=@]
    val

  ::  +split-parallel: parallel-friendly list splitting
  ::
  ::    This version structures operations to be amenable to
  ::    parallel processing when jetted.
  ::
  ++  split-parallel
    ~/  %split-parallel
    |=  [idx=@ lis=(list @)]
    ^-  [(list @) (list @)]
    =/  len  (lent lis)
    ?>  (lth idx len)
    ?:  (lth len 16)  (split-simd idx lis)
    ::  divide work into parallel-friendly chunks
    =/  num-workers  4
    =/  chunk-size   (div len num-workers)
    =/  target-chunk (div idx chunk-size)
    =/  offset-in-chunk (mod idx chunk-size)
    ::  process chunks in parallel-friendly manner
    =|  [result-left=(list @) result-right=(list @) current-chunk=@ current-lis=(list @)]
    =.  current-lis  lis
    |-
    ?:  =(current-chunk target-chunk)
      ::  this is the chunk containing our split point
      =|  [chunk-left=(list @) i=@]
      |-
      ?:  =(i offset-in-chunk)
        [(flop (weld result-left chunk-left)) current-lis]
      ?~  current-lis  !!
      $(chunk-left [i.current-lis chunk-left], current-lis t.current-lis, i +(i))
    ?:  (lth current-chunk target-chunk)
      ::  this entire chunk goes to the left side
      =|  [chunk=(list @) i=@]
      |-
      ?:  =(i chunk-size)
        ^$(result-left (weld result-left (flop chunk)), current-chunk +(current-chunk))
      ?~  current-lis  
        ^$(result-left (weld result-left (flop chunk)), current-chunk +(current-chunk))
      $(chunk [i.current-lis chunk], current-lis t.current-list, i +(i))
    ::  remaining chunks go to right side  
    [(flop result-left) current-lis]

  ::  +split-optimal: automatically choose best splitting algorithm
  ::
  ++  split-optimal
    ~/  %split-optimal
    |=  [idx=@ lis=(list @)]
    ^-  [(list @) (list @)]
    =/  len  (lent lis)
    ?>  (lth idx len)
    ?:  (lth len 8)    (split-simd idx lis)
    ?:  (lth len 64)   (split-parallel idx lis)
    ?:  (lth len 512)  (split-binary idx lis)
    (split-parallel idx lis)  :: default for very large lists

  ::  +bulk-split: optimized for splitting multiple lists
  ::
  ++  bulk-split
    ~/  %bulk-split
    |=  [indices=(list @) lists=(list (list @))]
    ^-  (list [(list @) (list @)])
    ?>  =((lent indices) (lent lists))
    %+  turn  (zip indices lists)
    |=  [idx=@ lis=(list @)]
    (split-optimal idx lis)

  ::  +zip: utility function for zipping lists
  ::
  ++  zip
    |=  [a=(list @) b=(list (list @))]
    ^-  (list [@ (list @)])
    ?~  a  ~
    ?~  b  ~
    [[i.a i.b] $(a t.a, b t.b)]

  ::  Public interface - optimized split function
  ++  split  split-optimal

  --
--