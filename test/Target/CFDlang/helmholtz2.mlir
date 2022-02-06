module  {
  cfdlang.program  {
    cfdlang.input @S : [11 11]
    cfdlang.input @D : [11 11 11]
    cfdlang.input @u : [11 11 11]
    cfdlang.define @t0 : [11 11 11]  {
      %0 = cfdlang.eval @S : [11 11]
      %1 = cfdlang.eval @u : [11 11 11]
      %2 = cfdlang.prod %0, %1 : [11 11], [11 11 11]
      %3 = cfdlang.cont %2 : [11 11 11 11 11] indices [2 5]
      cfdlang.yield %3 : [11 11 11]
    }
    cfdlang.define @t1 : [11 11 11]  {
      %0 = cfdlang.eval @S : [11 11]
      %1 = cfdlang.eval @t0 : [11 11 11]
      %2 = cfdlang.prod %0, %1 : [11 11], [11 11 11]
      %3 = cfdlang.cont %2 : [11 11 11 11 11] indices [2 5]
      cfdlang.yield %3 : [11 11 11]
    }
    cfdlang.define @t : [11 11 11]  {
      %0 = cfdlang.eval @S : [11 11]
      %1 = cfdlang.eval @t1 : [11 11 11]
      %2 = cfdlang.prod %0, %1 : [11 11], [11 11 11]
      %3 = cfdlang.cont %2 : [11 11 11 11 11] indices [2 5]
      cfdlang.yield %3 : [11 11 11]
    }
    cfdlang.define @r : [11 11 11]  {
      %0 = cfdlang.eval @D : [11 11 11]
      %1 = cfdlang.eval @t : [11 11 11]
      %2 = cfdlang.mul %0, %1 : [11 11 11], [11 11 11]
      cfdlang.yield %2 : [11 11 11]
    }
    cfdlang.define @v0 : [11 11 11]  {
      %0 = cfdlang.eval @S : [11 11]
      %1 = cfdlang.eval @u : [11 11 11]
      %2 = cfdlang.prod %0, %1 : [11 11], [11 11 11]
      %3 = cfdlang.cont %2 : [11 11 11 11 11] indices [1 5]
      cfdlang.yield %3 : [11 11 11]
    }
    cfdlang.define @v1 : [11 11 11]  {
      %0 = cfdlang.eval @S : [11 11]
      %1 = cfdlang.eval @v0 : [11 11 11]
      %2 = cfdlang.prod %0, %1 : [11 11], [11 11 11]
      %3 = cfdlang.cont %2 : [11 11 11 11 11] indices [1 5]
      cfdlang.yield %3 : [11 11 11]
    }
    cfdlang.output @v : [11 11 11]  {
      %0 = cfdlang.eval @S : [11 11]
      %1 = cfdlang.eval @v1 : [11 11 11]
      %2 = cfdlang.prod %0, %1 : [11 11], [11 11 11]
      %3 = cfdlang.cont %2 : [11 11 11 11 11] indices [1 5]
      cfdlang.yield %3 : [11 11 11]
    }
  }
}
