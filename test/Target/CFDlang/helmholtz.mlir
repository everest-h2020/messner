module  {
  cfdlang.program  {
    cfdlang.input @S : [11 11]
    cfdlang.input @D : [11 11 11]
    cfdlang.input @u : [11 11 11]
    cfdlang.define @t : [11 11 11]  {
      %0 = cfdlang.eval @S : [11 11]
      %1 = cfdlang.eval @S : [11 11]
      %2 = cfdlang.eval @S : [11 11]
      %3 = cfdlang.eval @u : [11 11 11]
      %4 = cfdlang.prod %2, %3 : [11 11], [11 11 11]
      %5 = cfdlang.prod %1, %4 : [11 11], [11 11 11 11 11]
      %6 = cfdlang.prod %0, %5 : [11 11], [11 11 11 11 11 11 11]
      %7 = cfdlang.cont %6 : [11 11 11 11 11 11 11 11 11] indices [2 7][4 8][6 9]
      cfdlang.yield %7 : [11 11 11]
    }
    cfdlang.define @r : [11 11 11]  {
      %0 = cfdlang.eval @D : [11 11 11]
      %1 = cfdlang.eval @t : [11 11 11]
      %2 = cfdlang.mul %0, %1 : [11 11 11], [11 11 11]
      cfdlang.yield %2 : [11 11 11]
    }
    cfdlang.output @v : [11 11 11]  {
      %0 = cfdlang.eval @S : [11 11]
      %1 = cfdlang.eval @S : [11 11]
      %2 = cfdlang.eval @S : [11 11]
      %3 = cfdlang.eval @r : [11 11 11]
      %4 = cfdlang.prod %2, %3 : [11 11], [11 11 11]
      %5 = cfdlang.prod %1, %4 : [11 11], [11 11 11 11 11]
      %6 = cfdlang.prod %0, %5 : [11 11], [11 11 11 11 11 11 11]
      %7 = cfdlang.cont %6 : [11 11 11 11 11 11 11 11 11] indices [1 7][3 8][5 9]
      cfdlang.yield %7 : [11 11 11]
    }
  }
}
