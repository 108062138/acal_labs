package acal_lab05.Hw3

import chisel3._
import chisel3.util._

class PRNG(seed: Int) extends Module {
  val io = IO(new Bundle {
    val gen = Input(Bool())
    val puzzle = Output(Vec(4, UInt(4.W)))
    val ready = Output(Bool())
  })

  val sIDLE :: sWAIT :: sCOMPUTE :: sFINISH :: Nil = Enum(4)
  val state = RegInit(sWAIT)
  val next_state = WireDefault(state)

  val cnt = RegInit(sCOMPUTE)
  val next_cnt = Wire(UInt(4.W))
  val lfsr = RegInit(seed.U(16.W))
  val next_lfsr = Wire(UInt(16.W))
  next_lfsr := lfsr
  next_cnt := cnt

  // Default outputs
  io.puzzle := VecInit(Seq.fill(4)(0.U(4.W)))
  io.ready := state === sFINISH

  val sequences = VecInit(
    VecInit(0.U, 1.U, 4.U, 3.U, 2.U, 5.U, 6.U, 7.U, 8.U, 9.U),
    VecInit(1.U, 5.U, 3.U, 7.U, 9.U, 2.U, 0.U, 8.U, 6.U, 4.U),
    VecInit(2.U, 5.U, 7.U, 3.U, 0.U, 6.U, 4.U, 8.U, 9.U, 1.U),
    VecInit(3.U, 1.U, 4.U, 5.U, 9.U, 2.U, 6.U, 0.U, 7.U, 8.U),
    VecInit(4.U, 6.U, 5.U, 3.U, 7.U, 1.U, 9.U, 2.U, 0.U, 8.U),
    VecInit(5.U, 4.U, 7.U, 3.U, 1.U, 0.U, 2.U, 8.U, 6.U, 9.U),
    VecInit(9.U, 4.U, 5.U, 6.U, 2.U, 7.U, 3.U, 0.U, 1.U, 8.U),
    VecInit(0.U, 5.U, 4.U, 3.U, 1.U, 2.U, 9.U, 6.U, 8.U, 7.U),
    VecInit(8.U, 1.U, 3.U, 5.U, 7.U, 9.U, 0.U, 4.U, 6.U, 0.U),
    VecInit(9.U, 7.U, 5.U, 8.U, 1.U, 0.U, 2.U, 4.U, 6.U, 3.U),
    VecInit(0.U, 8.U, 4.U, 6.U, 2.U, 1.U, 3.U, 5.U, 7.U, 9.U),
    VecInit(9.U, 0.U, 3.U, 6.U, 1.U, 2.U, 5.U, 8.U, 4.U, 7.U),
    VecInit(2.U, 4.U, 6.U, 8.U, 0.U, 3.U, 7.U, 5.U, 9.U, 1.U),
    VecInit(3.U, 6.U, 8.U, 2.U, 5.U, 1.U, 9.U, 4.U, 7.U, 0.U),
    VecInit(9.U, 7.U, 0.U, 3.U, 6.U, 4.U, 2.U, 8.U, 5.U, 1.U),
    VecInit(5.U, 8.U, 1.U, 4.U, 0.U, 7.U, 3.U, 6.U, 9.U, 2.U) 
  )

  val rotation = lfsr(3, 0) % 10.U  // Calculate rotation based on LFSR

  switch(state) {
    is(sIDLE) {
      next_state := sWAIT
    }
    is(sWAIT) {
      when(io.gen) {
        next_state := sCOMPUTE
      }
    }
    is(sCOMPUTE) {
      val feedback = lfsr(15) ^ lfsr(13) ^ lfsr(12) ^ lfsr(10)
      next_lfsr := Cat(lfsr(14, 0), feedback)
      next_state := sFINISH
    }
    is(sFINISH) {
      for(i <- 0 until 4){
        io.puzzle(i) := sequences(cnt)((i.U+rotation) % 10.U)
      }
      next_state := sWAIT
      next_cnt := cnt + 1.U
    }
  }

  // Apply updates
  state := next_state
  lfsr := next_lfsr
  cnt := next_cnt
}
