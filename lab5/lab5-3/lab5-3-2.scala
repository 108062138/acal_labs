package acal_lab05.Hw3

import chisel3._
import chisel3.util._

class NumGuess(seed:Int = 1) extends Module{
    require (seed > 0 , "Seed cannot be 0")

    val io  = IO(new Bundle{
        val gen = Input(Bool())
        val guess = Input(UInt(16.W))
        val puzzle = Output(Vec(4,UInt(4.W)))
        val ready  = Output(Bool())
        val g_valid  = Output(Bool())
        val A      = Output(UInt(3.W))
        val B      = Output(UInt(3.W))

        //don't care at Hw6-3-2 but should be considered at Bonus
        val s_valid = Input(Bool())
    })

    val sIDLE :: sPRODUCE_PUZZLE :: sGUESS :: Nil = Enum(3)

    val prng = Module(new PRNG(seed))

    val state = RegInit(sIDLE)
    val next_state = Wire(UInt(4.W))
    val cnt = RegInit(0.U(5.W))
    val next_cnt = Wire(UInt(5.W))
    val rem_puzzle = RegInit(0.U(16.W))
    val next_rem_puzzle = Wire(UInt(16.W))

    val a = Wire(Vec(4, UInt(2.W)))
    val b = Wire(Vec(4, UInt(2.W)))
    val total_a = Wire(UInt(3.W))
    val total_b = Wire(UInt(3.W))
    for(i <- 0 until 4){
        val gi = io.guess(i*4+3, i*4)
        val pi = rem_puzzle(i*4+3, i*4)

        val p0 = rem_puzzle(3, 0)
        val p1 = rem_puzzle(7, 4)
        val p2 = rem_puzzle(11, 8)
        val p3 = rem_puzzle(15, 12)
        when(gi===pi){
            a(i) := 1.U
            b(i) := 0.U
        }.otherwise{
            when(gi === p0 || gi === p1 || gi === p2 || gi === p3){
                a(i) := 0.U
                b(i) := 1.U
            }.otherwise{
                a(i) := 0.U
                b(i) := 0.U
            }
        }
    }

    prng.io.gen := io.gen
    io.puzzle := prng.io.puzzle
    io.ready := prng.io.ready
    
    io.g_valid := Mux(state === sGUESS && cnt === 2.U, true.B, false.B)
    total_a := a(0) + a(1) + a(2) + a(3)
    total_b := b(0) + b(1) + b(2) + b(3)
    when(state===sGUESS){
        when(a(0)===1.U && a(1)===1.U && a(2)===1.U && a(3)===1.U){
            // wtf
            io.A := 4.U
        }.otherwise{
            io.A := total_a
        }
        when(b(0)===1.U && b(1)===1.U && b(2)===1.U && b(3)===1.U){
            io.B := 4.U
        }.otherwise{
            io.B := total_b
        }
    }.otherwise{
        io.A := 0.U
        io.B := 0.U
    }
    
    next_state := state
    next_cnt := cnt
    next_rem_puzzle := rem_puzzle

    when(state===sPRODUCE_PUZZLE){
        when(prng.io.ready){
            next_rem_puzzle := Cat(prng.io.puzzle(3), prng.io.puzzle(2), prng.io.puzzle(1), prng.io.puzzle(0))
        }
    }

    switch(state){
        is(sIDLE){
            when(io.gen){
                next_state := sPRODUCE_PUZZLE
                next_cnt := 0.U
            }
        }
        is(sPRODUCE_PUZZLE){
            when(prng.io.ready){
                next_state := sGUESS
                next_cnt := 0.U
            }
        }
        is(sGUESS){
            when(cnt ===2.U){
                val all_a = a(0) + a(1) + a(2) + a(3)
                when(all_a === 4.U){
                    next_state := sIDLE
                }.otherwise{
                    next_state := sGUESS
                }
                next_cnt := 0.U
            }.otherwise{
                next_cnt := cnt + 1.U
                next_state := sGUESS
            }
        }
    }

    state:= next_state
    cnt := next_cnt
    rem_puzzle := next_rem_puzzle
}