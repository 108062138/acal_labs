package acal_lab05.Hw2

import chisel3._
import chisel3.util._

class NegIntGen extends Module{
    val io = IO(new Bundle{
        val key_in = Input(UInt(4.W))
        val value = Output(Valid(UInt(32.W)))
    })

    //please implement your code below
    io.value.valid := false.B
    io.value.bits := 0.U

    val sIDLE:: sACCEPT :: sPOS :: sSEE_LP :: sNEG :: sDEMO :: sFINISH :: Nil = Enum(7)
    val sZERO:: sONE :: sTWO :: sTHREE :: sFOUR :: sFIVE :: sSIX :: sSEVEN :: sEIGHT :: sNINE :: sADD :: sMINUS :: sMUL :: sLEFT_PARA :: sRIGHT_PARA :: sASSIGN :: Nil = Enum(16)
    
    val state = RegInit(sIDLE)
    val next_state = Wire(UInt(4.W))

    val psum = RegInit(0.S(32.W))
    val next_psum = Wire(SInt(32.W))

    val peek_cnt = RegInit(0.U(log2Ceil(36 + 1).W))
    val next_peek_cnt = Wire(UInt(log2Ceil(36 + 1).W))
    val res_sp = RegInit(0.U(log2Ceil(36 + 1).W))
    val next_res_sp = Wire(UInt(log2Ceil(36 + 1).W))
    

    val tag = Wire(UInt(4.W))
    val content = Wire(SInt(32.W))///

    // define depth and width

    val tokens_stack = Mem(56, UInt(36.W))
    val tokens_sp = RegInit(0.U(log2Ceil(36 + 1).W))
    val next_tokens_sp = Wire(UInt(log2Ceil(36 + 1).W))

    val extendedKeyIn = Cat(0.U(28.W), io.key_in)
    val in_buffer = RegNext(extendedKeyIn, init = 0.U(32.W))
    val encoded_in_buffer = Wire(UInt(4.W))
    val is_digit = Wire(Bool())
    val is_operator = Wire(Bool())

    io.value.valid := Mux(state === sFINISH, true.B, false.B)
    io.value.bits := tokens_stack(res_sp)(31, 0)

    encoded_in_buffer := in_buffer
    is_digit := in_buffer === sZERO || in_buffer === sONE || in_buffer === sTWO || in_buffer === sTHREE || in_buffer === sFOUR || in_buffer === sFIVE || in_buffer === sSIX || in_buffer === sSEVEN || in_buffer === sEIGHT || in_buffer === sNINE
    is_operator := in_buffer === sADD || in_buffer === sMINUS || in_buffer === sMUL

    state := next_state
    tokens_sp := next_tokens_sp
    psum := next_psum
    peek_cnt := next_peek_cnt
    res_sp := next_res_sp

    next_tokens_sp := tokens_sp
    next_state := state
    next_psum := psum
    next_peek_cnt := peek_cnt
    
    tag := tokens_stack(peek_cnt)(35,32)
    content := tokens_stack(peek_cnt)(31,0).asSInt()

    next_res_sp := res_sp
    when(state===sDEMO){
        // printf(p"Inside demo , sp $tokens_sp and current peek_cnt: $peek_cnt\n")
        next_peek_cnt := peek_cnt + 1.U
        when(tag===0.U){
            next_res_sp := peek_cnt
            printf(p"At demo $peek_cnt/$tokens_sp, I see $content\n")
        }.elsewhen(tag===sADD){
            printf(p"At demo $peek_cnt/$tokens_sp, I see +\n")
        }.elsewhen(tag===sMINUS){
            printf(p"At demo $peek_cnt/$tokens_sp, I see -\n")
        }.elsewhen(tag===sMUL){
            printf(p"At demo $peek_cnt/$tokens_sp, I see *\n")
        }.elsewhen(tag===sLEFT_PARA){
            printf(p"At demo $peek_cnt/$tokens_sp, I see (\n")
        }.elsewhen(tag===sRIGHT_PARA){
            printf(p"At demo $peek_cnt/$tokens_sp, I see )\n")
        }.elsewhen(tag===sASSIGN){
            printf(p"At demo $peek_cnt/$tokens_sp, I see =\n")
        }
    }.otherwise{
        next_peek_cnt := 0.U
    }

    when(is_digit){
        printf(p"At state: $state , I see $in_buffer\n")
    }.elsewhen(in_buffer === sADD){
        printf(p"At state: $state , I see +\n")
    }.elsewhen(in_buffer === sMINUS){
        printf(p"At state: $state , I see -\n")
    }.elsewhen(in_buffer === sMUL){
        printf(p"At state: $state , I see *\n")
    }.elsewhen(in_buffer === sLEFT_PARA){
        printf(p"At state: $state , I see (\n")
    }.elsewhen(in_buffer === sRIGHT_PARA){
        printf(p"At state: $state , I see )\n")
    }.otherwise{
        printf(p"At state: $state , I see =\n")
    }
    // printf(p"at state $state, psum: $psum ~~~\n")
    
    switch(state){
        is(sIDLE){
            next_psum := 0.S(32.W)
        }
        is(sACCEPT){
            when(is_digit){
                next_psum := in_buffer.asSInt()
            }.otherwise{
                next_psum := 0.S(32.W)
            }
        }
        is(sPOS){
            when(is_digit){
                printf(p"At sPOS , psum: $psum, <<3: ${psum<<3.U}, <<1: ${psum <<1.U}, and inbuffer: while${in_buffer.asSInt()}, I see =\n")
                next_psum := (psum <<3.U) + (psum <<1.U) + in_buffer.asSInt()
            }.otherwise{
                next_psum := 0.S(32.W)
            }
        }
        is(sSEE_LP){
            when(is_digit){
                next_psum := in_buffer.asSInt()
            }
        }
        is(sNEG){
            when(is_digit){
                next_psum := (psum <<3.U) + (psum <<1.U) + in_buffer.asSInt()
            }.otherwise{
                next_psum := 0.S(32.W)
            }
        }
        is(sDEMO){
            next_psum := 0.S(32.W)
        }
        is(sFINISH){
            next_psum := 0.S(32.W)
        }
    }

    switch(state){
        is(sIDLE){
            next_tokens_sp := 0.U
        }
        is(sACCEPT){
            when(!is_digit){
                tokens_stack(tokens_sp) := Cat(encoded_in_buffer, psum)
                next_tokens_sp := tokens_sp + 1.U

                when(is_digit){
                    printf(p"At AC, I place $in_buffer\n")
                }.elsewhen(in_buffer === sADD){
                    printf(p"At AC, I see +\n")
                }.elsewhen(in_buffer === sMINUS){
                    printf(p"At AC, I see -\n")
                }.elsewhen(in_buffer === sMUL){
                    printf(p"At AC, I see *\n")
                }.elsewhen(in_buffer === sLEFT_PARA){
                    printf(p"At AC, I see (\n")
                }.elsewhen(in_buffer === sRIGHT_PARA){
                    printf(p"At AC, I see )\n")
                }.otherwise{
                    printf(p"At AC, I see =\n")
                }
            }
        }
        is(sPOS){
            when(is_operator || in_buffer === sRIGHT_PARA || in_buffer === sASSIGN){
                tokens_stack(tokens_sp) := Cat(0.U(4.W), psum)
                tokens_stack(tokens_sp + 1.U) := Cat(encoded_in_buffer, psum)
                next_tokens_sp := tokens_sp + 2.U
            }
        }
        is(sSEE_LP){
            when(in_buffer === sLEFT_PARA){
                tokens_stack(tokens_sp) := Cat(encoded_in_buffer, psum)
                next_tokens_sp := tokens_sp + 1.U
            }
        }
        is(sNEG){
            when(in_buffer === sRIGHT_PARA){
                tokens_stack(tokens_sp) := Cat(0.U(4.W), -psum)
                tokens_stack(tokens_sp + 1.U) := Cat(encoded_in_buffer, psum)
                next_tokens_sp := tokens_sp + 2.U
            }
        }
        is(sDEMO){
            next_tokens_sp := tokens_sp
        }
        is(sFINISH){
            next_tokens_sp := 0.U
        }
    }

    switch(state){
        is(sIDLE){
            next_state := sACCEPT
        }
        is(sACCEPT){
            when(is_digit){
                next_state := sPOS
            }.elsewhen(is_operator || in_buffer === sRIGHT_PARA){
                next_state := sACCEPT
            }.elsewhen(in_buffer === sLEFT_PARA){
                next_state := sSEE_LP
            }.otherwise{
                next_state := sDEMO
            }
        }
        is(sPOS){
            when(is_digit){
                next_state := sPOS
            }.elsewhen(is_operator || in_buffer === sRIGHT_PARA){
                next_state := sACCEPT
            }.otherwise{
                next_state := sDEMO
            }
        }
        is(sSEE_LP){
            when(is_digit){
                next_state := sPOS
            }.elsewhen(in_buffer === sLEFT_PARA){
                next_state := sSEE_LP
            }.otherwise{
                // sNEGative operand
                next_state := sNEG
            }
        }
        is(sNEG){
            when(is_digit){
                next_state := sNEG
            }.otherwise{
                // see ')'

                next_state := sACCEPT
            }
        }
        is(sDEMO){
            when(peek_cnt === tokens_sp - 1.U){
                next_state := sFINISH
            }.otherwise{
                next_state := sDEMO
            }
        }
        is(sFINISH){
            next_state := sIDLE
        }
        // so something here
    }
}