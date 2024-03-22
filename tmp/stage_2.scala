package acal_lab05.Hw2

import chisel3._
import chisel3.util._

class LongCal extends Module{
    val io = IO(new Bundle{
        val key_in = Input(UInt(4.W))
        val value = Output(Valid(UInt(32.W)))
    })

    //please implement your code below

    val sFIRST_IDLE:: sACCEPT :: sPOS :: sSEE_LP :: sNEG :: sFIRST_DEMO :: sSECOND_IDLE :: sREMOVE_PARA :: sFIND_SMALLER_OPERATOR :: sLOOKING_FOR_LP :: sDUMP :: sSECOND_DEMO ::sFINISH ::Nil = Enum(13)
    val sZERO:: sONE :: sTWO :: sTHREE :: sFOUR :: sFIVE :: sSIX :: sSEVEN :: sEIGHT :: sNINE :: sADD :: sMINUS :: sMUL :: sLEFT_PARA :: sRIGHT_PARA :: sASSIGN :: Nil = Enum(16)
    
    val state = RegInit(sFIRST_IDLE)
    val next_state = Wire(UInt(5.W))

    val psum = RegInit(0.S(32.W))
    val next_psum = Wire(SInt(32.W))

    val peek_cnt = RegInit(0.U(log2Ceil(36 + 1).W))
    val next_peek_cnt = Wire(UInt(log2Ceil(36 + 1).W))
    val res_sp = RegInit(0.U(log2Ceil(36 + 1).W))
    val next_res_sp = Wire(UInt(log2Ceil(36 + 1).W))
    

    val removed_neg_tokens_tag = Wire(UInt(4.W))
    val content = Wire(SInt(32.W))///

    val operator_st_top_tag = Wire(UInt(4.W))

    // define depth:56 and width:36
    val removed_neg_tokens_stack = Mem(56, UInt(36.W))
    val removed_neg_tokens_sp = RegInit(0.U(log2Ceil(36 + 1).W))
    val next_removed_neg_tokens_sp = Wire(UInt(log2Ceil(36 + 1).W))

    val removed_para_tokens_stack = Mem(56, UInt(36.W))
    val removed_para_tokens_sp = RegInit(0.U(log2Ceil(36 + 1).W))
    val next_removed_para_tokens_sp = Wire(UInt(log2Ceil(36 + 1).W))

    val operator_tokens_stack = Mem(56, UInt(36.W))
    val operator_tokens_sp = RegInit(0.U(log2Ceil(36 + 1).W))
    val next_operator_tokens_sp = Wire(UInt(log2Ceil(36 + 1).W))

    val travel_removed_neg_sp = RegInit(0.U(log2Ceil(36 + 1).W))
    val next_travel_removed_neg_sp = Wire(UInt(log2Ceil(36 + 1).W))

    val extendedKeyIn = Cat(0.U(28.W), io.key_in)
    val in_buffer = RegNext(extendedKeyIn, init = 0.U(32.W))
    val encoded_in_buffer = Wire(UInt(4.W))
    val is_digit = Wire(Bool())
    val is_operator = Wire(Bool())

    io.value.valid := Mux(state === sFINISH, true.B, false.B)
    io.value.bits := removed_neg_tokens_stack(res_sp)(31, 0)

    encoded_in_buffer := in_buffer
    is_digit := in_buffer === sZERO || in_buffer === sONE || in_buffer === sTWO || in_buffer === sTHREE || in_buffer === sFOUR || in_buffer === sFIVE || in_buffer === sSIX || in_buffer === sSEVEN || in_buffer === sEIGHT || in_buffer === sNINE
    is_operator := in_buffer === sADD || in_buffer === sMINUS || in_buffer === sMUL

    // sequential part
    state := next_state
    removed_neg_tokens_sp := next_removed_neg_tokens_sp
    psum := next_psum
    peek_cnt := next_peek_cnt
    res_sp := next_res_sp
    removed_para_tokens_sp := next_removed_para_tokens_sp
    operator_tokens_sp := next_operator_tokens_sp
    travel_removed_neg_sp := next_travel_removed_neg_sp

    // init combinational
    next_removed_neg_tokens_sp := removed_neg_tokens_sp
    next_state := state
    next_psum := psum
    next_peek_cnt := peek_cnt
    next_removed_para_tokens_sp := removed_para_tokens_sp
    next_operator_tokens_sp := operator_tokens_sp
    next_travel_removed_neg_sp := travel_removed_neg_sp
    
    when(state===sFIRST_DEMO){
        removed_neg_tokens_tag := removed_neg_tokens_stack(peek_cnt)(35,32)
        content := removed_neg_tokens_stack(peek_cnt)(31,0).asSInt()
    }.otherwise{
        removed_neg_tokens_tag := removed_neg_tokens_stack(travel_removed_neg_sp)(35,32)
        content := removed_neg_tokens_stack(travel_removed_neg_sp)(31,0).asSInt()
    }

    operator_st_top_tag := operator_tokens_stack(operator_tokens_sp - 1.U)(35,32)

    val top_is_beq = Wire(Bool())
    val operator_st_top_tag_priority = Wire(UInt(4.W))
    val removed_neg_tokens_tag_priority = Wire(UInt(4.W))

    operator_st_top_tag_priority := 0.U
    switch(operator_st_top_tag){
        is(sLEFT_PARA){
            operator_st_top_tag_priority := 0.U
        }
        is(sADD){
            operator_st_top_tag_priority := 1.U
        }
        is(sMINUS){
            operator_st_top_tag_priority := 1.U
        }
        is(sMUL){
            operator_st_top_tag_priority := 2.U
        }
    }
    removed_neg_tokens_tag_priority := 0.U
    switch(removed_neg_tokens_tag){
        is(sLEFT_PARA){
            removed_neg_tokens_tag_priority := 0.U
        }
        is(sADD){
            removed_neg_tokens_tag_priority := 1.U
        }
        is(sMINUS){
            removed_neg_tokens_tag_priority := 1.U
        }
        is(sMUL){
            removed_neg_tokens_tag_priority := 2.U
        }
    }

    top_is_beq := removed_neg_tokens_tag_priority <= operator_st_top_tag_priority

    next_res_sp := res_sp

    // my printer
    switch(state){
        is(sFIRST_IDLE){
            printf(p"sFIRST_IDLE\n")
        }
        is(sACCEPT){
            printf(p"sACCEPT\n")
        }
        is(sPOS){
            printf(p"sPOS\n")
        }
        is(sSEE_LP){
            printf(p"sSEE_LP\n")
        }
        is(sNEG){
            printf(p"sNEG\n")
        }
        is(sFIRST_DEMO){
            // printf(p"sFIRST_DEMO\n")
            next_peek_cnt := peek_cnt + 1.U
            when(removed_neg_tokens_tag===0.U){
                next_res_sp := peek_cnt
                printf(p"At demo $peek_cnt/$removed_neg_tokens_sp, I see $content\n")
            }.elsewhen(removed_neg_tokens_tag===sADD){
                printf(p"At demo $peek_cnt/$removed_neg_tokens_sp, I see +\n")
            }.elsewhen(removed_neg_tokens_tag===sMINUS){
                printf(p"At demo $peek_cnt/$removed_neg_tokens_sp, I see -\n")
            }.elsewhen(removed_neg_tokens_tag===sMUL){
                printf(p"At demo $peek_cnt/$removed_neg_tokens_sp, I see *\n")
            }.elsewhen(removed_neg_tokens_tag===sLEFT_PARA){
                printf(p"At demo $peek_cnt/$removed_neg_tokens_sp, I see (\n")
            }.elsewhen(removed_neg_tokens_tag===sRIGHT_PARA){
                printf(p"At demo $peek_cnt/$removed_neg_tokens_sp, I see )\n")
            }.elsewhen(removed_neg_tokens_tag===sASSIGN){
                printf(p"At demo $peek_cnt/$removed_neg_tokens_sp, I see =\n")
            }
        }
        is(sSECOND_IDLE){
            printf(p"sSECOND_IDLE\n")
        }
        is(sREMOVE_PARA){
            //printf(p"sREMOVE_PARA, locate at: $travel_removed_neg_sp\n")
            // when(removed_neg_tokens_tag===0.U){
            //     printf(p"At remove para $travel_removed_neg_sp/$removed_neg_tokens_sp, I see $content\n")
            // }.elsewhen(removed_neg_tokens_tag===sADD){
            //     printf(p"At remove para $travel_removed_neg_sp/$removed_neg_tokens_sp, I see +\n")
            // }.elsewhen(removed_neg_tokens_tag===sMINUS){
            //     printf(p"At remove para $travel_removed_neg_sp/$removed_neg_tokens_sp, I see -\n")
            // }.elsewhen(removed_neg_tokens_tag===sMUL){
            //     printf(p"At remove para $travel_removed_neg_sp/$removed_neg_tokens_sp, I see *\n")
            // }.elsewhen(removed_neg_tokens_tag===sLEFT_PARA){
            //     printf(p"At remove para $travel_removed_neg_sp/$removed_neg_tokens_sp, I see (\n")
            // }.elsewhen(removed_neg_tokens_tag===sRIGHT_PARA){
            //     printf(p"At remove para $travel_removed_neg_sp/$removed_neg_tokens_sp, I see )\n")
            // }.elsewhen(removed_neg_tokens_tag===sASSIGN){
            //     printf(p"At remove para $travel_removed_neg_sp/$removed_neg_tokens_sp, I see =\n")
            // }
        }
        is(sFIND_SMALLER_OPERATOR){
        }
        is(sLOOKING_FOR_LP){
        }
        is(sDUMP){
        }
        is(sFINISH){
            printf(p"sFINISH\n")
            next_peek_cnt := 0.U
        }
    }

    // handle next_travel_removed_neg_sp
    switch(state){
        is(sREMOVE_PARA){
            when(removed_neg_tokens_tag === 0.U){
                // is a number. put to res and move on
                // put to res
                next_removed_para_tokens_sp := removed_para_tokens_sp + 1.U
                removed_para_tokens_stack(removed_para_tokens_sp) := removed_neg_tokens_stack(travel_removed_neg_sp)
                printf(p"at REMOVE, put $content  into stack at $removed_para_tokens_sp\n")
                // move on
                next_travel_removed_neg_sp := travel_removed_neg_sp + 1.U
            }.elsewhen(removed_neg_tokens_tag === sLEFT_PARA){
                // is (. put to op stack and move on
                // put to op-stack
                next_operator_tokens_sp := operator_tokens_sp + 1.U
                operator_tokens_stack(operator_tokens_sp) := removed_neg_tokens_stack(travel_removed_neg_sp)
                // move on
                next_travel_removed_neg_sp := travel_removed_neg_sp + 1.U
            }.elsewhen(removed_neg_tokens_tag === sASSIGN){
                // meet assign, just move on
                next_travel_removed_neg_sp := travel_removed_neg_sp + 1.U
            }
        }
        is(sFIND_SMALLER_OPERATOR){
            // since sp always points to the current empty one, we have to cal. sp-1 to get top
            when(operator_tokens_sp === 0.U){
                // edge case: empty op stack. we just put op inside op stack and move on

                // put op inside op stack
                next_operator_tokens_sp := operator_tokens_sp + 1.U
                operator_tokens_stack(operator_tokens_sp) := removed_neg_tokens_stack(travel_removed_neg_sp)
                // move on
                next_travel_removed_neg_sp := travel_removed_neg_sp + 1.U
            }.otherwise{
                when(!top_is_beq){
                    // branch case: top beq is violated
                    // put op inside op stack
                    next_operator_tokens_sp := operator_tokens_sp + 1.U
                    operator_tokens_stack(operator_tokens_sp) := removed_neg_tokens_stack(travel_removed_neg_sp)
                    // move on
                    next_travel_removed_neg_sp := travel_removed_neg_sp + 1.U
                }.otherwise{
                    // normal case: put op out of op-stack into res. don't move on
                    next_operator_tokens_sp := operator_tokens_sp - 1.U
                    next_removed_para_tokens_sp := removed_para_tokens_sp + 1.U
                    removed_para_tokens_stack(removed_para_tokens_sp) := operator_tokens_stack(operator_tokens_sp - 1.U)
                    when(operator_st_top_tag === sADD){
                        printf(p"find smaller put +  into stack at $removed_para_tokens_sp\n")
                    }.elsewhen(operator_st_top_tag === sMINUS){
                        printf(p"find smaller put -  into stack at $removed_para_tokens_sp\n")
                    }.elsewhen(operator_st_top_tag === sMUL){
                        printf(p"find smaller put *  into stack at $removed_para_tokens_sp\n")
                    }
                }
            }
        }
        is(sLOOKING_FOR_LP){
            when(operator_st_top_tag===sLEFT_PARA){
                // see (, just pop it out from op stack and move on
                // put it out from op stack
                next_operator_tokens_sp := operator_tokens_sp - 1.U
                // move on
                next_travel_removed_neg_sp := travel_removed_neg_sp + 1.U
            }.otherwise{
                // normal case, pop op out of op-stack into rs. don't move on
                next_operator_tokens_sp := operator_tokens_sp - 1.U
                next_removed_para_tokens_sp := removed_para_tokens_sp + 1.U
                removed_para_tokens_stack(removed_para_tokens_sp) := operator_tokens_stack(operator_tokens_sp - 1.U)

                when(operator_st_top_tag === sADD){
                    printf(p"at lp, put +  into stack at $removed_para_tokens_sp\n")
                }.elsewhen(operator_st_top_tag === sMINUS){
                    printf(p"at lp, put -  into stack at $removed_para_tokens_sp\n")
                }.elsewhen(operator_st_top_tag === sMUL){
                    printf(p"at lp, put *  into stack at $removed_para_tokens_sp\n")
                }
            }
        }
        is(sDUMP){
            when(operator_tokens_sp>0.U){
                next_operator_tokens_sp := operator_tokens_sp - 1.U
                next_removed_para_tokens_sp := removed_para_tokens_sp + 1.U
                removed_para_tokens_stack(removed_para_tokens_sp) := operator_tokens_stack(operator_tokens_sp - 1.U)

                when(operator_st_top_tag === sADD){
                    printf(p"at dump, put +  into stack at $removed_para_tokens_sp\n")
                }.elsewhen(operator_st_top_tag === sMINUS){
                    printf(p"at dump, put -  into stack at $removed_para_tokens_sp\n")
                }.elsewhen(operator_st_top_tag === sMUL){
                    printf(p"at dump, put *  into stack at $removed_para_tokens_sp\n")
                }
            }
        }
        is(sFIRST_DEMO){
            next_travel_removed_neg_sp := 0.U
        }
        is(sFINISH){
            next_travel_removed_neg_sp := 0.U
            next_operator_tokens_sp := 0.U
            next_removed_para_tokens_sp := 0.U
        }
    }
    
    // handle next_psum
    switch(state){
        is(sFIRST_IDLE){
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
        is(sFIRST_DEMO){
            next_psum := 0.S(32.W)
        }
        is(sFINISH){
            next_psum := 0.S(32.W)
        }
    }

    // generate next_removed_neg_tokens_sp
    switch(state){
        is(sFIRST_IDLE){
            next_removed_neg_tokens_sp := 0.U
        }
        is(sACCEPT){
            when(!is_digit){
                removed_neg_tokens_stack(removed_neg_tokens_sp) := Cat(encoded_in_buffer, psum)
                next_removed_neg_tokens_sp := removed_neg_tokens_sp + 1.U
            }
        }
        is(sPOS){
            when(is_operator || in_buffer === sRIGHT_PARA || in_buffer === sASSIGN){
                removed_neg_tokens_stack(removed_neg_tokens_sp) := Cat(0.U(4.W), psum)
                removed_neg_tokens_stack(removed_neg_tokens_sp + 1.U) := Cat(encoded_in_buffer, psum)
                next_removed_neg_tokens_sp := removed_neg_tokens_sp + 2.U
            }
        }
        is(sSEE_LP){
            when(in_buffer === sLEFT_PARA){
                removed_neg_tokens_stack(removed_neg_tokens_sp) := Cat(encoded_in_buffer, psum)
                next_removed_neg_tokens_sp := removed_neg_tokens_sp + 1.U
            }
        }
        is(sNEG){
            when(in_buffer === sRIGHT_PARA){
                removed_neg_tokens_stack(removed_neg_tokens_sp) := Cat(0.U(4.W), -psum)
                removed_neg_tokens_stack(removed_neg_tokens_sp + 1.U) := Cat(encoded_in_buffer, psum)
                next_removed_neg_tokens_sp := removed_neg_tokens_sp + 2.U
            }
        }
        is(sFIRST_DEMO){
            next_removed_neg_tokens_sp := removed_neg_tokens_sp
        }
        is(sFINISH){
            next_removed_neg_tokens_sp := 0.U
        }
    }

    switch(state){
        is(sFIRST_IDLE){
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
                next_state := sFIRST_DEMO
            }
        }
        is(sPOS){
            when(is_digit){
                next_state := sPOS
            }.elsewhen(is_operator || in_buffer === sRIGHT_PARA){
                next_state := sACCEPT
            }.otherwise{
                next_state := sFIRST_DEMO
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
        is(sFIRST_DEMO){
            when(peek_cnt === removed_neg_tokens_sp - 1.U){
                next_state := sSECOND_IDLE
            }.otherwise{
                next_state := sFIRST_DEMO
            }
        }
        is(sSECOND_IDLE){
            next_state := sREMOVE_PARA
        }
        is(sREMOVE_PARA){
            when(removed_neg_tokens_tag === sLEFT_PARA){
                next_state := sREMOVE_PARA
            }.elsewhen(removed_neg_tokens_tag === 0.U){
                next_state := sREMOVE_PARA
            }.elsewhen(removed_neg_tokens_tag === sRIGHT_PARA){
                next_state := sLOOKING_FOR_LP
            }.elsewhen(removed_neg_tokens_tag === sADD || removed_neg_tokens_tag === sMINUS || removed_neg_tokens_tag === sMUL){
                next_state := sFIND_SMALLER_OPERATOR
            }.otherwise{
                // operator_st_top_tag === sASSIGN
                next_state := sDUMP
            }
        }
        is(sFIND_SMALLER_OPERATOR){
            when(operator_tokens_sp>0.U){
                when(top_is_beq){
                    next_state := sFIND_SMALLER_OPERATOR
                }.otherwise{
                    next_state := sREMOVE_PARA
                }
            }.otherwise{
                next_state := sREMOVE_PARA
            }
        }
        is(sLOOKING_FOR_LP){
            when(operator_st_top_tag===sLEFT_PARA){
                next_state := sREMOVE_PARA
            }.otherwise{
                next_state := sLOOKING_FOR_LP
            }
        }
        is(sDUMP){
            // I intentionally skip this one
            when(operator_tokens_sp>0.U){
                next_state := sDUMP
            }.otherwise{
                next_state := sFINISH
            }
        }
        is(sSECOND_DEMO){
            next_state := sFINISH
        }
        is(sFINISH){
            next_state := sFIRST_IDLE
        }
    }
}