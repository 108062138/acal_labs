package acal_lab05.Hw1

import chisel3._
import chisel3.util._

class TrafficLight_p(Ytime:Int, Gtime:Int, Ptime:Int) extends Module{
  val io = IO(new Bundle{
    val P_button = Input(Bool())
    val H_traffic = Output(UInt(2.W))
    val V_traffic = Output(UInt(2.W))
    val P_traffic = Output(UInt(2.W))
    val timer     = Output(UInt(5.W))
    val cur_state = Output(UInt(4.W))
  })

  //please implement your code below...
  // here tests
  //parameter declaration
  val Off = 0.U
  val Red = 1.U
  val Yellow = 2.U
  val Green = 3.U

  val sIdle :: sHGVR :: sHYVR :: sHRVG :: sHRVY :: sPG  :: Nil = Enum(6)
  // state: record the current state, which is updated by next_state
  // come_from_state: remember the special change source in fsm. It is updated by next_come_from_state
  // cnt: counter, updated bu next_cnt
  // cnt_done: just a normal wire that indicate finish the counting and enter the next state in fsm
  val state = RegInit(sIdle)
  val next_state = Wire(UInt(4.W))
  val come_from_state = RegInit(sIdle)
  val next_come_from_state = Wire(UInt(4.W))
  val cnt = RegInit(0.U(4.W))
  val next_cnt = Wire(UInt(4.W))
  val cnt_done = Wire(Bool())

  // wire connection
  cnt_done := cnt === 0.U
  
  io.timer := cnt
  io.cur_state := state
  io.H_traffic := Off
  io.V_traffic := Off
  io.P_traffic := Off

  switch(state){
    is(sHGVR){
      io.H_traffic := Green
      io.V_traffic := Red
      io.P_traffic := Red
    }
    is(sHYVR){
      io.H_traffic := Yellow
      io.V_traffic := Red
      io.P_traffic := Red
    }
    is(sHRVG){
      io.H_traffic := Red
      io.V_traffic := Green
      io.P_traffic := Red
    }
    is(sHRVY){
      io.H_traffic := Red
      io.V_traffic := Yellow
      io.P_traffic := Red
    }
    is(sPG){
      io.H_traffic := Red
      io.V_traffic := Red
      io.P_traffic := Green
    }
  }
  // handling state
  state := next_state
  when(state===sIdle){
    next_state := sHGVR
  }.otherwise{
    when(io.P_button){
      next_state := sPG;
    }.otherwise{
      when(cnt_done){
        when(state===sHGVR)      {next_state := sHYVR}
        .elsewhen(state===sHYVR) {next_state := sHRVG}
        .elsewhen(state===sHRVG) {next_state := sHRVY}
        .elsewhen(state===sHRVY) {next_state :=   sPG}
        .otherwise{
          // recover the the state
          when(come_from_state===sIdle)     {next_state := sHGVR}
          .elsewhen(come_from_state===sHGVR){next_state := sHGVR}
          .elsewhen(come_from_state===sHYVR){next_state := sHYVR}
          .elsewhen(come_from_state===sHRVG){next_state := sHRVG}
          .otherwise                        {next_state := sHRVY}
        }
      }.otherwise{next_state := state}
    }
  }

  // handling cnt
  cnt:= next_cnt
  println(cnt)
  when(cnt_done){
    when(state===sHGVR)     {next_cnt := Ytime.U - 1.U}
    .elsewhen(state===sHYVR){next_cnt := Gtime.U - 1.U}
    .elsewhen(state===sHRVG){next_cnt := Ytime.U - 1.U}
    .elsewhen(state===sHRVY){next_cnt := Ptime.U - 1.U}
    .elsewhen(state===sIdle){next_cnt := Gtime.U - 1.U}
    .otherwise{
      when(come_from_state===sIdle)     {next_cnt := Gtime.U - 1.U}
      .elsewhen(come_from_state===sHGVR){next_cnt := Gtime.U - 1.U}
      .elsewhen(come_from_state===sHYVR){next_cnt := Ytime.U - 1.U}
      .elsewhen(come_from_state===sHRVG){next_cnt := Gtime.U - 1.U}
      .otherwise                        {next_cnt := Ytime.U - 1.U}
    }
  }.otherwise{
    when(io.P_button && state =/= sPG){next_cnt := Ptime.U - 1.U}
    .otherwise                     {next_cnt := cnt - 1.U}
  }

  printf(p"come_from_state: $come_from_state dddd wtf here???\n")
  // handling come_from_state
  come_from_state := next_come_from_state
  when(io.P_button && state =/= sPG){
    next_come_from_state := state
  }.otherwise{
    when(state === sPG){next_come_from_state := come_from_state}
    .otherwise         {next_come_from_state := sIdle}
  }
}