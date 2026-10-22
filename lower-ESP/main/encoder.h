#ifndef ENCODER_H
#define ENCODER_H

#include "driver/pulse_cnt.h"

// ====================== 编码器引脚 ======================
// 左电机 (M1) — E1A=GPIO36, E1B=GPIO35
#define ENC_LEFT_A      36
#define ENC_LEFT_B      35

// 右电机 (M2) — E2A=GPIO38, E2B=GPIO37
#define ENC_RIGHT_A     38
#define ENC_RIGHT_B     37

// ====================== 外部接口 ======================
void encoder_init(void);
int  encoder_read_left(void);     // 读左轮脉冲计数并清零
int  encoder_read_right(void);    // 读右轮脉冲计数并清零

#endif
