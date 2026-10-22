#include "sdkconfig.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/uart.h"
#include "driver/pulse_cnt.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "encoder.h"

static const char *TAG = "ESP32_CHASSIS";
static volatile bool g_inspection_active = false; // 视觉触发侦查期间锁定电机
static volatile bool g_link_lost = false;      // 失联标志：上位机超时未下发指令
static volatile int  g_enc_left  = 0;         // 左轮编码器计数（每周期清零）
static volatile int  g_enc_right = 0;
static volatile bool g_motion_cancel_requested = false;
static volatile bool g_motion_active = false; // 长动作执行期间暂停失联超时
static volatile int64_t g_last_cmd_us = 0;     // 最近一次有效指令的时间戳
static portMUX_TYPE g_encoder_mux = portMUX_INITIALIZER_UNLOCKED;
static portMUX_TYPE g_speed_mux = portMUX_INITIALIZER_UNLOCKED;
static SemaphoreHandle_t g_uart_mutex;

typedef struct {
    bool enabled;
    float target_left;
    float target_right;
    float integral_left;
    float integral_right;
    float filtered_left;
    float filtered_right;
} SpeedController;

static SpeedController g_speed = {0};

// ===== Telemetry =====
/* Keep periodic E: encoder telemetry off during normal operation. */
static bool g_telemetry_enabled = false;
static int  g_telemetry_interval_ms = 100;
static int  g_telemetry_counter = 0;
static int64_t g_enc_left_total = 0;

typedef struct {
    int code;
    int v1;
    int v2;
    int v3;
} chassis_command_t;

static QueueHandle_t g_command_queue;

static int64_t g_enc_right_total = 0;         // 右轮编码器计数（每周期清零）

// ================= 引脚与串口配置 =================
#define UART_NUM         UART_NUM_0  // USB直连使用UART0
#define BUF_SIZE         1024

static void uart_send(const char *data, size_t len) {
    if (g_uart_mutex != NULL && xSemaphoreTake(g_uart_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        uart_write_bytes(UART_NUM, data, len);
        xSemaphoreGive(g_uart_mutex);
    }
}

// 向上位机回送动作完成应答
static void uart_send_ack(void) {
    uart_send("@\n", 2);
}

static void encoder_totals_snapshot(int64_t *left, int64_t *right) {
    portENTER_CRITICAL(&g_encoder_mux);
    *left = g_enc_left_total;
    *right = g_enc_right_total;
    portEXIT_CRITICAL(&g_encoder_mux);
}

// 电机 A (左轮)
#define M1_PWM_PIN       1   // 接 ENA 或 PWMA
#define M1_IN1_PIN       2   // 接 AIN1
#define M1_IN2_PIN       42  // 接 AIN2

// 电机 B (右轮)
#define M2_PWM_PIN       4   // 接 ENB 或 PWMB
#define M2_IN1_PIN       5   // 接 BIN1
#define M2_IN2_PIN       6   // 接 BIN2

// 电机方向修正
#define MOTOR_LEFT_REVERSE   1
#define MOTOR_RIGHT_REVERSE  0

// 双轴云台：GPIO8 水平扫描，GPIO3 控制高位摄像头俯仰。
#define SERVO_PIN              8
#define DETECT_SERVO_PIN       3
#define SERVO_HOME_ANGLE       105
#define DETECT_SERVO_NORMAL    0
#define DETECT_SERVO_CULVERT 90

// 编码器标定参数
#define ENC_FWD_TO_CENTER    1180    
#define ENC_TURN_PER_V2        13    
#define ENC_CROSS_PER_V2      1180    
#define TURN_SPEED_NORMAL        8
#define TURN_SPEED_PURE          8

// ── PID 巡线参数 ── (修正后移除了不合理的过零清空逻辑)
#define PID_KP              0.12f   
#define PID_KI              0.004f  
#define PID_KD              0.25f   
#define PID_INTEGRAL_MAX    30.0f   
#define PID_OUTPUT_MAX      50.0f   

// ── 左右轮速度 PI 参数 ──
// 目标速度单位为编码器脉冲/10ms；指令 8 的基础速度保持原取值范围。
// SPEED_TARGET_SCALE 是首先需要根据实车编码器数据标定的参数。
#define SPEED_TARGET_SCALE       1.0f
#define SPEED_TARGET_MAX        45.0f
#define SPEED_KP                 0.65f
#define SPEED_KI                 0.04f
#define SPEED_INTEGRAL_MAX     300.0f
#define SPEED_FILTER_ALPHA       0.35f
#define SPEED_FEEDFORWARD        1.0f
#define SPEED_MIN_PWM           11.0f

// ── 失联保护 ──
// 上位机超过 LINK_TIMEOUT_MS 未下发任何有效指令即视为失联，立即停车并锁定电机。
#define LINK_TIMEOUT_MS      1500
#define LINK_CHECK_PERIOD_MS  100

// PWM 设定
#define LEDC_MODE               LEDC_LOW_SPEED_MODE
#define LEDC_DUTY_RES           LEDC_TIMER_13_BIT 
#define LEDC_FREQUENCY_MOTOR    500              
#define LEDC_FREQUENCY_SERVO    50                

// ================= 硬件控制基础函数 =================

void init_hw() {
    // 电机定时器
    ledc_timer_config_t motor_timer = {
        .speed_mode = LEDC_MODE, .timer_num = LEDC_TIMER_0,
        .duty_resolution = LEDC_DUTY_RES, .freq_hz = LEDC_FREQUENCY_MOTOR, .clk_cfg = LEDC_AUTO_CLK
    };
    ledc_timer_config(&motor_timer);

    // 舵机定时器
    ledc_timer_config_t servo_timer = {
        .speed_mode = LEDC_MODE, .timer_num = LEDC_TIMER_1,
        .duty_resolution = LEDC_DUTY_RES, .freq_hz = LEDC_FREQUENCY_SERVO, .clk_cfg = LEDC_AUTO_CLK
    };
    ledc_timer_config(&servo_timer);

    // 通道配置
    ledc_channel_config_t m1 = { .channel=LEDC_CHANNEL_0, .duty=0, .gpio_num=M1_PWM_PIN, .speed_mode=LEDC_MODE, .timer_sel=LEDC_TIMER_0 };
    ledc_channel_config_t m2 = { .channel=LEDC_CHANNEL_1, .duty=0, .gpio_num=M2_PWM_PIN, .speed_mode=LEDC_MODE, .timer_sel=LEDC_TIMER_0 };
    ledc_channel_config_t sv = { .channel=LEDC_CHANNEL_2, .duty=0, .gpio_num=SERVO_PIN, .speed_mode=LEDC_MODE, .timer_sel=LEDC_TIMER_1 };
    ledc_channel_config_t detect_sv = { .channel=LEDC_CHANNEL_3, .duty=0, .gpio_num=DETECT_SERVO_PIN, .speed_mode=LEDC_MODE, .timer_sel=LEDC_TIMER_1 };
    ledc_channel_config(&m1);
    ledc_channel_config(&m2);
    ledc_channel_config(&sv);
    ledc_channel_config(&detect_sv);

    gpio_reset_pin(M1_IN1_PIN); gpio_set_direction(M1_IN1_PIN, GPIO_MODE_OUTPUT);
    gpio_reset_pin(M1_IN2_PIN); gpio_set_direction(M1_IN2_PIN, GPIO_MODE_OUTPUT);
    gpio_reset_pin(M2_IN1_PIN); gpio_set_direction(M2_IN1_PIN, GPIO_MODE_OUTPUT);
    gpio_reset_pin(M2_IN2_PIN); gpio_set_direction(M2_IN2_PIN, GPIO_MODE_OUTPUT);

}

// ── 【物理层驱动】单个电机：方向引脚 + PWM 占空比 ──
static void motor_drive(int pwm, int in1_pin, int in2_pin,
                        int reverse, ledc_channel_t channel) {
    if (pwm >  100) pwm =  100;
    if (pwm < -100) pwm = -100;

    if (pwm == 0) {                       // 双低电平 = 滑行停止
        gpio_set_level(in1_pin, 0);
        gpio_set_level(in2_pin, 0);
    } else {
        int forward = (pwm > 0) ? 1 : 0;
        gpio_set_level(in1_pin, forward ^ reverse);
        gpio_set_level(in2_pin, !forward ^ reverse);
    }

    ledc_set_duty(LEDC_MODE, channel, (abs(pwm) * 8191) / 100);
    ledc_update_duty(LEDC_MODE, channel);
}

void set_motor_pwm_direct(int m1_pwm, int m2_pwm) {
    motor_drive(m1_pwm, M1_IN1_PIN, M1_IN2_PIN, MOTOR_LEFT_REVERSE,  LEDC_CHANNEL_0);
    motor_drive(m2_pwm, M2_IN1_PIN, M2_IN2_PIN, MOTOR_RIGHT_REVERSE, LEDC_CHANNEL_1);
}

static void speed_control_disable(void) {
    portENTER_CRITICAL(&g_speed_mux);
    g_speed.enabled = false;
    g_speed.target_left = 0.0f;
    g_speed.target_right = 0.0f;
    g_speed.integral_left = 0.0f;
    g_speed.integral_right = 0.0f;
    g_speed.filtered_left = 0.0f;
    g_speed.filtered_right = 0.0f;
    portEXIT_CRITICAL(&g_speed_mux);
}

// 视觉侦查或失联期间只允许停车指令通过
static inline bool motion_locked(void) {
    return g_inspection_active || g_link_lost;
}

// ── 电机控制：受安全锁保护的 PWM 输出 ──
void set_motor(int m1_speed, int m2_speed) {
    /* 所有传统运动指令保持 PWM 直控，并阻止速度任务覆盖其输出。 */
    speed_control_disable();
    if (motion_locked() && (m1_speed != 0 || m2_speed != 0)) {
        return;
    }
    set_motor_pwm_direct(m1_speed, m2_speed);
}

void set_servo(int angle) {
    if (angle < 0) angle = 0;
    if (angle > 180) angle = 180;
    int duty = 204 + (angle * (1024 - 204) / 180);
    ledc_set_duty(LEDC_MODE, LEDC_CHANNEL_2, duty);
    ledc_update_duty(LEDC_MODE, LEDC_CHANNEL_2);
}

/* 原始脉宽标定接口：LEDC 50Hz 周期为20ms，单位为微秒。 */
static void set_servo_pulse_us(int pulse_us) {
    if (pulse_us < 400) pulse_us = 400;
    if (pulse_us > 2600) pulse_us = 2600;
    int duty = (pulse_us * 8192 + 10000) / 20000;
    ledc_set_duty(LEDC_MODE, LEDC_CHANNEL_2, duty);
    ledc_update_duty(LEDC_MODE, LEDC_CHANNEL_2);
}

static void set_detection_servo(int angle) {
    if (angle < 0) angle = 0;
    if (angle > 180) angle = 180;
    int duty = 204 + (angle * (1024 - 204) / 180);
    ledc_set_duty(LEDC_MODE, LEDC_CHANNEL_3, duty);
    ledc_update_duty(LEDC_MODE, LEDC_CHANNEL_3);
}

static void servo_slow_scan(int start_angle, int end_angle, int step, int delay_ms) {
    int dir = (end_angle > start_angle) ? 1 : -1;
    for (int angle = start_angle;
         (dir > 0) ? (angle <= end_angle) : (angle >= end_angle);
         angle += dir * step) {
        set_servo(angle);
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
    }
    set_servo(end_angle);
}

static void detection_servo_slow_move(int start_angle, int end_angle,
                                      int step, int delay_ms) {
    int dir = (end_angle > start_angle) ? 1 : -1;
    for (int angle = start_angle;
         (dir > 0) ? (angle <= end_angle) : (angle >= end_angle);
         angle += dir * step) {
        set_detection_servo(angle);
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
    }
    set_detection_servo(end_angle);
}

// 上位机视觉识别到涵洞后，通过指令 10 触发侦查与上报流程。
static void run_culvert_inspection(void) {
    g_inspection_active = true;
    set_motor(0, 0);
    ESP_LOGI(TAG, "Vision inspection started.");

    uart_send("Y_START\n", 8);

    /* 把高位摄像头从涵洞观察姿态抬到嫌疑人识别姿态。缓动可降低
     * 两只舵机同时突变造成的电源压降。 */
    detection_servo_slow_move(DETECT_SERVO_NORMAL,
                              DETECT_SERVO_CULVERT, 2, 10);
    vTaskDelay(pdMS_TO_TICKS(150));

    servo_slow_scan(SERVO_HOME_ANGLE, 180, 2, 6);
    vTaskDelay(pdMS_TO_TICKS(500));
    uart_send("Y_L\n", 4);
    vTaskDelay(pdMS_TO_TICKS(2000));

    servo_slow_scan(165, 25, 2, 6);
    vTaskDelay(pdMS_TO_TICKS(500));
    uart_send("Y_R\n", 4);
    vTaskDelay(pdMS_TO_TICKS(2000));

    servo_slow_scan(0, SERVO_HOME_ANGLE, 2, 6);
    detection_servo_slow_move(DETECT_SERVO_CULVERT,
                              DETECT_SERVO_NORMAL, 2, 10);
    uart_send("Y_END\n", 6);

    g_inspection_active = false;
    ESP_LOGI(TAG, "Vision inspection finished.");
}

// ================= PID 巡线控制器 =================
typedef struct {
    float kp, ki, kd;          
    float setpoint;            
    float integral;            
    float prev_error;          
    float out_min, out_max;    
    float integral_max;        
    bool  fresh;               
} PIDController;

static PIDController g_pid;   

static void pid_init(void) {
    g_pid.kp = PID_KP;
    g_pid.ki = PID_KI;
    g_pid.kd = PID_KD;
    g_pid.setpoint = 0.0f;
    g_pid.integral = 0.0f;
    g_pid.prev_error = 0.0f;
    g_pid.out_min = -PID_OUTPUT_MAX;
    g_pid.out_max =  PID_OUTPUT_MAX;
    g_pid.integral_max = PID_INTEGRAL_MAX;
    g_pid.fresh = true;
}

static void pid_reset(void) {
    g_pid.integral = 0.0f;
    g_pid.prev_error = 0.0f;
    g_pid.fresh = true;
}

static float pid_compute(float offset) {
    float error = offset;

    if (g_pid.fresh) {
        g_pid.prev_error = error;
        g_pid.integral = 0.0f;
        g_pid.fresh = false;
        float out = g_pid.kp * error;
        if      (out > g_pid.out_max) out = g_pid.out_max;
        else if (out < g_pid.out_min) out = g_pid.out_min;
        return out;
    }

    float p_term = g_pid.kp * error;

    g_pid.integral += (error + g_pid.prev_error) * 0.5f;
    if (g_pid.integral >  g_pid.integral_max) g_pid.integral =  g_pid.integral_max;
    if (g_pid.integral < -g_pid.integral_max) g_pid.integral = -g_pid.integral_max;
    float i_term = g_pid.ki * g_pid.integral;

    float d_term = g_pid.kd * (error - g_pid.prev_error);
    g_pid.prev_error = error;

    float output = p_term + i_term + d_term;
    if      (output > g_pid.out_max) output = g_pid.out_max;
    else if (output < g_pid.out_min) output = g_pid.out_min;

    return output;
}

static float clamp_float(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static void speed_control_set_targets(float left, float right) {
    left = clamp_float(left * SPEED_TARGET_SCALE, -SPEED_TARGET_MAX, SPEED_TARGET_MAX);
    right = clamp_float(right * SPEED_TARGET_SCALE, -SPEED_TARGET_MAX, SPEED_TARGET_MAX);

    if (left == 0.0f && right == 0.0f) {
        speed_control_disable();
        set_motor_pwm_direct(0, 0);
        return;
    }

    portENTER_CRITICAL(&g_speed_mux);
    if (!g_speed.enabled) {
        g_speed.integral_left = 0.0f;
        g_speed.integral_right = 0.0f;
        g_speed.filtered_left = 0.0f;
        g_speed.filtered_right = 0.0f;
    }
    g_speed.target_left = left;
    g_speed.target_right = right;
    g_speed.enabled = true;
    portEXIT_CRITICAL(&g_speed_mux);
}

static int speed_pi_step(float target, int encoder_delta,
                         float *filtered, float *integral) {
    float target_magnitude = (target >= 0.0f) ? target : -target;
    if (target_magnitude < 0.01f) {
        *filtered = 0.0f;
        *integral = 0.0f;
        return 0;
    }

    float measured = (float)abs(encoder_delta);
    *filtered += SPEED_FILTER_ALPHA * (measured - *filtered);
    float error = target_magnitude - *filtered;

    *integral = clamp_float(*integral + error,
                            -SPEED_INTEGRAL_MAX, SPEED_INTEGRAL_MAX);

    float pwm = SPEED_FEEDFORWARD * target_magnitude
              + SPEED_KP * error
              + SPEED_KI * (*integral);
    pwm = clamp_float(pwm, 0.0f, 100.0f);
    if (pwm > 0.0f && pwm < SPEED_MIN_PWM) pwm = SPEED_MIN_PWM;

    int signed_pwm = (int)(pwm + 0.5f);
    return (target >= 0.0f) ? signed_pwm : -signed_pwm;
}

static void speed_control_update(int delta_l, int delta_r) {
    int pwm_l = 0;
    int pwm_r = 0;
    bool enabled;

    portENTER_CRITICAL(&g_speed_mux);
    enabled = g_speed.enabled;
    if (enabled) {
        pwm_l = speed_pi_step(g_speed.target_left, delta_l,
                              &g_speed.filtered_left, &g_speed.integral_left);
        pwm_r = speed_pi_step(g_speed.target_right, delta_r,
                              &g_speed.filtered_right, &g_speed.integral_right);
    }
    portEXIT_CRITICAL(&g_speed_mux);

    if (!enabled) return;
    if (motion_locked() || g_motion_cancel_requested) {
        speed_control_disable();
        set_motor_pwm_direct(0, 0);
        return;
    }
    set_motor_pwm_direct(pwm_l, pwm_r);
}

// ── 编码器读取任务（100 Hz）──
void encoder_task(void *pvParameters) {
    while (1) {
        int delta_l = encoder_read_left();
        int delta_r = encoder_read_right();
        g_enc_left  = delta_l;
        g_enc_right = delta_r;
        speed_control_update(delta_l, delta_r);
        portENTER_CRITICAL(&g_encoder_mux);
        g_enc_left_total  += delta_l;
        g_enc_right_total += delta_r;
        portEXIT_CRITICAL(&g_encoder_mux);

        if (g_telemetry_enabled) {
            g_telemetry_counter++;
            int ticks = g_telemetry_interval_ms / 10;
            if (ticks < 1) ticks = 1;
            if (g_telemetry_counter % ticks == 0) {
                char buf[48];
                int64_t total_l, total_r;
                encoder_totals_snapshot(&total_l, &total_r);
                int n = snprintf(buf, sizeof(buf), "E:%lld,%lld\n",
                             (long long)total_l, (long long)total_r);
                uart_send(buf, n);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ================= 编码器闭环运动函数 =================

#define MOTION_TIMEOUT_TICKS  700   // 10ms/tick → 7 秒上限

/* 累加编码器脉冲直到达到目标。turning 为真时左右轮脉冲相加（原地转向两轮
 * 反向等量贡献），否则取平均（直行两轮同向）。任一安全标志置位即提前退出。 */
static bool wait_for_pulses_mode(int target_pulses, bool turning, const char *what,
                                 bool stop_on_success) {
    int ticks = 0;
    int64_t start_l = 0;
    int64_t start_r = 0;
    encoder_totals_snapshot(&start_l, &start_r);

    while (!motion_locked() && !g_motion_cancel_requested &&
           ticks < MOTION_TIMEOUT_TICKS) {
        vTaskDelay(pdMS_TO_TICKS(10));
        if (motion_locked() || g_motion_cancel_requested) break;
        int64_t now_l = 0;
        int64_t now_r = 0;
        encoder_totals_snapshot(&now_l, &now_r);
        int64_t delta_l = llabs(now_l - start_l);
        int64_t delta_r = llabs(now_r - start_r);
        int64_t progress = turning
            ? (delta_l + delta_r)
            : ((delta_l + delta_r) / 2);
        if (progress >= target_pulses) {
            if (stop_on_success) set_motor(0, 0);
            return true;
        }
        ticks++;
    }
    if (ticks >= MOTION_TIMEOUT_TICKS) {
        ESP_LOGW(TAG, "%s timeout!", what);
    }
    set_motor(0, 0);
    return false;
}

static bool wait_for_pulses(int target_pulses, bool turning, const char *what) {
    return wait_for_pulses_mode(target_pulses, turning, what, true);
}

/* Keep the PI targets active through the junction ACK and line-follow handoff.
 * Cancellation, link safety and timeout still stop the motors. */
static bool drive_cross_continuous(int speed, int target_pulses) {
    if (speed <= 0 || target_pulses <= 0) {
        set_motor(0, 0);
        return false;
    }
    speed_control_set_targets((float)speed, (float)speed);
    return wait_for_pulses_mode(target_pulses, false, "drive_cross_continuous", false);
}

static bool drive_straight_enc(int speed, int target_pulses) {
    target_pulses = abs(target_pulses);
    if (speed == 0 || target_pulses == 0) return false;
    /* Fixed-distance motions use the same left/right speed PI as line following. */
    speed_control_set_targets((float)speed, (float)speed);
    return wait_for_pulses(target_pulses, false, "drive_straight_enc");
}

static bool turn_in_place_enc(int dir, int target_pulses, int speed) {
    target_pulses = abs(target_pulses);
    speed = abs(speed);
    if (dir == 0 || target_pulses == 0 || speed == 0) return false;
    int signed_speed = (dir > 0) ? speed : -speed;
    /* In-place turns also use PI targets with opposite wheel directions. */
    speed_control_set_targets((float)signed_speed, (float)-signed_speed);
    return wait_for_pulses(target_pulses, true, "turn_in_place_enc");
}

static void uart_send_motion_result(bool completed) {
    if (completed) uart_send_ack();
    else uart_send("ERR:MOTION\n", 11);
}

// ================= 协议解析与业务逻辑 =================

void parse_and_execute(int code, int v1, int v2, int v3) {
    g_motion_cancel_requested = false;
    switch (code) {
        case 1:  
        case 11: 
            if (code == 11 && v1 == 15) { 
                set_motor(v1, v1);
                if (v2 > 0) vTaskDelay(pdMS_TO_TICKS(v2 * 10)); 
                set_motor(0, 0);
            } else {
                set_motor(v1, v1); 
            }
            break;

        case 6:  // 纯距离编码器闭环直行
        case 16:
            uart_send_motion_result(drive_straight_enc(v1, v1 * v2));
            break;

        case 2:  // 路口对齐后转向：先前冲对齐中心，再原地转向
            {
            bool completed = drive_straight_enc(10, ENC_FWD_TO_CENTER);
            if (completed) {
                vTaskDelay(pdMS_TO_TICKS(50));
                completed = turn_in_place_enc(
                    v2, abs(v2) * ENC_TURN_PER_V2, TURN_SPEED_NORMAL
                );
            }
            pid_reset();
            uart_send_motion_result(completed);
            }
            break;

        case 3:  // 纯原地转向（无前冲，适合出发区等任意位置掉头）
            {
            bool completed = turn_in_place_enc(
                v2, abs(v2) * ENC_TURN_PER_V2, TURN_SPEED_PURE
            );
            pid_reset();
            uart_send_motion_result(completed);
            }
            break;

        case 4:
            {
            bool completed = turn_in_place_enc(
                15, abs(15) * ENC_TURN_PER_V2, TURN_SPEED_PURE
            );
            if (completed) completed = drive_straight_enc(-11, 25);
            if (completed) completed = turn_in_place_enc(
                26, abs(26) * ENC_TURN_PER_V2, TURN_SPEED_PURE
            );
            if (completed) completed = drive_straight_enc(-11, 25);
            if (completed) completed = turn_in_place_enc(
                30, abs(30) * ENC_TURN_PER_V2, TURN_SPEED_PURE
            );
            if (completed) completed = drive_straight_enc(-11, 25);
            if (completed) completed = turn_in_place_enc(
                90, abs(90) * ENC_TURN_PER_V2, TURN_SPEED_PURE
            );
            if (completed) completed = drive_straight_enc(-11, 40);
            pid_reset();
            set_motor(0, 0);
            uart_send_motion_result(completed);
            }
            break;

        case 7:  // 连续前进通过路口，完成 ACK 后由巡线接管，不清零速度
            {
            bool completed = drive_cross_continuous(v1, v2 * ENC_CROSS_PER_V2);
            pid_reset();
            uart_send_motion_result(completed);
            }
            break;

        case 8:  // PID 巡线差速
            {
                if (v1 == 0) {
                    pid_reset();
                    set_motor(0, 0);
                    break;
                }
                float correction = pid_compute((float)v2);
                speed_control_set_targets((float)v1 + correction,
                                          (float)v1 - correction);
            }
            break;

        case 9:  // 舵机角度
            set_servo(v1);
            break;

        case 18: // 高位摄像头俯仰舵机角度（独立云台测试/标定）
            set_detection_servo(v1);
            uart_send_ack();
            break;

        case 19: // 水平舵机原始脉宽，v1=微秒，用于端点标定
            set_servo_pulse_us(v1);
            uart_send_ack();
            break;

        case 10: // 上位机视觉识别到涵洞，触发侦查扫描与 Y_* 上报
            run_culvert_inspection();
            break;

        case 5:  // 急停
            pid_reset();
            set_motor(0, 0);
            uart_send_ack();
            break;

        case 14:  // 请求单次编码器累计值
            {
                char buf[48];
                int64_t total_l, total_r;
                encoder_totals_snapshot(&total_l, &total_r);
                int n = snprintf(buf, sizeof(buf), "E:%lld,%lld\n",
                                 (long long)total_l, (long long)total_r);
                uart_send(buf, n);
                uart_send_ack();
            }
            break;

        case 15:  // 遥测使能：v1=1 开启，v1=0 关闭
            g_telemetry_enabled = (v1 != 0);
            g_telemetry_counter = 0;
            uart_send_ack();
            break;

        case 17:  // 设置遥测间隔（ms，20-1000）
            if (v1 >= 20 && v1 <= 1000) {
                g_telemetry_interval_ms = v1;
            }
            uart_send_ack();
            break;
    }
}

// ================= 失联保护任务 =================
// 上位机若在 LINK_TIMEOUT_MS 内没有下发任何有效指令，判定为失联：停车、清空
// 指令队列并置位 g_link_lost 锁住电机。收到下一条有效指令后自动解锁。
static void link_watchdog_task(void *pvParameters) {
    while (1) {
        int64_t idle_ms = (esp_timer_get_time() - g_last_cmd_us) / 1000;
        bool lost = (idle_ms > LINK_TIMEOUT_MS) && !g_motion_active;

        if (lost && !g_link_lost) {
            g_link_lost = true;
            g_motion_cancel_requested = true;
            xQueueReset(g_command_queue);
            pid_reset();
            speed_control_disable();
            set_motor_pwm_direct(0, 0);   // 绕过安全锁，确保停车一定生效
            ESP_LOGW(TAG, "Link lost (%lld ms idle), motors locked.",
                     (long long)idle_ms);
        } else if (!lost && g_link_lost) {
            g_link_lost = false;
            ESP_LOGI(TAG, "Link restored.");
        }
        vTaskDelay(pdMS_TO_TICKS(LINK_CHECK_PERIOD_MS));
    }
}

// ================= 串口监听任务 =================

static void motion_task(void *pvParameters) {
    chassis_command_t command;
    while (1) {
        if (xQueueReceive(g_command_queue, &command, portMAX_DELAY) == pdTRUE) {
            g_motion_active = true;
            parse_and_execute(command.code, command.v1, command.v2, command.v3);
            g_motion_active = false;
        }
    }
}

void uart_rx_task(void *pvParameters) {
    uint8_t* data = (uint8_t*) malloc(BUF_SIZE);
    configASSERT(data != NULL);
    char cmd_buf[64];
    int buf_idx = 0;

    while (1) {
        int len = uart_read_bytes(UART_NUM, data, BUF_SIZE, 10 / portTICK_PERIOD_MS);
        for (int i = 0; i < len; i++) {
            if (data[i] == '@') {
                buf_idx = 0;
            } else if (data[i] == '#') {
                cmd_buf[buf_idx] = '\0';
                int c = 0, v1 = 0, v2 = 0, v3 = 0;
                if (sscanf(cmd_buf, "|%d|%d|%d|%d", &c, &v1, &v2, &v3) >= 2) {
                    g_last_cmd_us = esp_timer_get_time();  // 喂狗
                    /* 收到有效命令时立即解锁，避免命令先于看门狗任务执行。 */
                    g_link_lost = false;
                    if (c == 5) {
                        /* Emergency stop must never wait behind a long motion. */
                        g_motion_cancel_requested = true;
                        xQueueReset(g_command_queue);
                        pid_reset();
                        set_motor(0, 0);
                        uart_send("@\n", 2);
                    } else {
                        chassis_command_t command = { c, v1, v2, v3 };
                        if (xQueueSend(g_command_queue, &command, 0) != pdTRUE) {
                            ESP_LOGW(TAG, "Command queue full; dropping code %d", c);
                        }
                    }
                }
                buf_idx = 0;
            } else {
                if (buf_idx < (int)sizeof(cmd_buf) - 1) cmd_buf[buf_idx++] = data[i];
            }
        }
    }
}

void app_main(void) {
    init_hw();
    
    uart_config_t uart_cfg = {
        .baud_rate = 115200, .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE, .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE, .source_clk = UART_SCLK_DEFAULT
    };
    uart_param_config(UART_NUM, &uart_cfg);
    uart_driver_install(UART_NUM, BUF_SIZE * 2, 0, 0, NULL, 0);

    encoder_init();
    pid_init();
    set_servo(SERVO_HOME_ANGLE);  // 舵机初始位置
    set_detection_servo(DETECT_SERVO_NORMAL);  // 高位摄像头常态保持 0°

    g_uart_mutex = xSemaphoreCreateMutex();
    g_command_queue = xQueueCreate(12, sizeof(chassis_command_t));
    configASSERT(g_uart_mutex != NULL);
    configASSERT(g_command_queue != NULL);

    xTaskCreate(uart_rx_task, "uart_rx_task", 4096, NULL, 10, NULL);
    xTaskCreate(motion_task, "motion_task", 3072, NULL, 8, NULL);
    xTaskCreate(encoder_task, "encoder_task", 4096, NULL, 6, NULL);

    /* 先喂一次狗，避免上电后尚未收到指令就立刻判定失联。 */
    g_last_cmd_us = esp_timer_get_time();
    xTaskCreate(link_watchdog_task, "link_watchdog", 2560, NULL, 9, NULL);

    ESP_LOGI(TAG, "Chassis System Online.");
}
