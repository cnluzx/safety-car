#include "encoder.h"
#include "esp_log.h"

static const char *ENC_TAG = "ENCODER";

// 两个 PCNT 单元句柄，分别对应左右编码器
static pcnt_unit_handle_t g_unit_left  = NULL;
static pcnt_unit_handle_t g_unit_right = NULL;

/*
 * 初始化一个正交编码器（×4 解码）
 *
 * 原理：
 *   通道0 监视 A 相边沿，以 B 相电平决定方向
 *   通道1 监视 B 相边沿，以 A 相电平决定方向（交叉反相）
 *   两个通道同时往同一个 UNIT 计数器加减 → ×4 分辨率
 */
static void pcnt_quadrature_init(pcnt_unit_handle_t *unit_out, int pin_a, int pin_b)
{
    // ----- 1. 创建 PCNT 单元 -----
    pcnt_unit_config_t unit_cfg = {
        .low_limit  = -32767,
        .high_limit =  32767,
    };
    ESP_ERROR_CHECK(pcnt_new_unit(&unit_cfg, unit_out));

    // ----- 2. 通道0：边沿 = A 相，电平控制 = B 相 -----
    pcnt_chan_config_t ch0_cfg = {
        .edge_gpio_num  = pin_a,
        .level_gpio_num = pin_b,
    };
    pcnt_channel_handle_t ch0 = NULL;
    ESP_ERROR_CHECK(pcnt_new_channel(*unit_out, &ch0_cfg, &ch0));

    // 边沿动作：上升沿+1，下降沿-1
    ESP_ERROR_CHECK(pcnt_channel_set_edge_action(ch0,
        PCNT_CHANNEL_EDGE_ACTION_INCREASE,    // pos edge
        PCNT_CHANNEL_EDGE_ACTION_DECREASE));  // neg edge

    // 电平动作：B 高电平时反转，B 低电平时保持
    // API: pcnt_channel_set_level_action(chan, high_act, low_act)
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(ch0,
        PCNT_CHANNEL_LEVEL_ACTION_INVERSE,     // high level: B=1 → 反转
        PCNT_CHANNEL_LEVEL_ACTION_KEEP));      // low  level: B=0 → 保持

    // ----- 3. 通道1：边沿 = B 相，电平控制 = A 相（交叉反相）-----
    pcnt_chan_config_t ch1_cfg = {
        .edge_gpio_num  = pin_b,
        .level_gpio_num = pin_a,
    };
    pcnt_channel_handle_t ch1 = NULL;
    ESP_ERROR_CHECK(pcnt_new_channel(*unit_out, &ch1_cfg, &ch1));

    // 边沿动作：上升沿+1，下降沿-1
    ESP_ERROR_CHECK(pcnt_channel_set_edge_action(ch1,
        PCNT_CHANNEL_EDGE_ACTION_INCREASE,
        PCNT_CHANNEL_EDGE_ACTION_DECREASE));

    // 电平动作：A 高电平时保持，A 低电平时反转（与通道0交叉）
    // API: pcnt_channel_set_level_action(chan, high_act, low_act)
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(ch1,
        PCNT_CHANNEL_LEVEL_ACTION_KEEP,        // high level: A=1 → 保持
        PCNT_CHANNEL_LEVEL_ACTION_INVERSE));   // low  level: A=0 → 反转

    // ----- 4. 加毛刺滤波（10μs）-----
    pcnt_glitch_filter_config_t filter_cfg = {
        .max_glitch_ns = 10000,
    };
    ESP_ERROR_CHECK(pcnt_unit_set_glitch_filter(*unit_out, &filter_cfg));

    // ----- 5. 清零并启动 -----
    ESP_ERROR_CHECK(pcnt_unit_clear_count(*unit_out));
    ESP_ERROR_CHECK(pcnt_unit_enable(*unit_out));
    ESP_ERROR_CHECK(pcnt_unit_start(*unit_out));
}

void encoder_init(void)
{
    ESP_LOGI(ENC_TAG, "Left  encoder: A=%d B=%d", ENC_LEFT_A, ENC_LEFT_B);
    pcnt_quadrature_init(&g_unit_left, ENC_LEFT_A, ENC_LEFT_B);

    ESP_LOGI(ENC_TAG, "Right encoder: A=%d B=%d", ENC_RIGHT_A, ENC_RIGHT_B);
    pcnt_quadrature_init(&g_unit_right, ENC_RIGHT_A, ENC_RIGHT_B);

    ESP_LOGI(ENC_TAG, "Encoders ready (x4 quadrature mode).");
}

// ---- 读取计数并清零（与 STM32 Read_Encoder 行为一致）----
static int read_and_clear(pcnt_unit_handle_t unit)
{
    int count = 0;
    if (unit) {
        pcnt_unit_get_count(unit, &count);
        pcnt_unit_clear_count(unit);
    }
    return count;
}

int encoder_read_left(void)
{
    return read_and_clear(g_unit_left);
}

int encoder_read_right(void)
{
    return read_and_clear(g_unit_right);
}
