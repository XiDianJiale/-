main.c部分

```markdown
# 代码知识点分解 - main.c

## 文件头信息

```c
/**
 ****************************************************************************************************
 * @file        main.c
 * @author      ÕýµãÔ­×ÓÍÅ¶Ó(ALIENTEK)
 * @version     V1.0
 * @date        2020-04-25
 * @brief       ´¥ÃþÆÁ ÊµÑé
 * @license     Copyright (c) 2020-2032, ¹ãÖÝÊÐÐÇÒíµç×Ó¿Æ¼¼ÓÐÏÞ¹«Ë¾
 ****************************************************************************************************
 */
```


- 作者信息、版本、日期及许可证信息。

## 引入头文件

```c
#include "./SYSTEM/sys/sys.h"
#include "./SYSTEM/usart/usart.h"
#include "./SYSTEM/delay/delay.h"
#include "./USMART/usmart.h"
#include "./BSP/LED/led.h"
#include "./BSP/LCD/lcd.h"
#include "./BSP/KEY/key.h"
#include "./BSP/TOUCH/touch.h"
```

### 描述
- 引入系统和外设相关的头文件，如系统管理、串口通信、延时、LED、LCD、按键和触摸屏。

## 1.函数：`load_draw_dialog`


### 描述
- 清屏并在LCD上显示“RST”字样。

## 2.函数：`lcd_draw_bline`

```c
void lcd_draw_bline(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint8_t size, uint16_t color)
{
    // 省略部分代码
}
```

### 描述
- 在LCD上绘制一条粗线，参数包括起点坐标、终点坐标、线宽和颜色。
- 利用Bresenham算法绘制带宽的线条。

## 3.函数：`rtp_test`

```c
void rtp_test(void)
{
    // 省略部分代码
}
```

### 描述
- 执行触摸屏测试程序。
- 处理触摸事件，绘制点或调用`load_draw_dialog()`函数。

## 4.常量定义：`POINT_COLOR_TBL`

```c
const uint16_t POINT_COLOR_TBL[10] = {RED, GREEN, BLUE, BROWN, YELLOW, MAGENTA, CYAN, LIGHTBLUE, BRRED, GRAY};
```

### 描述
- 定义了一组颜色，用于触摸点的绘制。

## 5.函数：`ctp_test`

```c
void ctp_test(void)
{
    // 省略部分代码
}
```

### 6.描述
- 完成触摸点的绘制，处理多个触摸点的逻辑。
- 使用`lcd_draw_bline()` 绘制已被触摸的路径。

## 7.主函数：`main`

```c
int main(void)
{
    HAL_Init();
    sys_stm32_clock_init(RCC_PLL_MUL9);
    delay_init(72);
    usart_init(115200);
    led_init();
    lcd_init();
    key_init();
    tp_dev.init();

    // 省略部分代码

    if (tp_dev.touchtype & 0X80)
    {
        ctp_test(); // µçÈÝÆÁ²âÊÔ
    }
    else
    {
        rtp_test(); // µç×èÆÁ²âÊÔ
    }
}
```


- 初始化硬件相关模块，并根据触摸屏类型选择执行 `ctp_test()` 或 `rtp_test()`。

