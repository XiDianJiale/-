1.GPIO

使用前需要进行初始化配置
通过 STM32 的 HAL 库进行设置 GPIO

命名方式：  GPIOA (PA0…PA14)    GPIOB   GPOIC …
代码中对应指定  “GPIOx； GPIO_Pin_0”

1输出：

HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET); // PA5 高电平

HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET); // PA5 低电平

2.输入：

GPIO_PinState state = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0); // 读取 PA0 状态 

if (state == GPIO_PIN_SET) { 

    // 开关被按下 

}

**输入输出的八种模式：**
将GPI0分为八种模式（**输入4种+输出4种）。八种模式分别为：

输入浮空 GPIO_Mode_IN_FLOATING

输入上拉 GPIO_Mode_IPU

输入下拉 GPIO_Mode_IPD

模拟输入 GPIO_Mode_AIN

；
输出：

具有上拉或下拉功能的**开漏输出** GPIO_Mode_Out_OD   在led实验中只支持低电平去驱动

具有上拉或下拉功能的**推挽输出 **GPIO_Mode_Out_PP   有高/低电平驱动led的功能

具有上拉或下拉功能的复用**功能推挽 **GPIO_Mode_AF_PP   

具有上拉或下拉功能的复用**功能开漏** GPIO_Mode_AF_OD
