- **时钟信号（Clock Signals）**：
    
    - **使用场景**：为电路提供同步参考时钟，控制数据流的速率和时序。
    - **特点**：通常为周期性方波信号，频率和占空比根据设计要求设置。
- **数据信号（Data Signals）**：
    
    - **使用场景**：传输逻辑数据，在不同的模块之间交换信息。
    - **特点**：可以是单端信号或差分信号，单向或双向传输。
- **控制信号（Control Signals）**：
    
    - **使用场景**：控制电路的操作模式和行为，如使能信号（enable）、读写信号（read/write）、选择信号（select）等。
    - **特点**：通常为单端信号，用于启动或停止某个操作，选择不同的工作模式等。
- **使能信号（Enable Signals）**：
    
    - **使用场景**：控制特定模块或电路的激活状态。
    - **特点**：一般为单端信号，高电平或低电平有效。
- **状态信号（Status Signals）**：
    
    - **使用场景**：指示电路或模块的当前状态，如完成信号（done）、忙碌信号（busy）、错误信号（error）等。
    - **特点**：用于反馈电路运行状态，通常为单端信号。
- **中断信号（Interrupt Signals）**：
    
    - **使用场景**：通知处理器或控制单元某个事件发生，需要处理器介入。
    - **特点**：异步信号，高电平或低电平有效。
- **电源信号（Power Signals）**：
    
    - **使用场景**：为电路提供电源电压和接地。
    - **特点**：包括电源引脚（Vcc, Vdd）和接地引脚（GND）。
- **配置信号（Configuration Signals）**：
    
    - **使用场景**：用于配置FPGA的初始设置，如配置模式选择、配置数据输入等。
    - **特点**：通常在FPGA启动时使用，决定FPGA的工作模式和初始状态
- **数据信号（TMDS Data Signals）**：
    
    - **使用场景**：传输视频、音频和其他数据。通常有三个数据通道（tmds_d_p[0]、tmds_d_p[1]、tmds_d_p[2]）用于传输视频数据。
    - **特点**：每个通道都有正（P）和负（N）两个信号线，采用差分方式传输数据，以减少电磁干扰和噪声影响。
- **时钟信号（TMDS Clock Signal）**：
    
    - **使用场景**：提供同步时钟信号，确保数据接收端能够正确地采样和解释数据。通常对应信号为tmds_clk_p。
    - **特点**：与数据信号同步传输，用于锁定和解码接收到的数据。

#### 复位信号（Reset Signal）

- **复位信号（Reset Signal）**：
    - **使用场景**：在电路启动或重置时，初始化所有的寄存器和状态机到已知状态，确保系统处于可控的初始状态。常见信号名为resetn或reset。
    - **特点**：通常为单端信号，可以是高电平有效（active high）或低电平有效（active low）。上电时触发，或由外部信号触发。



[在verilog文件和物理约束文件中需要对信号的不同方面定义](信号在不同文件中的定义：.md)