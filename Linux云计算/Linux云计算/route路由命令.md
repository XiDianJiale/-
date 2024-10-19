路由： 源主机到目标主机之间的 转发过程
计算机之间的数据传输必须经过网络  直接连接两台计算机/通过一个个节点连接



分类:   静态路由  和  动态路由
（Linux机器配置都是静态路由，由运维人员通过route命令管理）


Destination 表示网络号   network的意思
Gateway :  表示网关地址
Genmask: 子网掩码地址的 表示
Flags ： 路由标记 ， 标记当前的网络状态

route add default gw 192.168.178.2

