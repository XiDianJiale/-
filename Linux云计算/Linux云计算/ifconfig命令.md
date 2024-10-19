windows下的 ipconfig

作用 ：  查看网络接口信息，  配置网络接口


ifconfig --help

2.修改，设置网卡
ifconfig  ens33 down 停止网卡
ifconfig  ens33 up 启用网卡

3.修改IP地址
ifconfig ens33:0 192.168.178.111 netmask 255.255.255.0 up

4.修改机器MAC地址信息
ifconfig ens33 hw ether 00:0c:29:13:10:CF

5.永久修改网络设备信息（ifconfig——>临时修改）
vim ifcfg-ens33 查看