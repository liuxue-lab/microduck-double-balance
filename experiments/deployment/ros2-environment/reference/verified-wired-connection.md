# Jetson Orin Nano 有线直连与 SSH 连接指南

## 1. 文档目的

本文记录已经实际连接成功并验收过的方案，用于在 Ubuntu 笔记本与 Jetson Orin Nano Super 之间建立稳定的有线直连。

日常使用时，网线负责 SSH、文件传输和设备管理；Wi-Fi 继续负责访问互联网。

## 2. 已验证的网络配置

| 设备 | 网卡或连接 | IPv4 地址 | 用途 |
|---|---|---:|---|
| Ubuntu 笔记本 | `enp4s0` / `jetson-direct` | `192.168.50.1/24` | 连接 Jetson |
| Jetson | `enP8p1s0` / `jetson-wired` | `192.168.50.2/24` | SSH 与文件传输 |
| Jetson SSH 用户 | `jetson` | `jetson@192.168.50.2` | 远程登录 |
| Jetson 主机名 | `yahboom` | 登录后显示 | 设备身份确认 |

网络拓扑：

```text
Ubuntu 笔记本
enp4s0 = 192.168.50.1/24
        │
        │ 以太网线直连
        │
Jetson Orin Nano Super
enP8p1s0 = 192.168.50.2/24
```

重要原则：

- 笔记本和 Jetson 的 Wi-Fi 负责互联网。
- `192.168.50.0/24` 有线网段只负责两台设备之间的通信。
- 有线连接不配置默认网关和 DNS。
- 不要把有线连接设置为默认互联网出口。

## 3. 每次开机后的日常连接流程

### 第 1 步：连接硬件

1. 给 Jetson 通电并等待系统启动。
2. 使用以太网线直连笔记本与 Jetson。
3. 笔记本保持正常连接 Wi-Fi。

### 第 2 步：启用笔记本有线连接

在 Ubuntu 笔记本终端执行：

```bash
sudo nmcli connection up "jetson-direct"
```

### 第 3 步：检查物理链路

```bash
cat /sys/class/net/enp4s0/carrier
```

正确输出：

```text
1
```

结果含义：

- `1`：网线物理链路已经建立。
- `0`：Jetson 未开机、网线未插好、网线故障或接口没有建立链路。

### 第 4 步：检查笔记本地址

```bash
ip -4 -br addr show enp4s0
```

正确结果应包含：

```text
192.168.50.1/24
```

### 第 5 步：测试 Jetson 是否可达

```bash
ping -c 3 192.168.50.2
```

成功时会收到来自 `192.168.50.2` 的回复，并且丢包率应为 `0%`。

### 第 6 步：通过 SSH 登录

```bash
ssh jetson@192.168.50.2
```

第一次连接时可能询问是否信任主机：

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

确认地址确实是自己的 Jetson 后输入：

```text
yes
```

随后输入 Jetson 用户 `jetson` 的密码。成功登录后的提示符类似：

```text
jetson@yahboom:~$
```

看到该提示符，即表示连接完成。

## 4. 最短日常命令

网络配置正确后，每次开机通常只需要执行：

```bash
sudo nmcli connection up "jetson-direct"
ping -c 3 192.168.50.2
ssh jetson@192.168.50.2
```

如果有线连接能够自动启用，可以直接执行：

```bash
ssh jetson@192.168.50.2
```

## 5. 修复笔记本端 `jetson-direct`

如果 `ip -4 -br addr show enp4s0` 显示以下地址：

```text
10.42.0.1/24
192.168.8.1/24
```

说明它不是已经验收成功的最终配置。将现有 `jetson-direct` 改回固定地址：

```bash
sudo nmcli connection modify "jetson-direct" \
  connection.interface-name enp4s0 \
  ipv4.method manual \
  ipv4.addresses "192.168.50.1/24" \
  ipv4.gateway "" \
  ipv4.dns "" \
  ipv4.never-default yes \
  ipv6.method disabled
```

重新激活连接：

```bash
sudo nmcli connection down "jetson-direct"
sudo nmcli connection up "jetson-direct"
```

重新验收：

```bash
ip -4 -br addr show enp4s0
ping -c 3 192.168.50.2
```

通过条件：

```text
LaptopAddress=192.168.50.1/24
JetsonPing=PASS
```

## 6. `jetson-direct` 不存在时的创建方法

先检查连接名称：

```bash
nmcli connection show
```

如果列表中没有 `jetson-direct`，执行：

```bash
sudo nmcli connection add \
  type ethernet \
  ifname enp4s0 \
  con-name "jetson-direct" \
  ipv4.method manual \
  ipv4.addresses "192.168.50.1/24" \
  ipv4.never-default yes \
  ipv6.method disabled
```

然后启用并检查：

```bash
sudo nmcli connection up "jetson-direct"
ip -4 -br addr show enp4s0
```

## 7. 修复 Jetson 端 `jetson-wired`

只有在笔记本地址正确，但 `192.168.50.2` 始终无法访问，并且确认物理链路正常时，才检查 Jetson 端。

需要暂时给 Jetson 接显示器和键盘，在 Jetson 本机终端执行：

```bash
nmcli device status
nmcli connection show
```

此前已经验证过的 Jetson 有线接口和连接名称分别为：

```text
Interface=enP8p1s0
Connection=jetson-wired
```

恢复固定地址：

```bash
sudo nmcli connection modify "jetson-wired" \
  ipv4.method manual \
  ipv4.addresses "192.168.50.2/24" \
  ipv4.gateway "" \
  ipv4.dns "" \
  ipv4.never-default yes
```

启用并检查：

```bash
sudo nmcli connection up "jetson-wired"
ip -4 -br addr show enP8p1s0
ip route
```

正确结果应包含：

```text
192.168.50.2/24
```

路由表中不应出现通过 `enP8p1s0` 建立的默认互联网路由。Jetson 的默认路由应继续由 Wi-Fi 承担。

## 8. 常见故障判断

| 现象 | 说明 | 处理方法 |
|---|---|---|
| `carrier` 输出 `0` | 物理链路未建立 | 检查 Jetson 电源、网线和两端网口 |
| `enp4s0` 显示 `DOWN` 或 `NO-CARRIER` | 笔记本没有检测到网线链路 | 确认 Jetson 已启动并重新插拔网线 |
| 笔记本地址是 `10.42.0.1` | NetworkManager 使用了共享网络模式 | 将 `jetson-direct` 改为 `192.168.50.1/24` 手动模式 |
| 笔记本地址是 `192.168.8.1` | 使用了此前未成功的探测网段 | 恢复为 `192.168.50.1/24` |
| `ping 192.168.50.2` 无回复 | Jetson 地址未启用或物理链路异常 | 先检查 `carrier`，再检查 Jetson 的 `jetson-wired` |
| `Connection refused` | 网络已通，但 Jetson SSH 服务未运行 | 在 Jetson 本机检查 SSH 服务 |
| `Permission denied` | 用户名或密码错误 | 使用用户 `jetson` 并输入 Jetson 登录密码 |
| `No route to host` | 本机地址、接口或路由不正确 | 检查 `enp4s0` 是否为 `192.168.50.1/24` |

Jetson SSH 服务检查命令：

```bash
sudo systemctl status ssh --no-pager
```

如果服务没有运行：

```bash
sudo systemctl enable --now ssh
```

## 9. 已确认不能使用的地址

此前曾测试亚博默认地址：

```text
192.168.8.88
```

实际结果为：

```text
Destination Host Unreachable
192.168.8.88 INCOMPLETE
```

这说明电脑在该网段内无法通过 ARP 找到 Jetson。因此：

- 不要使用 `ssh jetson@192.168.8.88`。
- 不要把 `192.168.8.88` 当作当前 Jetson 地址。
- 不要依赖异常或未解析成功的 `jetson.local`。
- 当前经过验证的 SSH 地址是 `192.168.50.2`。

## 10. 文件传输

### 从笔记本发送文件到 Jetson

```bash
scp 本地文件 jetson@192.168.50.2:/home/jetson/
```

示例：

```bash
scp test.py jetson@192.168.50.2:/home/jetson/
```

### 从 Jetson 下载文件到笔记本当前目录

```bash
scp jetson@192.168.50.2:/home/jetson/文件名 .
```

## 11. 退出 SSH

在 Jetson 远程终端执行：

```bash
exit
```

返回笔记本终端后，SSH 会话即结束。

## 12. 最终验收清单

依次执行：

```bash
cat /sys/class/net/enp4s0/carrier
ip -4 -br addr show enp4s0
ping -c 3 192.168.50.2
ssh jetson@192.168.50.2
```

全部满足以下条件即为连接成功：

- `carrier` 输出 `1`。
- 笔记本有线地址为 `192.168.50.1/24`。
- `192.168.50.2` 可以正常 Ping 通。
- SSH 用户为 `jetson`。
- 登录后的主机名为 `yahboom`。
- 笔记本和 Jetson 的互联网默认路由仍然使用 Wi-Fi。

## 13. 一句话记忆

```text
插网线并启动 Jetson → 笔记本 192.168.50.1 → Jetson 192.168.50.2 → ssh jetson@192.168.50.2
```
