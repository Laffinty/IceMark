# IceMark BETA

> 一个多线程圆周率基准测试程序，2005年左右的作品。

## 起源

这是作者学生时代（大约2009-2010年）写的一个玩具程序。
那时候刚接触多线程编程，想写一个基准测试工具来衡量 CPU 的并发性能。
代码很稚嫩，逻辑也很粗糙，但它确实能跑起来。

## 项目结构

```
MultiThread/
├── IceMark.cpp        # 主程序源代码
├── MultiThread.dsp    # VC++ 6.0 项目文件
├── MultiThread.dsw    # VC++ 6.0 工作区文件
├── Release/           # 编译产物（可执行文件）
└── Debug/             # 调试版本编译产物
```

## 算法说明

程序使用多线程并发计算圆周率，算法基于进制转换模拟长除法：
- Group A：4线程并行
- Group B：2线程并行
- Group C：单线程

最终得分 = x1×20 + x2×40 + x3×30

> 注：由于早期对多线程同步的理解不深，所有线程共用一个 Mutex，实际上是串行执行的。这反而成了它的一个"特色"。

## 编译

```bash
# 使用 Visual Studio
打开 MultiThread.dsw 或 MultiThread.sln

# 命令行编译
cl /EHsc IceMark.cpp /Fe:IceMark.exe
```

## 已知问题

- 全局 mutex 导致线程实际串行执行
- 内存泄漏（new 未 delete）
- `SetThreadPriority` 写法是声明而非调用
- 代码大量重复（7个函数几乎一模一样）

这些问题是有意保留的，作为童年回忆的一部分。

---

*"所有带走的未必留下，丢弃的也未必遗忘。"*