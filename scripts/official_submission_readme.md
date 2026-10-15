# 赛题2 Windows EXE提交包

本包用于2026年10月15日发布的官方测试接口。无需安装Python、pip、CUDA或其他软件，不需要联网。支持Windows10/11 x64。

## 运行

将官方完整测试数据放在EXE同级，保持以下结构：

```text
测试目录/
├── EatingDetection.exe
├── ACC_test_data/
│   ├── userinfobean.txt
│   ├── HNU00001/
│   │   ├── collect_data0_....txt
│   │   └── collect_data1_....txt
│   └── HNU00002/
└── results.csv                 # 运行成功后生成，不是提交包预置数据
```

双击EatingDetection.exe，或在命令行直接运行它，不需要参数。程序总是以EXE所在目录定位ACC_test_data和results.csv，不依赖启动时的工作目录。每个人的全部TXT通过官方DataReader.iter_data读取，InfoReader读取官方人口信息，save_result保存预测。

输出只包含预测进食事件，CSV列名及顺序固定：

```csv
externalid,beforeTime,afterTime
```

编号对应个人目录并按官方规则大写/移除空白；时间为原始采集轴13位Unix毫秒整数，开始小于结束。没有进食预测也会产生表头。UTF-8 BOM便于Excel读取。

每次成功运行原子替换本轮全部结果，不追加旧结果。退出码0为成功；退出码2为失败。失败时不会发布半成品，但之前成功的results.csv可能仍在，请勿将旧结果当作本次结果。请在命令提示符中运行以查看错误。

可选依赖/模型自检（不读测试数据）：

```text
EatingDetection.exe --self-check
```

## 模型与依赖

EXE采用PyInstaller onefile自包含打包，内置CPython、科学计算库、LightGBM和部署模型。运行时在Windows临时目录释放内置DLL，结束后由bootloader清理；这不是所有库的传统静态链接，也不依赖机器预装Python。首次启动解包可能需要数秒，请勿连续重复启动。需有Windows临时目录及EXE目录的写入权限。

build_manifest.json记录EXE/包内文件SHA-256、正式模型文件哈希、源代码标识与依赖版本。official_api为原样官方接口来源，licenses为第三方许可证；不需要运行它们。

