# HeLo：从个人电脑迁移到实验室 Windows RTX 4090

## 2026-11-03 更新：当前采用 GitHub 同步代码

实验室已通过 ToDesk 连接，项目实际目录为 `D:\helo\HeLo`，解释器为 `C:\conda_envs\helo4090\python.exe`。Python 3.10.22、PyTorch 2.5.1+cu121、NumPy 1.26.4、SciPy 1.14.1、scikit-learn 1.4.2 已通过依赖检查；两张 RTX 4090 均已通过基础前向、反向及 Adam 更新检查。完整 HeLo smoke test 尚待同步代码与数据后执行。

用户回传的实验室仓库为 `reproduction` 分支、提交 `9d453d7`，缺少后续模型修复和复现入口。代码通过用户的 GitHub 仓库同步：先检查 `git status --short --branch`，确认本地修改后再用 `git pull --ff-only origin reproduction`。不要通过重装依赖解决缺失脚本，也不要用旧迁移包覆盖新的 Git 工作区。

数据与实验结果仅在本地和实验室设备之间传输，不提交到 GitHub。现有 ZIP 是旧工作区快照：需要取数据时，先核对整包 SHA-256，再解压到独立暂存目录并在暂存目录运行 `verify_transfer.py`，随后仅将校验后的 `data/DMER_raw_preserved` 复制到项目的新数据目录。若目标数据目录已存在，先比较与核验，不直接覆盖。

`verify_transfer.py` 用于带有 `TRANSFER_MANIFEST.json` 的迁移包目录；普通 Git 克隆不包含该清单，不能直接用它验证 Git 工作区。使用 Git 提交号确认代码来源，使用迁移包校验和确认数据来源。

下文保留 2026-10-30 的历史迁移流程。尚未执行的命令必须将示例项目路径 `D:\Research\HeLo` 改为 `D:\helo\HeLo`，将环境路径 `D:\conda_envs\helo4090` 改为 `C:\conda_envs\helo4090`。环境创建和安装步骤已经完成，无需重复。SSH 仍未配置；模型检查、训练、恢复和持续运行按验收顺序继续。

### 已确认的解压目录：直接生成训练标签

实验室现有 73 份原始特征位于 `D:\DATA\数据集\DMER\03_特征_Features\Features`，73 份问卷位于 `D:\DATA\数据集\DMER\01_原始数据_Raw\Raw_data\被试编号\被试编号\Panas.csv`。特征只有 `feas` 和 `vids`，需生成 `dis_label` 后再运行模型。无需重新打包或解压视频。

更新后的准备脚本同时支持原有 `--source` 压缩包模式，以及成对使用的 `--features-dir` / `--panas-dir` 解压目录模式。两种方式共用视频ID匹配、评分检查、归一化和输出读回检查；目录方式只查找两层以内的问卷，不递归遍历视频树。示例（实验室 PowerShell）：

```powershell
Set-Location -LiteralPath 'D:\helo\HeLo'
$labPrepareArguments = @(
    '-u', 'prepare_dmer.py'
    '--features-dir', 'D:\DATA\数据集\DMER\03_特征_Features\Features'
    '--panas-dir', 'D:\DATA\数据集\DMER\01_原始数据_Raw\Raw_data'
    '--output', 'D:\helo\HeLo\data\DMER_raw_preserved'
    '--allow-out-of-range'
    '--subjects'
) + (1..73)
& 'C:\conda_envs\helo4090\python.exe' @labPrepareArguments
```

输出目录必须没有已有准备报告或同名被试MAT，脚本不会覆盖它们。原始输入只读；新的特征和标签保存在项目已被 Git 忽略的 `data/DMER_raw_preserved`。该输出也与训练入口和计划任务脚本的默认数据位置一致。

报告保留每份源文件的SHA-256，并增加 `source_format` 与 `source_content_sha256`。后者按被试编号排序，对编号、特征SHA-256、问卷SHA-256组成的JSON数组（紧凑分隔符，UTF-8）再计算SHA-256，便于跨路径和压缩方式核对源内容。目录模式下 `source` 记录两个输入目录，`archives` 为空；旧压缩包模式的 `source` 仍是源目录字符串。准备完成后须核对73人、54315个样本、源内容校验值以及异常值保留策略，再进行模型smoke test。

整理日期：2026-10-30。已知实验室是 Windows，目前尚未取得连接地址和账号。本指南给出操作流程，尚未在实验室实际连接或验证环境。

## 1. 先明确两台电脑分别做什么

你的笔记本负责远程操作和查看结果，4090电脑负责存放数据、建立Conda环境和运行训练。两台电脑的盘符、Python环境和文件路径彼此独立；不能在本机终端运行Python却期待调用远程4090。

建议路线：学校VPN/校园网 → Windows远程桌面 → 在实验室电脑配置并训练。需要长期编辑代码时，再使用 VS Code Remote-SSH。远程桌面和SSH是两种连接方式，不要求同时配置。

## 2. 现在先向实验室获取这些信息

可直接把下面的话发给管理电脑的同学（此处只是待发送文字，没有代你发送）：

> 我准备在实验室4090电脑上复现HeLo。请提供：电脑IP或主机名、Windows版本/版本类型、供我使用的登录账号、远程桌面是否已启用及端口、校外访问是否需要学校VPN，以及允许存放代码/数据和创建Conda环境的盘符。也请确认GPU的可用时间、系统内存和剩余磁盘空间。若已启用SSH，请同时提供SSH端口。

密码通过实验室认可的方式获取，不需要发到本聊天。Windows登录通常需要账号密码，不能把Windows Hello的PIN直接当作远程登录密码。

实验室同学可在那台电脑上检查：

```powershell
hostname
whoami
ipconfig
nvidia-smi
```

IP要选择实际接入校园网的网卡地址，不能随便取WSL、虚拟机或VPN虚拟网卡的地址。若没有固定地址，可请管理员提供稳定主机名或地址。

## 3. 首选：从你的笔记本打开远程桌面

### 实验室电脑上完成

1. 在“设置 → 系统 → 关于”确认版本类型。Windows家庭版不能作为内置远程桌面主机；专业版等支持远程桌面的版本才适用。若是家庭版，使用实验室已有且获准的远控方案，或配置后文SSH，不需要因此重装系统。
2. 在“设置 → 系统 → 远程桌面”启用远程桌面，把你的账号加入允许远程访问的用户。保留网络级别身份验证。
3. 管理员确认校园网/VPN到该主机的路由与防火墙规则，以及机器保持开机、训练期间不自动睡眠。Windows桌面系统的远程登录可能影响现场会话，提前协调使用时间。

### 你的笔记本上完成

1. 校外先连学校提供的VPN。仅有校园内网IP通常不能从公网直连；某些学校VPN也不开放实验室网段，需要管理员确认。不要自行把3389或22端口暴露到公网。
2. 按 `Win+R`，输入 `mstsc`。
3. 填写实验室主机地址；非默认远程桌面端口用 `地址:端口`，输入获准账号和密码。
4. 首次证书提示先与实验室确认目标主机，不要把未知主机当作实验室电脑。
5. 连上后，在远程桌面里打开PowerShell执行 `hostname` 和 `nvidia-smi`，确认看到实验室主机和RTX4090。

如连接失败，可在笔记本PowerShell执行（先把示例地址替换为实际值）：

```powershell
Test-NetConnection -ComputerName 实验室实际地址 -Port 3389
```

端口不通优先查VPN、路由、主机开机状态和防火墙；端口通而登录失败再查账号和远程登录权限。[微软远程桌面说明](https://support.microsoft.com/en-us/windows/experience/connectivity-networking/how-to-use-remote-desktop)

## 4. 可选：配置SSH与VS Code远程编辑

仅在你需要Remote-SSH，且实验室同意启用SSH时进行。实验室电脑的管理员PowerShell执行：

```powershell
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
# 仅当未安装时执行下一行
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
Get-NetFirewallRule -Name OpenSSH-Server-In-TCP -ErrorAction SilentlyContinue
```

OpenSSH安装通常会创建入站规则；缺失或被学校策略限制时由管理员处理，不要关闭整个防火墙。以上命令不是在你的笔记本安装SSH服务。[微软OpenSSH安装说明](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse)

在笔记本PowerShell测试（替换账号、地址、端口）：

```powershell
ssh -p 22 实际账号@实验室实际地址
```

首次连接时，和管理员核对服务器主机密钥指纹；先确认普通SSH能登录，再配置VS Code。在本机安装微软Remote-SSH扩展，按 `Ctrl+Shift+P` → `Remote-SSH: Connect to Host`，连接后远程平台选择Windows。

可在本机 `%USERPROFILE%\.ssh\config` 添加配置，下面的 `LAB_ADDRESS` 和 `LAB_USER` 必须替换：

```text
Host lab4090
    HostName LAB_ADDRESS
    User LAB_USER
    Port 22
    ServerAliveInterval 30
```

随后可执行 `ssh lab4090`。在VS Code远程窗口打开实验室的项目目录，并将Python解释器选为实验室的 `D:\conda_envs\helo4090\python.exe`；终端中用 `hostname` 再次确认执行位置。[VS Code官方Remote-SSH指南](https://code.visualstudio.com/docs/remote/ssh)

## 5. 迁移当前代码与已处理数据

推荐实验室目录（先确认D盘存在、有空间并允许写入；也可以统一换成E盘）：

```text
D:\Research\HeLo\                 代码
D:\Research\HeLo\data\DMER_raw_preserved\  73个被试的派生特征和标签
D:\Research\HeLo\runs\            权重、预测与指标
D:\conda_envs\helo4090\            新Conda环境
```

本轮提供 `HeLo_4090_transfer_20261030.zip`：含当前代码、测试、协议、73个处理后的MAT和准备报告，不含原始视频、旧Conda环境或Git凭据。文件清单与SHA-256记录在包内 `TRANSFER_MANIFEST.json`，源提交及未提交状态记录在 `SOURCE_PROVENANCE.json`。这是一份当前工作目录快照，不能仅用Git提交号代表包内全部代码。

传输方法任选一种：

- **远程桌面**：连接前在“显示选项 → 本地资源 → 详细信息 → 驱动器”勾选存放迁移包的本地盘；在远程“此电脑”中找到重定向驱动器，将ZIP复制到实验室D盘。也可按实验室许可使用U盘/共享目录。
- **已配置SSH**：在笔记本PowerShell执行下面命令，文件会传到实验室账号的用户目录。

```powershell
scp "E:\BitGraduate1\2026.9-10论文\output\transfer\HeLo_4090_transfer_20261030.zip" lab4090:HeLo_4090_transfer_20261030.zip
```

在实验室PowerShell中，将ZIP路径改成实际位置，解压到新的空项目目录：

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\HeLo_4090_transfer_20261030.zip" -DestinationPath 'D:\Research\HeLo'
```

包的根目录直接是项目文件，解压后应有 `D:\Research\HeLo\run_reproduction.py`，不要多套一层HeLo目录。不要覆盖已经运行中的实验目录。

目前本地新增脚本尚未全部提交或推送，因此仅执行 `git clone` 可能缺少迁移所需文件。后续可在确认提交并推送后改用Git同步代码；数据继续单独传输，不加入GitHub。已有数据已派生出913维特征，先做当前实验无需传输几十GB原始视频。

## 6. 在实验室新建Conda环境

不要直接复制笔记本的 `E:\conda_envs\helo` 文件夹。实验室若已有Conda，直接使用；没有则由你或管理员安装官方Miniconda，安装位置可选有权限的D盘目录。先打开**实验室电脑的Anaconda Prompt/Miniconda Prompt**：

```bat
conda --version
conda create --prefix D:\conda_envs\helo4090 python=3.10 pip -y
conda activate D:\conda_envs\helo4090
python -c "import sys; print(sys.executable)"
```

最后一行应输出 `D:\conda_envs\helo4090\python.exe`。按路径创建环境显示完整路径是正常的。[Conda环境管理](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)

本次建议先验证下列固定迁移组合，而不是追逐最新版本：Python3.10、PyTorch2.5.1 CUDA12.1、NumPy1.26.4、SciPy1.14.1、scikit-learn1.4.2。**这是4090迁移候选环境，尚未在实验室实测，也不同于作者的torch1.12.1+cu116。** Ada存在兼容较早二进制的条件，不能简单断言旧环境一定不能运行；这里选择另建环境进行验证并保留版本差异记录。[NVIDIA Ada兼容性说明](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html)

在已激活的实验室Conda提示符中执行：

```bat
cd /d D:\Research\HeLo
python -m pip install -r requirements-4090.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip check
```

PyTorch安装索引和版本来自[官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。本项目当前没有使用torchvision或torchaudio，无需一并安装。**不要再安装旧的 `requirements-repro.txt`，否则会把torch降回cu116版本。**

先执行 `nvidia-smi` 检查驱动；如果提示驱动过旧，交由管理员确认升级兼容驱动。正常使用预编译PyTorch进行本项目训练通常不需要另装完整CUDA Toolkit；`nvidia-smi`显示的CUDA版本是驱动支持能力，不等于当前Python所用的CUDA运行时。后者看下一步的 `torch.version.cuda`。驱动匹配参见[NVIDIA兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

## 7. 正式训练前验收

下面开始统一使用**实验室PowerShell**；显式指定环境Python，不依赖当前终端是否已经激活Conda：

```powershell
Set-Location 'D:\Research\HeLo'
$py = 'D:\conda_envs\helo4090\python.exe'
& $py -c "import sys,torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
& $py verify_transfer.py
& $py -m unittest discover -s tests -v
& $py smoke_test.py --data data/DMER_raw_preserved/1.mat
```

预期看到实验室环境路径、CUDA可用、RTX4090、文件校验通过、测试通过，以及真实数据前向/反向/参数更新和状态恢复通过。仅有 `import torch` 成功不够。若冒烟检查失败，先保留报错解决，暂不启动73人任务。

登记实验室环境：

```powershell
New-Item -ItemType Directory -Path environment_records -Force | Out-Null
& $py -m pip freeze | Out-File environment_records\pip-freeze-4090.txt -Encoding utf8
nvidia-smi | Out-File environment_records\nvidia-smi-4090.txt -Encoding utf8
```

## 8. 先1人300轮，再73人完整被试内实验

实验室PowerShell中先执行（仍使用上一步的 `$py`）：

```powershell
& $py -u run_reproduction.py --implementation code-repaired --subjects 1 --epochs 300 --batch-size 128 --output runs/4090_subject1
```

完成后看 `runs/4090_subject1/comparison.md` 和 `subject_1.json`。单人结果只用于验收，不能冒充论文73人平均值；4090的速度也以该次实测为准，不按显卡型号直接承诺加速倍数。

通过后，完整73人命令为：

```powershell
& $py -u run_reproduction.py --implementation code-repaired --epochs 300 --batch-size 128 --output runs/dmer_code_73_4090
```

保持论文的8:2随机样本划分、300轮、batch128、Adam学习率0.001、六项指标及逐被试等权平均；不要因为4090显存大就把batch增大，否则也会改变当前KL/CC相对权重。这里不调用原始 `train.py`/`test.py`，它们仍有已知评估及路径问题，新入口负责完整评估和记录。

**复现限制仍然存在**：`code-repaired`保留公开模型的大部分结构，修复了标签参数注册和评估随机性；公开GW/IPOT与正文Sinkhorn不一致。`paper-equations`属于有明确假设的公式解释诊断，不能视为已获作者确认的正确版本。36号评分13在全体参考实验中原样保留并标记，另报排除36号的均值。详细差异见 `REPRODUCTION_PROTOCOL.md`。更换GPU不会自动解决这些问题。

本入口目前支持DMER被试内实验，**不包含WESAD、留一被试73折、论文全部对照方法及全部消融**。这些需要另行准备协议/代码，不能把本入口运行结束称作整篇论文全部复现。

## 9. 长任务断开远程后如何继续

短测试可以直接在远程桌面终端运行。关闭远程桌面窗口通常只是断开会话，但注销账号、学校会话超时策略、睡眠或重启可能终止进程。不要依赖关闭SSH或VS Code后训练一定存活。

完整任务建议使用实验室Windows“任务计划程序”，并先做断开后继续运行的实测：

1. 选择“创建任务”，名称 `HeLo-DMER-4090`，使用获准的实验室账号。
2. 要求注销后也运行时，选择“不管用户是否登录都要运行”；按提示在实验室系统中输入该账号密码。管理员需允许该账号作为批处理任务登录。不要在脚本里保存明文密码。
3. “操作 → 启动程序”：程序为 `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`。
4. “添加参数”填写下面这一整行（若环境盘符不同同步替换）：

```text
-NoProfile -File "D:\Research\HeLo\start_lab_training.ps1" -PythonExe "D:\conda_envs\helo4090\python.exe" -RunName "dmer_code_73_4090"
```

5. “起始于”填写 `D:\Research\HeLo`，不加引号。无需为了训练默认勾选最高权限。
6. “设置”选择已有实例运行时“不启动新实例”；检查并取消不合适的最长运行时限；不要把任务配置成用户闲置才运行。关闭自动睡眠需先与机器使用者协调。
7. 手动“运行”任务，确认日志、`nvidia-smi`、`progress.json`都在更新；断开远程后等几分钟再连回核对。任务账户和交互桌面的环境不同，这一步必须实际测试，不能只看任务已创建。

若组织策略阻止PS脚本或计划任务，让管理员按实验室规范配置，不要全局放宽执行策略。计划任务账户需有项目读写权，数据使用D盘真实目录，避免仅在远程桌面会话中存在的重定向盘或映射盘。[微软任务安全上下文说明](https://learn.microsoft.com/en-us/windows/win32/taskschd/security-contexts-for-running-tasks)

提供的 `start_lab_training.ps1` 使用指定环境的Python、独立输出/错误日志；已有本次运行manifest时自动添加 `--resume`。它不自动修改驱动、注册计划任务或重启机器。不要同时在终端启动同一个输出目录的任务。

## 10. 查看进度、中断恢复和取回结果

实验室PowerShell：

```powershell
Get-Content runs/dmer_code_73_4090/progress.json
Get-Content runs/dmer_code_73_4090/summary.json
Get-ChildItem logs -Filter '*.log'
```

`progress.json`在保存断点时更新，`summary.json`在完成一名被试后才更新；暂时没有summary不一定代表失败。完整标准是summary的 `status` 为 `complete` 且 `completed_subjects` 为73人，再检查错误日志。查看某个日志的最新输出可以用 `Get-Content 实际日志路径 -Tail 30 -Wait`。

若任务中断，先确认旧进程已退出，再使用原命令加 `--resume`：

```powershell
& $py -u run_reproduction.py --implementation code-repaired --epochs 300 --batch-size 128 --output runs/dmer_code_73_4090 --resume
```

每25轮保存一次恢复状态。恢复要求代码、数据、目录和实验参数一致；不要把笔记本旧环境/旧路径的断点直接混入新的4090结果目录。当前恢复日志可能包含回滚后重复轮次，分析训练曲线时按轮次取最后一条；最终指标来自最后完整模型。

CPU内存不足和CUDA显存不足是两类问题。笔记本旧环境曾在保存断点时发生CPU分配失败，原因尚未确认，不能归因于4GB显存，也不能保证更换GPU就消失。实验室先测试保存和恢复；遇到错误保留完整日志并检查系统可用内存/提交内存。不要擅自关闭别人的进程或改系统分页文件。

需要取回的目录是 `runs/dmer_code_73_4090`，主要文件：

| 文件 | 用途 |
|---|---|
| comparison.md / comparison.csv | 与论文表1六项指标对照 |
| summary.json / subject_*.json | 总体、逐被试与简单基线指标 |
| predictions_*.npz / split_*.npz | 全部预测、标签及划分索引 |
| model_*.pt | 最终模型权重 |
| manifest.json / source_snapshot | 参数、版本、数据校验和和代码快照 |

可用远程桌面重定向盘复制回笔记本。若通过SSH，先在实验室把已完成的结果目录压缩到该账号用户目录，然后在笔记本取回：

```powershell
# 实验室：确认对应任务已完成后执行
Compress-Archive -Path 'D:\Research\HeLo\runs\dmer_code_73_4090' -DestinationPath "$env:USERPROFILE\HeLo_4090_results.zip"
```

```powershell
# 笔记本：先创建接收目录
New-Item -ItemType Directory -Path 'E:\BitGraduate1\2026.9-10论文\output\lab_results' -Force | Out-Null
scp lab4090:HeLo_4090_results.zip "E:\BitGraduate1\2026.9-10论文\output\lab_results\"
```

结果只在相同数据和协议范围内比较；报告中写明4090、Windows、PyTorch/CUDA版本及代码实现标签。第一阶段完成的是DMER被试内参考复现，不把相似度当准确率，也不把一个被试的结果当全体结论。
