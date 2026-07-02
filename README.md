# zyf-WAN-Lora-Loader

ComfyUI 自定义节点，专为 **WAN 2.2** 设计的高噪/低噪成对 LoRA 加载器。**单个节点**即可完成双模型输入、多组 LoRA 配对加载、双模型输出。

灵感来自 [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) 的 **Power Lora Loader**。

---

## 功能特性
<img width="586" height="407" alt="image" src="https://github.com/user-attachments/assets/486f432b-56c5-4f3e-acf0-171538719b3e" />

### 核心功能

- **单节点双模型** — 上游同时传入 `high_noise` 模型和 `low_noise` 模型，下游直接输出加载完 LoRA 的两套模型，无需复制节点。
- **成对 LoRA 自动配对** — 每个高噪/低噪 LoRA 对存放在同一个子文件夹中。选择其中一个，自动从同文件夹获取另一个。无需依赖文件名中的 `_high`/`_low` 标记。因为很多lora名称并不规范!
- **批量添加** — 点击 **+ Add LoRA** 按钮添加多组 LoRA，每组独立设置高噪/低噪权重。
- **单文件兼容** — 如果子文件夹只有一个模型，只加载该模型，另一侧显示 `(no high-noise pair)` 或 `(no low-noise pair)` 提示。

### 双独立选择框

| 框 | 位置 | 颜色 | 功能 |
|---|---|---|---|
| 高噪（H） | 上 | 紫色背景/紫字 | 选择高噪 LoRA → 自动获取同文件夹低噪 |
| 低噪（L） | 下 | 蓝色背景/蓝字 | 选择低噪 LoRA → 自动获取同文件夹高噪 |

两个框紧密排列，无分隔线。显示格式为 `父文件夹名/文件名`，例如 `26增强走路姿势/Wan2.2-i2v_Normal-H.safetensors`。

### 权重调节

- **[−] [+] 按钮** — 每次增减 0.05
- **点击数值** — 弹出 ComfyUI 原生输入框，精确输入任意数值
- **拖拽调节** — 鼠标左键按住权重数值框左右拖动，实时调节（无上下限）
- **显示精度** — 保留两位小数

### 开关控制

每行前端有一个 **toggle 开关**：
- **开启（绿色）** — 该组 LoRA 正常加载，名称和权重正常显示
- **关闭（红色）** — 整组 LoRA 被忽略，名称和权重全部变为灰色

### 右键菜单

在任意 LoRA 行上**右键点击**弹出菜单：

| 选项 | 说明 |
|---|---|
| 📄 View TXT | 打开同文件夹下的 TXT 文件查看器（无 TXT 时灰色不可用） |
| ⬆️ Move Up | 将该行上移一行（顺序持久化保存） |
| ⬇️ Move Down | 将该行下移一行（顺序持久化保存） |
| 🗑️ Remove | 删除该行，其余行自动补齐 |

### TXT 文件查看器

- 居中靠上弹出的大窗口（85vh 高度）
- 显示 LoRA 同文件夹下的 `.txt` 文件内容
- 支持文本选择和复制
- 点击遮罩层或 Close 按钮关闭

---

## 输入 / 输出

### 输入（可选）

| 名称 | 类型 | 说明 |
|---|---|---|
| `model_high` | MODEL | 上游高噪扩散模型 |
| `model_low` | MODEL | 上游低噪扩散模型 |

### 输出

| 名称 | 类型 | 说明 |
|---|---|---|
| `HIGH_NOISE_MODEL` | MODEL | 加载完所有高噪 LoRA 后的模型 |
| `LOW_NOISE_MODEL` | MODEL | 加载完所有低噪 LoRA 后的模型 |

---

## 使用步骤

1. 将 `zyf-WAN-Lora-Loader` 文件夹放入 `ComfyUI/custom_nodes/`
2. 重启 ComfyUI
3. 在节点菜单的 `zyf-WAN-Lora-Loader` 分类中找到 **Wan 2.2 LoRA Loader (zyf)**
4. 将上游的 `high_noise` 和 `low_noise` 模型分别连接到节点的两个输入端
5. 点击 **+ Add LoRA** 添加一行，自动弹出高噪 LoRA 选择菜单
6. 点击高噪框（紫色）或低噪框（蓝色）选择 LoRA 文件，同文件夹的另一文件自动配对
7. 通过 [−] [+] 按钮、拖拽或点击输入来调节权重
8. 通过开关控制是否启用某组 LoRA
9. 右键行可查看 TXT、移动顺序、删除行

---

## 文件结构

```
zyf-WAN-Lora-Loader/
├── __init__.py                  # 注册节点和 Web 目录
├── wan_lora_loader.py           # 后端节点逻辑 + API 路由
├── web/
│   └── zyf_wan_lora_loader.js   # 前端自定义 UI
└── README.md
```

## API 路由

| 路由 | 方法 | 说明 |
|---|---|---|
| `/zyf_wan_lora/lora-tree` | GET | 返回 LoRA 文件夹树结构 |
| `/zyf_wan_lora/find-pair` | GET | 根据选中文件查找同文件夹的配对文件 |
| `/zyf_wan_lora/find-txt` | GET | 查找同文件夹下的 TXT 文件并返回内容 |

---

## 技术说明

- 前端使用 LiteGraph 自定义 widget 体系，每个 LoRA 行为独立 `WanLoraRowWidget` 实例
- 右键菜单通过重写 `getSlotInPosition` / `getSlotMenuOptions` 方法融入 LiteGraph 事件流
- 行顺序通过 `value.index` 字段持久化，移动后自动重新索引
- 后端 `load_loras` 按 index 顺序依次应用 LoRA，关闭的组自动跳过
- 无额外 Python 依赖，仅使用 ComfyUI 内置模块

---

## 测试环境

- ComfyUI（Windows / Python 3.12 / PyTorch 2.9.1 + CUDA 13.0）
