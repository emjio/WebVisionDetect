# 🎯 目标检测Web应用

<p align="center">
  <img src="https://img.shields.io/badge/ONNX-Runtime-blue?style=flat-square&logo=onnx" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/JavaScript-ES6+-yellow?style=flat-square&logo=javascript" alt="JavaScript">
  <img src="https://img.shields.io/badge/HTML5-Canvas-orange?style=flat-square&logo=html5" alt="HTML5 Canvas">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

<p align="center">
  <b>基于YOLOv8 ONNX模型的实时目标检测系统</b><br>
  支持图片上传检测和摄像头实时检测，可识别手机、书本、键盘、鼠标四类物品
</p>

---

## ✨ 功能特性

| 功能 | 描述 |
|------|------|
| 📸 **图片上传检测** | 支持上传本地图片进行目标检测，快速获取检测结果 |
| 🎥 **实时摄像头检测** | 调用设备摄像头进行实时检测，支持自动连续检测 |
| 🎚️ **动态置信度调整** | 通过滑块实时调整检测阈值，适应不同场景需求 |
| 📱 **多类别检测** | 支持检测手机、书本、键盘、鼠标四类常见物品 |
| 🎨 **现代化UI** | 美观的用户界面，响应式设计，支持移动端访问 |
| ⚡ **高性能推理** | 使用ONNXRuntime Web在浏览器端直接运行模型推理 |
| 🔒 **隐私保护** | 所有检测在本地完成，图片数据不会上传到服务器 |

---

## 🚀 快速开始

### 方式一：使用Node.js（推荐）

```bash
# 克隆项目
git clone <your-repo-url>
cd object-dectet

# 安装依赖（可选，项目已配置CDN）
npm install

# 启动开发服务器
npm run dev
```

### 方式二：使用Python

```bash
# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000
```

### 方式三：使用VS Code Live Server

安装 [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) 插件，右键点击 `index.html` 选择 "Open with Live Server"。

---

## 📖 使用指南

### 图片上传检测

1. 打开应用后，选择 **"上传图片"** 模式
2. 点击 **"📁 选择图片"** 按钮，选择要检测的图片（支持 JPG、PNG 格式）
3. 点击 **"开始检测"** 按钮
4. 等待检测完成，结果会显示在画布上：
   - 🟢 绿色框：检测到的目标位置
   - 标签：显示类别名称和置信度
5. 右侧面板显示详细的检测信息

### 摄像头实时检测

1. 选择 **"实时摄像头"** 模式
2. 点击 **"启动摄像头"** 按钮
3. 首次使用需要授权摄像头权限，请点击"允许"
4. 启用 **"自动检测"** 选项，系统会自动进行连续检测
5. 检测结果会实时显示在视频画面上
6. 点击 **"停止摄像头"** 按钮结束检测

> ⚠️ **注意**：摄像头功能需要在 HTTPS 环境或 localhost 下运行

---

## 🛠️ 技术栈

- **[ONNXRuntime Web](https://onnxruntime.ai/docs/get-started/with-javascript.html)** - 在浏览器中高效运行ONNX模型
- **[HTML5 Canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)** - 图像处理和检测结果绘制
- **[WebRTC](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)** - 访问设备摄像头
---

## 📁 项目结构

```
object-dectet/
├── 📄 index.html          # 主页面 - 用户界面
├── 📄 main.js             # 核心逻辑 - 模型加载、推理、后处理
├── 📄 style.css           # 样式文件 - 响应式布局
├── 📄 package.json        # 项目配置
├── 📄 README.md           # 说明文档
└── 🧠 best.onnx           # YOLOv8 ONNX模型文件（需自行放置）
```

---

## 🧠 模型说明

### 支持的检测类别

| 类别ID | 类别名称 | 图标 |
|--------|----------|------|
| 0 | book（书本） | 📚 |
| 1 | cell phone（手机） | 📱 |
| 2 | keyboard（键盘） | ⌨️ |
| 3 | mouse（鼠标） | 🖱️ |

### 模型要求

- **格式**：ONNX (`.onnx`)
- **输入尺寸**：640×640 像素
- **输入格式**：RGB，归一化到 [0, 1]
- **输出格式**：`[1, 8, 8400]` - 包含边界框坐标和4个类别分数

### 使用自定义模型

如果你想使用自己的YOLOv8模型：

1. 将模型导出为ONNX格式：
   ```bash
   yolo export model=your_model.pt format=onnx imgsz=640
   ```

2. 将导出的 `.onnx` 文件重命名为 `best.onnx`

3. 替换项目根目录下的 `best.onnx` 文件

4. 如果类别不同，修改 `main.js` 中的 `CLASS_NAMES` 映射：
   ```javascript
   const CLASS_NAMES = {
       0: 'your_class_0',
       1: 'your_class_1',
       // ...
   };
   ```

---

## ⚙️ 高级配置

### 调整检测参数

在 `main.js` 中可以修改以下参数：

```javascript
// 置信度阈值（0-1之间）
let confidenceThreshold = 0.3;

// NMS（非极大值抑制）阈值
const nmsThreshold = 0.6;

// 最小检测框尺寸（像素）
const minBoxSize = 20;
```

### 性能优化建议

1. **降低检测频率**：在摄像头模式下，可以通过调整 `detectionInterval` 参数降低检测频率
2. **使用WebGL加速**：ONNXRuntime Web 默认使用 WebGL 后端进行加速
3. **缩小输入图片**：过大的图片会增加预处理时间，建议将图片尺寸控制在合理范围内

---

## 🔧 故障排除

### 常见问题

#### ❌ 模型加载失败

**症状**：页面显示"模型加载失败"

**解决方案**：
- ✅ 确保 `best.onnx` 文件存在于项目根目录
- ✅ 检查浏览器控制台（F12）查看详细错误信息
- ✅ 确保使用 HTTP 服务器运行，而不是直接打开 HTML 文件
- ✅ 检查模型文件是否损坏，尝试重新下载

#### ❌ 摄像头无法启动

**症状**：点击"启动摄像头"无反应或报错

**解决方案**：
- ✅ 检查浏览器权限设置，确保已授权摄像头访问
- ✅ 确保使用 HTTPS 协议或 localhost 环境
- ✅ 检查是否有其他应用占用了摄像头
- ✅ 尝试刷新页面后重新授权

#### ❌ 检测结果不准确

**症状**：检测框位置偏移或漏检

**解决方案**：
- ✅ 调整置信度阈值滑块，尝试不同的阈值
- ✅ 确保输入图片清晰，目标物体可见
- ✅ 检查模型是否针对当前检测任务训练
- ✅ 查看浏览器控制台，检查坐标转换是否正确

#### ❌ 检测速度很慢

**症状**：检测耗时过长，卡顿明显

**解决方案**：
- ✅ 关闭浏览器的其他标签页，释放内存
- ✅ 降低摄像头分辨率
- ✅ 使用性能更好的设备
- ✅ 确保浏览器支持 WebGL

---

## 🌐 浏览器兼容性

| 浏览器 | 版本要求 | 支持状态 |
|--------|----------|----------|
| Chrome | 80+ | ✅ 完全支持 |
| Edge | 80+ | ✅ 完全支持 |
| Firefox | 75+ | ✅ 完全支持 |
| Safari | 14+ | ⚠️ 部分支持（摄像头功能受限） |
| 移动端浏览器 | - | ⚠️ 性能可能受限 |

---



## 📄 许可证

本项目基于 [MIT License](./LICENSE) 开源。



## 🙏 致谢

- [ONNXRuntime](https://onnxruntime.ai/) - 提供高效的模型推理引擎
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 优秀的目标检测模型
- [Ultralytics](https://ultralytics.com/) - YOLO模型的开发和维护团队

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/emjio">emjio</a>
</p>
