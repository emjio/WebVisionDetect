// ONNX模型和检测相关变量
let session = null;
let modelLoaded = false;
let isProcessing = false;
let cameraStream = null;
let cameraInterval = null;
let autoDetect = true;

// 置信度阈值（可动态调整）
let confidenceThreshold = 0.3; // 默认0.3，与Python代码一致

// YOLOv8 类别名称映射（根据数据集配置）
// 数据集类别顺序：
//   0: book
//   1: cell phone
//   2: keyboard
//   3: mouse
const CLASS_NAMES = {
    0: 'book',        // 书本
    1: 'cell phone',  // 手机
    2: 'keyboard',    // 键盘
    3: 'mouse'        // 鼠标
};

// 目标类别过滤（可选，与Python代码中的TARGET_CLASSES对应）
// 设置为空Set()，显示所有检测到的类别（不过滤）
const TARGET_CLASSES = new Set(); // 空Set表示不过滤任何类别

// DOM元素
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const fileInput = document.getElementById('file-input');
const detectBtn = document.getElementById('detect-btn');
const modeSelect = document.getElementById('mode-select');
const uploadControls = document.getElementById('upload-controls');
const cameraControls = document.getElementById('camera-controls');
const startCameraBtn = document.getElementById('start-camera-btn');
const stopCameraBtn = document.getElementById('stop-camera-btn');
const autoDetectCheckbox = document.getElementById('auto-detect');
const loading = document.getElementById('loading');
const status = document.getElementById('status');
const detectionInfo = document.getElementById('detection-info');
const modelStatus = document.getElementById('model-status');
const modelDetails = document.getElementById('model-details');
const debugOutput = document.getElementById('debug-output');
const toggleDebugBtn = document.getElementById('toggle-debug');
const confidenceSlider = document.getElementById('confidence-slider');
const confidenceValue = document.getElementById('confidence-value');

// 初始化
async function init() {
    await loadModel();
    setupEventListeners();
    updateStatus('模型已加载，可以开始检测');
}

// 加载ONNX模型
async function loadModel() {
    try {
        showLoading(true);
        updateModelStatus('正在加载模型...');
        
        // 加载ONNX模型
        session = await ort.InferenceSession.create('best.onnx', {
            executionProviders: ['webgl', 'wasm'],
            graphOptimizationLevel: 'all'
        });
        
        modelLoaded = true;
        
        // 打印模型信息用于调试
        console.log('模型加载成功');
        console.log('Session对象:', session);
        console.log('输入名称:', session.inputNames);
        console.log('输出名称:', session.outputNames);
        console.log('输入元数据:', session.inputMetadata);
        console.log('输出元数据:', session.outputMetadata);
        
        // 安全地获取模型信息
        let inputInfo = [];
        let outputInfo = [];
        let modelInputSize = 640; // 默认值
        let inputShape = [1, 3, 640, 640]; // 默认值
        let outputShape = null;
        
        try {
            if (session.inputNames && session.inputNames.length > 0) {
                inputInfo = session.inputNames.map(name => {
                    const metadata = session.inputMetadata && session.inputMetadata[name];
                    return {
                        name,
                        shape: metadata?.shape || '未知',
                        type: metadata?.type || '未知'
                    };
                });
                
                // 尝试获取输入形状
                const firstInputName = session.inputNames[0];
                const firstInputMetadata = session.inputMetadata && session.inputMetadata[firstInputName];
                if (firstInputMetadata && firstInputMetadata.shape) {
                    inputShape = firstInputMetadata.shape;
                    modelInputSize = inputShape[inputShape.length - 1] || 640;
                }
            }
            
            if (session.outputNames && session.outputNames.length > 0) {
                outputInfo = session.outputNames.map(name => {
                    const metadata = session.outputMetadata && session.outputMetadata[name];
                    return {
                        name,
                        shape: metadata?.shape || '未知',
                        type: metadata?.type || '未知'
                    };
                });
                
                // 尝试获取输出形状
                const firstOutputName = session.outputNames[0];
                const firstOutputMetadata = session.outputMetadata && session.outputMetadata[firstOutputName];
                if (firstOutputMetadata && firstOutputMetadata.shape) {
                    outputShape = firstOutputMetadata.shape;
                }
            }
        } catch (error) {
            console.warn('获取模型元数据时出错:', error);
        }
        
        console.log('输入信息:', inputInfo);
        console.log('输出信息:', outputInfo);
        
        // 显示模型详情
        const shapeStr = Array.isArray(inputShape) ? inputShape.join(' × ') : '未知';
        const outputShapeStr = outputShape ? outputShape.join(' × ') : '未知';
        
        modelDetails.innerHTML = `
            输入: ${shapeStr}<br>
            输出: ${outputShapeStr}
        `;
        
        addDebugInfo('模型加载', `输入: ${JSON.stringify(inputInfo)}, 输出: ${JSON.stringify(outputInfo)}`);
        
        updateModelStatus(`模型已加载 (输入尺寸: ${modelInputSize}x${modelInputSize})`);
        
        // 保存模型输入尺寸供后续使用
        window.modelInputSize = modelInputSize;
        
        showLoading(false);
    } catch (error) {
        console.error('模型加载失败:', error);
        updateModelStatus('模型加载失败: ' + error.message);
        showLoading(false);
        alert('模型加载失败，请确保best.onnx文件存在');
    }
}

// 设置事件监听器
function setupEventListeners() {
    // 模式切换
    modeSelect.addEventListener('change', (e) => {
        if (e.target.value === 'upload') {
            uploadControls.style.display = 'flex';
            cameraControls.style.display = 'none';
            stopCamera();
        } else {
            uploadControls.style.display = 'none';
            cameraControls.style.display = 'flex';
        }
    });

    // 文件选择
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            loadImageFromFile(file);
            detectBtn.disabled = false;
        }
    });

    // 检测按钮
    detectBtn.addEventListener('click', () => {
        if (fileInput.files[0]) {
            detectImage();
        }
    });

    // 摄像头控制
    startCameraBtn.addEventListener('click', startCamera);
    stopCameraBtn.addEventListener('click', stopCamera);

    // 自动检测
    autoDetectCheckbox.addEventListener('change', (e) => {
        autoDetect = e.target.checked;
    });

    // 置信度滑块
    if (confidenceSlider && confidenceValue) {
        // 初始化显示
        confidenceValue.textContent = confidenceThreshold.toFixed(2);
        confidenceSlider.value = Math.round(confidenceThreshold * 100);
        
        // 滑块变化事件
        confidenceSlider.addEventListener('input', (e) => {
            confidenceThreshold = parseFloat(e.target.value) / 100;
            confidenceValue.textContent = confidenceThreshold.toFixed(2);
            
            // 如果当前有图片且已检测过，自动重新检测
            if (canvas.width > 0 && canvas.height > 0 && !isProcessing) {
                // 延迟一下，避免滑块拖动时频繁检测
                clearTimeout(window.confidenceTimeout);
                window.confidenceTimeout = setTimeout(() => {
                    if (modeSelect.value === 'upload' && fileInput.files[0]) {
                        detectImage();
                    } else if (modeSelect.value === 'camera' && cameraStream) {
                        // 摄像头模式下，下次自动检测时会使用新阈值
                    }
                }, 300);
            }
        });
    }

    // 调试信息切换
    toggleDebugBtn.addEventListener('click', () => {
        const isVisible = debugOutput.style.display !== 'none';
        debugOutput.style.display = isVisible ? 'none' : 'block';
        toggleDebugBtn.textContent = isVisible ? '显示调试' : '隐藏调试';
    });
}

// 从文件加载图片
function loadImageFromFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            drawImageToCanvas(img);
            updateStatus('图片已加载');
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// 绘制图片到画布
function drawImageToCanvas(img) {
    // 计算适合的尺寸，保持宽高比
    const maxWidth = 800;
    const maxHeight = 600;
    let width = img.width;
    let height = img.height;

    if (width > maxWidth || height > maxHeight) {
        const ratio = Math.min(maxWidth / width, maxHeight / height);
        width = width * ratio;
        height = height * ratio;
    }

    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, width, height);
}

// 预处理图片数据
function preprocessImage(imageData, originalWidth, originalHeight) {
    // 获取模型输入尺寸（使用保存的尺寸或默认值）
    let targetSize = 640; // 默认值
    
    if (window.modelInputSize) {
        targetSize = window.modelInputSize;
    } else {
        // 尝试从session获取
        try {
            if (session && session.inputNames && session.inputNames.length > 0) {
                const firstInputName = session.inputNames[0];
                const metadata = session.inputMetadata && session.inputMetadata[firstInputName];
                if (metadata && metadata.shape) {
                    const inputShape = metadata.shape;
                    targetSize = inputShape[inputShape.length - 1] || 640;
                }
            }
        } catch (error) {
            console.warn('无法获取模型输入尺寸，使用默认值640:', error);
        }
    }
    
    // 创建临时画布，保持宽高比并添加padding
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = targetSize;
    tempCanvas.height = targetSize;
    const tempCtx = tempCanvas.getContext('2d');
    
    // 计算缩放比例，保持宽高比
    const scale = Math.min(targetSize / originalWidth, targetSize / originalHeight);
    const scaledWidth = originalWidth * scale;
    const scaledHeight = originalHeight * scale;
    
    // 计算居中位置
    const dx = (targetSize - scaledWidth) / 2;
    const dy = (targetSize - scaledHeight) / 2;
    
    // 填充黑色背景
    tempCtx.fillStyle = '#000000';
    tempCtx.fillRect(0, 0, targetSize, targetSize);
    
    // 绘制缩放后的图片（居中）
    tempCtx.drawImage(imageData, dx, dy, scaledWidth, scaledHeight);
    
    // 获取像素数据
    const imageDataResized = tempCtx.getImageData(0, 0, targetSize, targetSize);
    const data = imageDataResized.data;
    
    // 转换为RGB格式并归一化到[0,1]
    // 注意：有些模型需要归一化到[-1,1]或使用ImageNet标准化
    const input = new Float32Array(3 * targetSize * targetSize);
    
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i] / 255.0;
        const g = data[i + 1] / 255.0;
        const b = data[i + 2] / 255.0;
        const idx = Math.floor(i / 4);
        const h = Math.floor(idx / targetSize);
        const w = idx % targetSize;
        
        // CHW格式: [C, H, W]
        // 通道顺序：R, G, B
        input[h * targetSize + w] = r;
        input[targetSize * targetSize + h * targetSize + w] = g;
        input[2 * targetSize * targetSize + h * targetSize + w] = b;
    }
    
    // 保存缩放信息用于后处理
    window.lastPreprocessInfo = {
        scale,
        dx,
        dy,
        targetSize,
        originalWidth,
        originalHeight
    };
    
    // 返回tensor: [1, 3, H, W]
    return new ort.Tensor('float32', input, [1, 3, targetSize, targetSize]);
}

// 执行检测
async function detectImage() {
    if (!modelLoaded || isProcessing) return;
    
    isProcessing = true;
    updateStatus('检测中...');
    
    try {
        // 获取画布上的图片
        const img = new Image();
        img.src = canvas.toDataURL();
        
        await new Promise((resolve) => {
            img.onload = resolve;
        });
        
        // 预处理
        const inputTensor = preprocessImage(img, canvas.width, canvas.height);
        
        // 获取输入输出名称
        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];
        
        // 推理
        const outputs = await session.run({ [inputName]: inputTensor });
        const output = outputs[outputName];
        
        // 打印输出信息用于调试
        const outputSample = Array.from(output.data).slice(0, 100);
        const outputInfo = {
            shape: output.dims,
            dataRange: {
                min: Math.min(...outputSample),
                max: Math.max(...outputSample)
            }
        };
        
        console.log('模型输出形状:', output.dims);
        console.log('模型输出数据范围:', outputInfo.dataRange);
        addDebugInfo('模型输出', `形状: ${output.dims.join(' × ')}, 数据范围: [${outputInfo.dataRange.min.toFixed(3)}, ${outputInfo.dataRange.max.toFixed(3)}]`);
        
        // 后处理
        const detections = postprocess(output, canvas.width, canvas.height);
        
        console.log('检测结果数量:', detections.length);
        addDebugInfo('检测结果', `发现 ${detections.length} 个目标`);
        
        if (detections.length > 0) {
            console.log('第一个检测结果:', detections[0]);
            addDebugInfo('示例检测', JSON.stringify(detections[0], null, 2));
        }
        
        // 绘制结果
        if (detections.length > 0) {
            console.log('开始绘制检测结果，数量:', detections.length);
            drawDetections(detections);
        } else {
            console.log('没有检测结果需要绘制');
            // 即使没有检测结果，也确保原图显示
            if (originalImageData) {
                ctx.putImageData(originalImageData, 0, 0);
            }
        }
        
        // 显示信息
        displayDetectionInfo(detections);
        updateStatus(`检测完成，发现 ${detections.length} 个目标`);
        
    } catch (error) {
        console.error('检测失败:', error);
        updateStatus('检测失败: ' + error.message);
    } finally {
        isProcessing = false;
    }
}

// 后处理检测结果
function postprocess(output, imgWidth, imgHeight) {
    const detections = [];
    
    // 获取输出数据
    const outputData = output.data;
    const outputDims = output.dims;
    
    console.log('=== 后处理开始 ===');
    console.log('输出维度:', outputDims);
    console.log('输出数据长度:', outputData.length);
    console.log('当前置信度阈值:', confidenceThreshold);
    
    // 获取预处理信息
    const preprocessInfo = window.lastPreprocessInfo || { 
        scale: 1, 
        dx: 0, 
        dy: 0, 
        targetSize: 640,
        originalWidth: imgWidth,
        originalHeight: imgHeight
    };
    
    const targetSize = preprocessInfo.targetSize;
    const scale = preprocessInfo.scale;
    const dx = preprocessInfo.dx;
    const dy = preprocessInfo.dy;
    
    console.log('预处理信息:', preprocessInfo);
    console.log(`坐标转换参数: targetSize=${targetSize}, scale=${scale.toFixed(4)}, dx=${dx.toFixed(2)}, dy=${dy.toFixed(2)}`);
    
    // 根据输出维度判断格式
    // YOLO常见格式:
    // 1. [1, num_detections, 85] - 85 = 4(bbox) + 1(objectness) + 80(classes)
    // 2. [1, 25200, 85] - YOLOv5/v8格式
    // 3. [num_detections, 6] - 6 = x1, y1, x2, y2, conf, class_id (已处理格式)
    // 4. [1, num_detections, 9] - 9 = 4(bbox) + 1(objectness) + 4(classes) 对于4类模型
    
    let numDetections = 1;
    let detectionSize = outputData.length;
    
    // 解析输出维度
    // YOLOv8输出通常是 [1, num_detections, features] 或 [num_detections, features]
    if (outputDims.length === 2) {
        // 格式: [num_detections, features]
        numDetections = outputDims[0];
        detectionSize = outputDims[1];
    } else if (outputDims.length === 3) {
        // 格式可能是:
        // [batch, features, num_detections] - 如 [1, 8, 8400] (转置格式)
        // [batch, num_detections, features] - 如 [1, 8400, 8] (正常格式)
        // 参考C#实现：if (dim2 > dim1 && dim2 > 1000) 认为是转置格式
        const dim1 = outputDims[1];
        const dim2 = outputDims[2];
        
        let isTransposed = false;
        
        // 判断转置格式：dim2 > dim1 且 dim2 > 1000
        if (dim2 > dim1 && dim2 > 1000) {
            // 格式: [1, features, num_detections] - 转置格式
            numDetections = dim2;
            detectionSize = dim1;
            isTransposed = true;
            console.log(`检测到转置格式 [batch, features, num_detections]: ${outputDims.join(' × ')}`);
        } else {
            // 格式: [1, num_detections, features] - 正常格式
            numDetections = dim1;
            detectionSize = dim2;
            isTransposed = false;
            console.log(`检测到正常格式 [batch, num_detections, features]: ${outputDims.join(' × ')}`);
        }
        
        // 保存转置标志供后续使用
        window.isTransposedFormat = isTransposed;
    } else if (outputDims.length === 1) {
        // 格式: [features] - 单个检测
        numDetections = 1;
        detectionSize = outputDims[0];
    } else {
        // 尝试计算
        numDetections = outputDims[outputDims.length - 2] || 1;
        detectionSize = outputDims[outputDims.length - 1] || outputData.length;
    }
    
    // 验证计算是否正确
    const expectedTotal = numDetections * detectionSize;
    if (Math.abs(expectedTotal - outputData.length) > 10) {
        console.warn(`维度解析可能错误: 期望 ${expectedTotal} 个元素，实际 ${outputData.length} 个`);
        // 重新计算
        detectionSize = Math.floor(outputData.length / numDetections);
        console.warn(`重新计算 detectionSize = ${detectionSize}`);
    }
    
    console.log(`输出维度解析: ${outputDims.join(' × ')}`);
    console.log(`解析格式: ${numDetections} 个检测候选, 每个 ${detectionSize} 维`);
    console.log(`预期总元素数: ${numDetections * detectionSize}, 实际: ${outputData.length}`);
    
    // 打印前几个检测候选的原始数据用于调试
    if (numDetections > 0) {
        console.log('前3个检测候选的完整原始数据:');
        for (let i = 0; i < Math.min(3, numDetections); i++) {
            const offset = i * detectionSize;
            const sample = [];
            for (let j = 0; j < detectionSize; j++) {
                sample.push(outputData[offset + j]);
            }
            console.log(`候选 ${i} 完整数据 (${detectionSize}维):`, sample.map(v => v.toFixed(2)));
            console.log(`候选 ${i} 解析: x=${sample[0].toFixed(2)}, y=${sample[1].toFixed(2)}, w=${sample[2].toFixed(2)}, h=${sample[3].toFixed(2)}, obj=${sample[4].toFixed(3)}`);
            if (detectionSize > 5) {
                const classScores = sample.slice(5).map((v, idx) => `类别${idx}=${v.toFixed(3)}`);
                console.log(`候选 ${i} 类别分数:`, classScores.join(', '));
            }
        }
    }
    
    // 使用全局置信度阈值（可通过滑块动态调整）
    // confidenceThreshold 已在全局定义，默认0.3
    const nmsThreshold = 0.5; // 提高NMS阈值，减少重复检测
    
    // 解析检测结果
    let validCandidates = 0;
    let filteredByConfidence = 0;
    let filteredByClass = 0;
    
    // 获取转置标志
    const isTransposed = window.isTransposedFormat || false;
    
    for (let i = 0; i < numDetections; i++) {
        let offset, cx, cy, w, h, class0, class1, class2, class3;
        
        if (detectionSize === 8) {
            // 格式: [cx, cy, w, h, class0, class1, class2, class3] - 中心点坐标 + 4个类别分数（无objectness）
            // 参考C#实现：支持转置格式 [batch, features, num_detections]
            if (isTransposed) {
                // 转置格式：索引 = featureIndex * numDetections + detectionIndex
                cx = outputData[0 * numDetections + i];
                cy = outputData[1 * numDetections + i];
                w = outputData[2 * numDetections + i];
                h = outputData[3 * numDetections + i];
                class0 = outputData[4 * numDetections + i];
                class1 = outputData[5 * numDetections + i];
                class2 = outputData[6 * numDetections + i];
                class3 = outputData[7 * numDetections + i];
            } else {
                // 正常格式：索引 = detectionIndex * numFeatures + featureIndex
                offset = i * detectionSize;
                cx = outputData[offset + 0];
                cy = outputData[offset + 1];
                w = outputData[offset + 2];
                h = outputData[offset + 3];
                class0 = outputData[offset + 4];
                class1 = outputData[offset + 5];
                class2 = outputData[offset + 6];
                class3 = outputData[offset + 7];
            }
            
            // 找到最大类别分数
            const classScores = [class0, class1, class2, class3];
            let maxClassConf = Math.max(...classScores);
            let maxClassId = classScores.indexOf(maxClassConf);
            
            if (i < 3) {
                console.log(`候选 ${i} 8维格式: cx=${cx.toFixed(2)}, cy=${cy.toFixed(2)}, w=${w.toFixed(2)}, h=${h.toFixed(2)}`);
                console.log(`候选 ${i} 类别分数: 类别0=${class0.toFixed(3)}, 类别1=${class1.toFixed(3)}, 类别2=${class2.toFixed(3)}, 类别3=${class3.toFixed(3)}`);
                console.log(`候选 ${i} 最大类别: 类别${maxClassId}, 分数=${maxClassConf.toFixed(3)}`);
            }
            
            // 归一化置信度（参考C#实现）
            // 如果类别分数 > 1.0，认为是logits，使用sigmoid
            const sigmoid = (x) => 1 / (1 + Math.exp(-x));
            let finalConf = maxClassConf > 1.0 ? sigmoid(maxClassConf) : maxClassConf;
            
            // 限制置信度在合理范围内
            finalConf = Math.max(0, Math.min(1, finalConf));
            
            if (i < 3 && maxClassConf > 1.0) {
                console.log(`候选 ${i}: 类别分数 ${maxClassConf.toFixed(3)} -> sigmoid ${finalConf.toFixed(3)}`);
            }
            
            // 获取类别名称
            const className = CLASS_NAMES[maxClassId] || `类别${maxClassId}`;
            
            if (finalConf > confidenceThreshold) {
                validCandidates++;
                
                // 坐标格式判断（参考C#实现）
                // 判断是否归一化：maxCoord <= 2.0
                const maxCoord = Math.max(Math.max(cx, cy), Math.max(w, h));
                const isNormalized = maxCoord <= 2.0;
                
                // 判断是否为固定尺寸格式（相对于640）
                const isFixedSize = (cx > targetSize || cy > targetSize || w > targetSize || h > targetSize);
                
                let normCx = cx, normCy = cy, normW = w, normH = h;
                
                if (isNormalized) {
                    // 归一化坐标 [0, 1]，转换为模型输入尺寸
                    normCx = cx * targetSize;
                    normCy = cy * targetSize;
                    normW = w * targetSize;
                    normH = h * targetSize;
                    if (i < 3) console.log(`候选 ${i}: 归一化坐标 -> 模型尺寸`);
                } else if (isFixedSize) {
                    // 固定尺寸格式（相对于640），缩放到实际输入尺寸
                    const scaleToResized = targetSize / 640.0;
                    normCx = cx * scaleToResized;
                    normCy = cy * scaleToResized;
                    normW = w * scaleToResized;
                    normH = h * scaleToResized;
                    if (i < 3) console.log(`候选 ${i}: 固定尺寸格式 -> 模型尺寸`);
                } else {
                    // 已经是相对于模型输入尺寸的坐标
                    if (i < 3) console.log(`候选 ${i}: 直接使用模型尺寸坐标`);
                }
                
                // Letterbox坐标映射（参考C#实现）
                // 先减去padding
                normCx = normCx - dx;
                normCy = normCy - dy;
                
                // 转换为左上角坐标系并还原到原图尺寸
                // x = (cx - w/2) / ratio
                let origX1 = (normCx - normW / 2) / scale;
                let origY1 = (normCy - normH / 2) / scale;
                let origX2 = (normCx + normW / 2) / scale;
                let origY2 = (normCy + normH / 2) / scale;
                
                // 确保坐标在原图范围内
                origX1 = Math.max(0, Math.min(imgWidth, origX1));
                origY1 = Math.max(0, Math.min(imgHeight, origY1));
                origX2 = Math.max(0, Math.min(imgWidth, origX2));
                origY2 = Math.max(0, Math.min(imgHeight, origY2));
                
                // 验证检测框有效性
                const boxWidth = origX2 - origX1;
                const boxHeight = origY2 - origY1;
                const minBoxSize = 20; // 最小检测框尺寸（像素）
                const minAspectRatio = 0.1; // 最小宽高比
                const maxAspectRatio = 10; // 最大宽高比
                
                // 计算宽高比
                const aspectRatio = boxHeight > 0 ? boxWidth / boxHeight : 0;
                
                if (boxWidth > minBoxSize && boxHeight > minBoxSize && 
                    aspectRatio >= minAspectRatio && aspectRatio <= maxAspectRatio &&
                    origX1 >= 0 && origY1 >= 0 && 
                    origX2 <= imgWidth && origY2 <= imgHeight) {
                    detections.push({
                        x1: origX1,
                        y1: origY1,
                        x2: origX2,
                        y2: origY2,
                        confidence: finalConf,
                        classId: maxClassId,
                        className: className
                    });
                    
                    if (detections.length <= 3) {
                        console.log(`检测 ${detections.length}: ${className}, 置信度=${finalConf.toFixed(3)}, 坐标=(${origX1.toFixed(1)}, ${origY1.toFixed(1)}, ${origX2.toFixed(1)}, ${origY2.toFixed(1)}), 尺寸=(${boxWidth.toFixed(1)}, ${boxHeight.toFixed(1)})`);
                    }
                } else {
                    if (i < 5) {
                        console.log(`候选 ${i}: 检测框无效 - 宽度=${boxWidth.toFixed(1)}, 高度=${boxHeight.toFixed(1)}, 坐标=(${origX1.toFixed(1)}, ${origY1.toFixed(1)}, ${origX2.toFixed(1)}, ${origY2.toFixed(1)})`);
                    }
                }
            } else {
                filteredByConfidence++;
                if (i < 5) {
                    console.log(`候选 ${i}: 置信度 ${finalConf.toFixed(3)} 低于阈值 ${confidenceThreshold}`);
                }
            }
        } else if (detectionSize === 6) {
            // 格式: [x1, y1, x2, y2, confidence, class_id] (已处理格式，如YOLOv8的某些导出)
            const x1 = outputData[offset + 0];
            const y1 = outputData[offset + 1];
            const x2 = outputData[offset + 2];
            const y2 = outputData[offset + 3];
            const conf = outputData[offset + 4];
            const classId = Math.round(outputData[offset + 5]);
            
            // 坐标已经在模型输出中处理，但需要从模型输入尺寸转换到原图尺寸
            // 如果坐标是归一化的[0,1]，需要先乘以targetSize
            let normX1 = x1, normY1 = y1, normX2 = x2, normY2 = y2;
            
            // 判断坐标是否归一化
            if (Math.abs(x1) <= 1 && Math.abs(y1) <= 1 && Math.abs(x2) <= 1 && Math.abs(y2) <= 1) {
                normX1 = x1 * targetSize;
                normY1 = y1 * targetSize;
                normX2 = x2 * targetSize;
                normY2 = y2 * targetSize;
            }
            
            // 转换回原图坐标（考虑padding和缩放）
            const origX1 = (normX1 - dx) / scale;
            const origY1 = (normY1 - dy) / scale;
            const origX2 = (normX2 - dx) / scale;
            const origY2 = (normY2 - dy) / scale;
            
            // 获取类别名称
            const className = CLASS_NAMES[classId] || `类别${classId}`;
            
            // 类别过滤（已禁用，显示所有类别）
            // if (TARGET_CLASSES.size > 0 && !TARGET_CLASSES.has(className)) {
            //     filteredByClass++;
            //     if (i < 5) {
            //         console.log(`候选 ${i}: 类别 ${className} 不在目标类别中，已过滤`);
            //     }
            //     continue;
            // }
            
            if (conf > confidenceThreshold) {
                validCandidates++;
                detections.push({
                    x1: Math.max(0, Math.min(imgWidth, origX1)),
                    y1: Math.max(0, Math.min(imgHeight, origY1)),
                    x2: Math.max(0, Math.min(imgWidth, origX2)),
                    y2: Math.max(0, Math.min(imgHeight, origY2)),
                    confidence: conf,
                    classId: classId,
                    className: className
                });
                
                if (detections.length <= 3) {
                    console.log(`检测 ${detections.length}: ${className}, 置信度=${conf.toFixed(3)}, 坐标=(${origX1.toFixed(1)}, ${origY1.toFixed(1)}, ${origX2.toFixed(1)}, ${origY2.toFixed(1)})`);
                }
            } else {
                filteredByConfidence++;
                if (i < 5) {
                    console.log(`候选 ${i}: 置信度 ${conf.toFixed(3)} 低于阈值 ${confidenceThreshold}`);
                }
            }
        } else if (detectionSize >= 5) {
            // YOLOv8标准格式: [x_center, y_center, width, height, objectness, ...class_scores]
            // 对于4类模型，detectionSize可能是9 (4+1+4)
            
            const x = outputData[offset + 0];
            const y = outputData[offset + 1];
            const w = outputData[offset + 2];
            const h = outputData[offset + 3];
            const objectness = outputData[offset + 4];
            
            if (i < 3) {
                console.log(`候选 ${i} 原始值: x=${x.toFixed(2)}, y=${y.toFixed(2)}, w=${w.toFixed(2)}, h=${h.toFixed(2)}, objectness=${objectness.toFixed(3)}`);
            }
            
            // 找到最大类别分数
            let maxClassConf = 0;
            let maxClassId = 0;
            const classScores = [];
            for (let j = 5; j < detectionSize; j++) {
                const classScore = outputData[offset + j];
                const classIdx = j - 5;
                classScores.push({ idx: classIdx, score: classScore });
                if (classScore > maxClassConf) {
                    maxClassConf = classScore;
                    maxClassId = classIdx;
                }
            }
            
            if (i < 3) {
                console.log(`候选 ${i} 类别分数详情:`, classScores.map(c => `类别${c.idx}=${c.score.toFixed(3)}`).join(', '));
                console.log(`候选 ${i} 最大类别: 类别${maxClassId}, 分数=${maxClassConf.toFixed(3)}`);
            }
            
            // 计算最终置信度
            // 如果值很大（>1），可能是原始logits，需要sigmoid处理
            // 如果值在[0,1]范围内，直接使用
            let normalizedObjectness = objectness;
            let normalizedClassConf = maxClassConf;
            
            // Sigmoid函数
            const sigmoid = (x) => 1 / (1 + Math.exp(-x));
            
            // 判断是否需要归一化
            if (Math.abs(objectness) > 1 || Math.abs(maxClassConf) > 1) {
                // 可能是logits，使用sigmoid
                normalizedObjectness = sigmoid(objectness);
                normalizedClassConf = sigmoid(maxClassConf);
                if (i < 3) {
                    console.log(`候选 ${i}: 使用sigmoid归一化, objectness: ${objectness.toFixed(3)} -> ${normalizedObjectness.toFixed(3)}, classConf: ${maxClassConf.toFixed(3)} -> ${normalizedClassConf.toFixed(3)}`);
                }
            }
            
            // 计算最终置信度（YOLOv8使用 objectness * class_score）
            const finalConf = normalizedObjectness * normalizedClassConf;
            
            // 获取类别名称
            const className = CLASS_NAMES[maxClassId] || `类别${maxClassId}`;
            
            // 类别过滤（已禁用，显示所有类别）
            // if (TARGET_CLASSES.size > 0 && !TARGET_CLASSES.has(className)) {
            //     filteredByClass++;
            //     if (i < 5) {
            //         console.log(`候选 ${i}: 类别 ${className} 不在目标类别中，已过滤`);
            //     }
            //     continue; // 跳过不在目标类别中的检测
            // }
            
            if (finalConf > confidenceThreshold) {
                validCandidates++;
                // YOLOv8输出格式判断
                let normX, normY, normW, normH;
                
                // 判断坐标格式：检查最大值来判断是归一化还是绝对坐标
                const maxCoord = Math.max(Math.abs(x), Math.abs(y), Math.abs(x + w), Math.abs(y + h));
                
                if (maxCoord <= 1.1) {
                    // 归一化坐标 [0, 1]
                    normX = x * targetSize;
                    normY = y * targetSize;
                    normW = w * targetSize;
                    normH = h * targetSize;
                    if (i < 3) console.log(`候选 ${i}: 归一化坐标 (${x.toFixed(3)}, ${y.toFixed(3)}, ${w.toFixed(3)}, ${h.toFixed(3)}) -> (${normX.toFixed(1)}, ${normY.toFixed(1)}, ${normW.toFixed(1)}, ${normH.toFixed(1)})`);
                } else if (maxCoord <= targetSize * 1.1) {
                    // 相对于输入尺寸的绝对坐标
                    normX = x;
                    normY = y;
                    normW = w;
                    normH = h;
                    if (i < 3) console.log(`候选 ${i}: 绝对坐标 (相对于${targetSize})`);
                } else {
                    // 可能是已经处理过的坐标，直接使用
                    normX = x;
                    normY = y;
                    normW = w;
                    normH = h;
                    if (i < 3) console.log(`候选 ${i}: 原始坐标 (可能已处理)`);
                }
                
                // 判断坐标格式：可能是中心点+宽高，也可能是左上角+宽高，或者直接是左上角+右下角
                // 通过检查w和h的值来判断：如果w和h都小于targetSize，可能是宽高；如果都大于targetSize，可能是右下角坐标
                let normX1, normY1, normX2, normY2;
                
                if (w > targetSize || h > targetSize) {
                    // 可能是左上角+右下角格式
                    normX1 = normX;
                    normY1 = normY;
                    normX2 = w;
                    normY2 = h;
                    if (i < 3) console.log(`候选 ${i}: 识别为左上角+右下角格式`);
                } else {
                    // 可能是中心点+宽高格式
                    normX1 = normX - normW / 2;
                    normY1 = normY - normH / 2;
                    normX2 = normX + normW / 2;
                    normY2 = normY + normH / 2;
                    if (i < 3) console.log(`候选 ${i}: 识别为中心点+宽高格式`);
                }
                
                // 确保坐标在模型输入尺寸范围内
                normX1 = Math.max(0, Math.min(targetSize, normX1));
                normY1 = Math.max(0, Math.min(targetSize, normY1));
                normX2 = Math.max(0, Math.min(targetSize, normX2));
                normY2 = Math.max(0, Math.min(targetSize, normY2));
                
                if (i < 3) {
                    console.log(`候选 ${i}: 模型坐标框(${normX1.toFixed(1)}, ${normY1.toFixed(1)}, ${normX2.toFixed(1)}, ${normY2.toFixed(1)})`);
                }
                
                // 转换回原图坐标（考虑padding和缩放）
                // 公式：原图坐标 = (模型坐标 - padding偏移) / 缩放比例
                let origX1 = (normX1 - dx) / scale;
                let origY1 = (normY1 - dy) / scale;
                let origX2 = (normX2 - dx) / scale;
                let origY2 = (normY2 - dy) / scale;
                
                if (i < 3) {
                    console.log(`候选 ${i}: 转换前(${normX1.toFixed(1)}, ${normY1.toFixed(1)}, ${normX2.toFixed(1)}, ${normY2.toFixed(1)}), dx=${dx.toFixed(1)}, dy=${dy.toFixed(1)}, scale=${scale.toFixed(4)}`);
                    console.log(`候选 ${i}: 转换后(${origX1.toFixed(1)}, ${origY1.toFixed(1)}, ${origX2.toFixed(1)}, ${origY2.toFixed(1)})`);
                }
                
                // 确保坐标在原图范围内
                origX1 = Math.max(0, Math.min(imgWidth, origX1));
                origY1 = Math.max(0, Math.min(imgHeight, origY1));
                origX2 = Math.max(0, Math.min(imgWidth, origX2));
                origY2 = Math.max(0, Math.min(imgHeight, origY2));
                
                if (i < 3) {
                    console.log(`候选 ${i}: 最终坐标(${origX1.toFixed(1)}, ${origY1.toFixed(1)}, ${origX2.toFixed(1)}, ${origY2.toFixed(1)}), 原图尺寸(${imgWidth}, ${imgHeight})`);
                }
                
                // 确保检测框有效（宽度和高度都大于0）
                if (origX2 > origX1 && origY2 > origY1) {
                    detections.push({
                        x1: origX1,
                        y1: origY1,
                        x2: origX2,
                        y2: origY2,
                        confidence: finalConf,
                        classId: maxClassId,
                        className: className
                    });
                } else {
                    console.warn(`候选 ${i}: 检测框无效，宽度或高度为0`);
                }
                
                if (detections.length <= 3) {
                    console.log(`检测 ${detections.length}: ${className}, 置信度=${finalConf.toFixed(3)}, objectness=${objectness.toFixed(3)}, classConf=${maxClassConf.toFixed(3)}, 坐标=(${origX1.toFixed(1)}, ${origY1.toFixed(1)}, ${origX2.toFixed(1)}, ${origY2.toFixed(1)})`);
                }
            } else {
                filteredByConfidence++;
                if (i < 5) {
                    console.log(`候选 ${i}: 最终置信度 ${finalConf.toFixed(3)} (objectness=${objectness.toFixed(3)} * classConf=${maxClassConf.toFixed(3)}) 低于阈值 ${confidenceThreshold}`);
                }
            }
        }
    }
    
    console.log(`=== 后处理统计 ===`);
    console.log(`总候选数: ${numDetections}`);
    console.log(`有效检测数: ${validCandidates}`);
    console.log(`因置信度过滤: ${filteredByConfidence}`);
    console.log(`因类别过滤: ${filteredByClass}`);
    console.log(`过滤前检测数量: ${detections.length}`);
    
    // 非极大值抑制 (NMS)
    const finalDetections = nms(detections, nmsThreshold);
    
    console.log(`NMS后检测数量: ${finalDetections.length}`);
    console.log('=== 后处理结束 ===');
    
    return finalDetections;
}

// 非极大值抑制
function nms(detections, threshold) {
    if (detections.length === 0) return [];
    
    // 按置信度排序
    detections.sort((a, b) => b.confidence - a.confidence);
    
    const selected = [];
    
    while (detections.length > 0) {
        const current = detections.shift();
        selected.push(current);
        
        // 移除与当前框IoU大于阈值的框
        detections = detections.filter(det => {
            const iou = calculateIoU(current, det);
            // 如果是同一类别且IoU高，移除；如果是不同类别但IoU很高，也移除（可能是误检）
            if (current.classId === det.classId) {
                return iou < threshold;
            } else {
                // 不同类别但IoU很高，可能是重复检测，使用更严格的阈值
                return iou < threshold * 0.6;
            }
        });
    }
    
    return selected;
}

// 计算IoU
function calculateIoU(box1, box2) {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);
    
    if (x2 <= x1 || y2 <= y1) return 0;
    
    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    const union = area1 + area2 - intersection;
    
    return intersection / union;
}

// 保存原始图片数据
let originalImageData = null;

// 绘制图片到画布
function drawImageToCanvas(img) {
    // 计算适合的尺寸，保持宽高比
    const maxWidth = 800;
    const maxHeight = 600;
    let width = img.width;
    let height = img.height;

    if (width > maxWidth || height > maxHeight) {
        const ratio = Math.min(maxWidth / width, maxHeight / height);
        width = width * ratio;
        height = height * ratio;
    }

    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, width, height);
    
    // 保存原始图片数据
    originalImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
}

// 绘制检测结果
function drawDetections(detections) {
    if (!detections || detections.length === 0) {
        console.log('drawDetections: 没有检测结果');
        return;
    }
    
    console.log('drawDetections: 准备绘制', detections.length, '个检测结果');
    
    // 恢复原始图片
    if (originalImageData) {
        ctx.putImageData(originalImageData, 0, 0);
        console.log('drawDetections: 已恢复原始图片');
    } else {
        // 如果没有保存的原图，尝试从canvas获取
        console.log('drawDetections: 没有保存的原图，从canvas重新加载');
        const img = new Image();
        img.src = canvas.toDataURL();
        img.onload = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            // 保存原图数据
            originalImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            drawBoxes(detections);
        };
        return;
    }
    
    drawBoxes(detections);
}

// 绘制检测框
function drawBoxes(detections) {
    if (!detections || detections.length === 0) {
        console.log('drawBoxes: 没有检测结果');
        return;
    }
    
    console.log('drawBoxes: 开始绘制', detections.length, '个检测框');
    console.log('画布尺寸:', canvas.width, 'x', canvas.height);
    
    let drawnCount = 0;
    detections.forEach((det, index) => {
        const x = Math.round(det.x1);
        const y = Math.round(det.y1);
        const w = Math.round(det.x2 - det.x1);
        const h = Math.round(det.y2 - det.y1);
        
        // 确保坐标有效
        if (w <= 0 || h <= 0 || x < 0 || y < 0 || x + w > canvas.width || y + h > canvas.height) {
            console.warn(`检测框 ${index} 无效:`, {
                x, y, w, h,
                x2: det.x2,
                y2: det.y2,
                canvasSize: `${canvas.width}x${canvas.height}`
            });
            return;
        }
        
        console.log(`绘制检测框 ${index + 1}:`, {
            className: det.className || CLASS_NAMES[det.classId] || `类别${det.classId}`,
            confidence: det.confidence.toFixed(3),
            bbox: `(${x}, ${y}, ${w}, ${h})`
        });
        
        // 绘制框（与Python代码中的绿色一致）
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        
        // 绘制标签（格式与Python代码一致："{label} {conf:.2f}"）
        const className = det.className || CLASS_NAMES[det.classId] || `类别${det.classId}`;
        const label = `${className} ${det.confidence.toFixed(2)}`;
        ctx.font = 'bold 14px Arial';
        const textWidth = ctx.measureText(label).width;
        const labelHeight = 22;
        const labelY = Math.max(labelHeight, y);
        
        // 绘制标签背景（绿色，与Python代码一致）
        ctx.fillStyle = 'rgba(0, 255, 0, 0.85)';
        ctx.fillRect(x, labelY - labelHeight, textWidth + 12, labelHeight);
        
        // 绘制标签文字
        ctx.fillStyle = '#000';
        ctx.fillText(label, x + 6, labelY - 6);
        
        drawnCount++;
    });
    
    console.log(`drawBoxes: 成功绘制 ${drawnCount}/${detections.length} 个检测框`);
}

// 显示检测信息
function displayDetectionInfo(detections) {
    if (detections.length === 0) {
        detectionInfo.innerHTML = '<p class="info-text">未检测到目标</p>';
        return;
    }
    
    let html = `<p class="info-text">检测到 <strong>${detections.length}</strong> 个目标：</p>`;
    
    detections.forEach((det, index) => {
        const className = det.className || CLASS_NAMES[det.classId] || `类别${det.classId}`;
        html += `
            <div class="detection-item">
                <strong>目标 ${index + 1}</strong><br>
                类别: ${className} (ID: ${det.classId})<br>
                置信度: ${(det.confidence * 100).toFixed(2)}%<br>
                位置: (${Math.round(det.x1)}, ${Math.round(det.y1)}) - (${Math.round(det.x2)}, ${Math.round(det.y2)})
            </div>
        `;
    });
    
    detectionInfo.innerHTML = html;
}

// 启动摄像头
async function startCamera() {
    try {
        updateStatus('正在启动摄像头...');
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        
        const video = document.createElement('video');
        video.srcObject = cameraStream;
        video.play();
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // 开始绘制视频帧
            let lastDetectionTime = 0;
            const detectionInterval = 200; // 每200ms检测一次，避免过于频繁
            
            function drawFrame() {
                if (cameraStream) {
                    // 先绘制视频帧
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    // 保存当前帧作为原图（用于绘制检测结果）
                    // 注意：在检测完成后会恢复这个原图，然后绘制检测框
                    originalImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    
                    // 自动检测（限制频率）
                    const now = Date.now();
                    if (autoDetect && !isProcessing && (now - lastDetectionTime) > detectionInterval) {
                        lastDetectionTime = now;
                        detectImage();
                    }
                    
                    requestAnimationFrame(drawFrame);
                }
            }
            
            drawFrame();
        };
        
        startCameraBtn.disabled = true;
        stopCameraBtn.disabled = false;
        updateStatus('摄像头已启动');
        
    } catch (error) {
        console.error('摄像头启动失败:', error);
        updateStatus('摄像头启动失败: ' + error.message);
        alert('无法访问摄像头，请检查权限设置');
    }
}

// 停止摄像头
function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    
    if (cameraInterval) {
        clearInterval(cameraInterval);
        cameraInterval = null;
    }
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    startCameraBtn.disabled = false;
    stopCameraBtn.disabled = true;
    updateStatus('摄像头已停止');
}

// 更新状态
function updateStatus(message) {
    status.textContent = message;
}

// 更新模型状态
function updateModelStatus(message) {
    modelStatus.textContent = message;
}

// 添加调试信息
function addDebugInfo(category, message) {
    const timestamp = new Date().toLocaleTimeString();
    const debugEntry = document.createElement('div');
    debugEntry.style.marginBottom = '5px';
    debugEntry.innerHTML = `<strong>[${timestamp}] ${category}:</strong> ${message}`;
    debugOutput.appendChild(debugEntry);
    debugOutput.scrollTop = debugOutput.scrollHeight;
}

// 显示/隐藏加载动画
function showLoading(show) {
    loading.style.display = show ? 'flex' : 'none';
}

// 页面加载完成后初始化
window.addEventListener('load', init);
