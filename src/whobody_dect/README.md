# 多阶段坐标变换可视化

## 保留的核心文件

### 1. simple_visualize.py
**主要可视化脚本**

功能：
- 摄像头读取
- MediaPipe检测和融合
- 两步坐标转换
- URDF标准输出
- 3D可视化（支持交互式视角）

使用方法：
```bash
python simple_visualize.py
```

### 4. 核心模块
- `fusion_module.py` - 数据融合
- `arm_coordinate_transformer.py` - 坐标转换
- `skeleton_config.json` - 配置文件

## 交互式视角控制

### 操作方式
- **鼠标左键拖拽**: 旋转视角
- **鼠标右键拖拽**: 平移视角
- **滚轮**: 缩放
- **双击**: 重置视角
- **按 'q'**: 退出程序

## 坐标系转换流程

1. **Stage 1**: Mediapipe原始世界坐标
   - X(右), Y(上), Z(前)

2. **Stage 2**: 第一人称视角局部坐标
   - 以肩膀为原点
   - X(右), Y(下), Z(前)

3. **Stage 3**: URDF标准坐标
   - X(前), Y(左), Z(上)
   - 符合机器人学标准

## 快速开始

```bash
# 运行可视化
python simple_visualize.py


```
