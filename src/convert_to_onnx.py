from ultralytics import YOLO

# 1. Load mô hình xịn nhất của bạn
model = YOLO(r'models\yolo26m_seg_train\weights\best.pt')

# 2. Xuất sang định dạng ONNX
# Tham số dynamic=True giúp mô hình linh hoạt nhận ảnh nhiều kích thước khác nhau
path = model.export(format='onnx', imgsz=768, dynamic=True)

print(f"✅ Đã xuất file ONNX thành công tại: {path}")