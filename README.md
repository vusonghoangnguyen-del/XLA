# Phát hiện đặc trưng ảnh đa tỷ lệ bằng đạo hàm Gaussian

Dự án cài đặt bằng Python các phương pháp phát hiện đặc trưng ảnh ở nhiều tỷ lệ:

- Làm trơn Gaussian tại nhiều giá trị `sigma`.
- Tính đạo hàm Gaussian bậc nhất: `Lx`, `Ly`.
- Tính độ lớn gradient và phát hiện biên.
- Tính đạo hàm Gaussian bậc hai: `Lxx`, `Lxy`, `Lyy`.
- Phân tích ma trận Hessian và phát hiện đường gờ (ridge).
- Sử dụng Laplacian of Gaussian để phát hiện blob.

Đây là thuật toán xử lý ảnh cổ điển, **không phải mô hình học máy**. Chương
trình không cần huấn luyện, không có trọng số học được và có thể xử lý ảnh
ngay sau khi cài đặt các thư viện cần thiết.

## 1. Yêu cầu hệ thống

- Python 3.10 trở lên.
- NumPy.
- OpenCV.
- Matplotlib.

## 2. Cài đặt trên Windows

Mở PowerShell tại thư mục dự án và chạy:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn script kích hoạt môi trường, có thể không cần kích hoạt và
gọi trực tiếp Python trong `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Cài đặt trên Linux hoặc macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Chạy chương trình

### 4.1. Chạy với cấu hình mặc định

```powershell
python gaussian_multiscale_features.py "đường_dẫn_đến_ảnh.tif"
```

Nếu chưa kích hoạt `.venv` trên Windows:

```powershell
.\.venv\Scripts\python.exe .\gaussian_multiscale_features.py "đường_dẫn_đến_ảnh.tif"
```

Chương trình hỗ trợ các định dạng ảnh OpenCV đọc được, ví dụ: PNG, JPG, JPEG,
BMP và TIFF.

### 4.2. Chọn các mức Sigma

Mặc định chương trình sử dụng `1 2 3 5`. Có thể thay đổi bằng:

```powershell
python gaussian_multiscale_features.py "input.tif" --sigmas 0.8 1.5 3 6
```

Ý nghĩa của Sigma:

- Sigma nhỏ: giữ nhiều chi tiết nhưng nhạy với nhiễu.
- Sigma trung bình: cân bằng giữa chi tiết và khả năng loại nhiễu.
- Sigma lớn: giữ các cấu trúc lớn nhưng làm mất các chi tiết nhỏ.

### 4.3. Chọn thư mục kết quả

```powershell
python gaussian_multiscale_features.py "input.tif" --output "results\anh_01"
```

Nếu không truyền `--output`, kết quả được lưu trong `multiscale_results`.

### 4.4. Điều chỉnh ngưỡng phát hiện biên

```powershell
python gaussian_multiscale_features.py "input.tif" --edge-percentile 80
```

`--edge-percentile` nằm trong khoảng từ 0 đến 100:

- Tăng giá trị: chỉ giữ lại các biên mạnh.
- Giảm giá trị: giữ thêm các biên yếu nhưng có thể làm tăng nhiễu.

Giá trị mặc định là `75`.

### 4.5. Chọn loại đường gờ

```powershell
python gaussian_multiscale_features.py "input.tif" --ridge-polarity bright
python gaussian_multiscale_features.py "input.tif" --ridge-polarity dark
python gaussian_multiscale_features.py "input.tif" --ridge-polarity both
```

- `bright`: phát hiện đường gờ sáng trên nền tối.
- `dark`: phát hiện đường gờ tối trên nền sáng.
- `both`: phát hiện cả hai loại đường gờ.

## 5. Ví dụ chạy đầy đủ

```powershell
.\.venv\Scripts\python.exe .\gaussian_multiscale_features.py `
  "D:\data\kidney.tif" `
  --sigmas 1 2 3 5 `
  --edge-percentile 75 `
  --ridge-polarity bright `
  --output ".\results\kidney"
```

## 6. Các file kết quả

Với mỗi giá trị Sigma, chương trình tạo các file:

| Hậu tố file | Nội dung |
| --- | --- |
| `gaussian` | Ảnh sau khi làm trơn Gaussian |
| `lx`, `ly` | Đạo hàm bậc nhất theo trục x và y |
| `gradient` | Độ lớn gradient |
| `edge_strength` | Gradient sau bước làm mảnh đường biên |
| `edge_mask` | Ảnh biên nhị phân cuối cùng |
| `lxx`, `lxy`, `lyy` | Các đạo hàm bậc hai dùng để tạo Hessian |
| `ridge` | Đáp ứng phát hiện đường gờ từ trị riêng Hessian |
| `blob_log` | Đáp ứng blob từ Laplacian of Gaussian |

Ngoài ra:

- `original_grayscale.png`: ảnh xám đầu vào.
- `multiscale_summary.png`: bảng tổng hợp để so sánh các Sigma.

Ví dụ, `sigma_3_edge_mask.png` là ảnh biên nhị phân tại `sigma = 3`.

## 7. Quy trình thuật toán

```text
Ảnh đầu vào
    -> Chuyển sang ảnh xám
    -> Lặp qua các giá trị Sigma
    -> Làm trơn và tính đạo hàm Gaussian
    -> Gradient -> Non-Maximum Suppression -> Edge
    -> Hessian -> Trị riêng -> Ridge
    -> Laplacian of Gaussian -> Blob
    -> Chuẩn hóa và lưu kết quả
```

Đạo hàm được chuẩn hóa theo tỷ lệ để kết quả tại các Sigma có thể so sánh với
nhau. Phép tích chập Gaussian được tách thành hai phép tích chập một chiều
bằng `cv2.sepFilter2D` để giảm chi phí tính toán.

## 8. Gợi ý kiểm thử

Nên chọn ít nhất 3 đến 5 ảnh có đặc điểm khác nhau:

- Ảnh có biên rõ và ít nhiễu.
- Ảnh có nhiều chi tiết nhỏ.
- Ảnh có đường hoặc gân kéo dài để kiểm tra ridge.
- Ảnh có blob sáng hoặc tối.
- Ảnh có nhiễu hoặc độ tương phản thấp.

Với mỗi ảnh, giữ cùng một bộ Sigma để so sánh. Cần kiểm tra chương trình tạo
đủ file đầu ra, không lỗi với các kích thước ảnh khác nhau và kết quả thay đổi
hợp lý khi Sigma tăng.

## 9. Lỗi thường gặp

### `ModuleNotFoundError: No module named 'cv2'`

```powershell
python -m pip install -r requirements.txt
```

Đảm bảo lệnh cài thư viện và lệnh chạy chương trình sử dụng cùng một Python.

### Không tìm thấy ảnh

- Kiểm tra lại đường dẫn.
- Đặt đường dẫn có dấu cách trong cặp dấu nháy kép.
- Trên Windows có thể dùng đường dẫn tuyệt đối.

### Kết quả có quá nhiều hoặc quá ít biên

Điều chỉnh `--edge-percentile`. Tăng giá trị để giữ biên mạnh, giảm giá trị
để giữ thêm các biên yếu.

### Ridge bị ngược độ sáng

Thử lần lượt `--ridge-polarity bright`, `dark` hoặc `both` tùy theo ảnh.

