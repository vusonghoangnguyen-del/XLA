# Phat hien dac trung anh da ty le bang dao ham Gaussian

Du an cai dat bang Python cac phep phat hien dac trung anh o nhieu ty le:

- Lam tron Gaussian tai nhieu gia tri `sigma`.
- Dao ham Gaussian bac mot: `Lx`, `Ly`.
- Gradient magnitude va phat hien bien.
- Dao ham Gaussian bac hai: `Lxx`, `Lxy`, `Lyy`.
- Ma tran Hessian va phat hien duong go (ridge).
- Laplacian of Gaussian va phat hien blob.

Day la thuat toan xu ly anh co dien, **khong phai mo hinh hoc may**. Chuong
trinh khong can train, khong co learnable weight va co the xu ly anh ngay sau
khi cai dat thu vien.

## 1. Yeu cau

- Python 3.10 tro len.
- NumPy.
- OpenCV.
- Matplotlib.

## 2. Cai dat tren Windows

Mo PowerShell tai thu muc du an va chay:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Neu PowerShell chan script kich hoat moi truong, co the khong can activate va
goi truc tiep Python trong `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Cai dat tren Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Chay chuong trinh

### Cach chay co ban

```powershell
python gaussian_multiscale_features.py "duong_dan_den_anh.tif"
```

Neu chua activate `.venv` tren Windows:

```powershell
.\.venv\Scripts\python.exe .\gaussian_multiscale_features.py "duong_dan_den_anh.tif"
```

Chuong trinh ho tro cac dinh dang anh OpenCV doc duoc, vi du: PNG, JPG, BMP
va TIFF.

### Chon cac muc Sigma

Mac dinh chuong trinh su dung `1 2 3 5`. Co the thay doi bang:

```powershell
python gaussian_multiscale_features.py "input.tif" --sigmas 0.8 1.5 3 6
```

Y nghia cua Sigma:

- Sigma nho: giu nhieu chi tiet, nhung nhay voi nhieu.
- Sigma trung binh: can bang giua chi tiet va kha nang loai nhieu.
- Sigma lon: giu cac cau truc lon, lam mat cac chi tiet nho.

### Chon thu muc ket qua

```powershell
python gaussian_multiscale_features.py "input.tif" --output "results\anh_01"
```

Neu khong truyen `--output`, ket qua duoc luu trong `multiscale_results`.

### Dieu chinh nguong phat hien bien

```powershell
python gaussian_multiscale_features.py "input.tif" --edge-percentile 80
```

`--edge-percentile` nam trong khoang 0 den 100:

- Tang gia tri: giu lai it bien hon, chu yeu la bien manh.
- Giam gia tri: giu nhieu bien yeu hon, nhung co the tang nhieu.

Gia tri mac dinh la `75`.

### Chon loai ridge

```powershell
python gaussian_multiscale_features.py "input.tif" --ridge-polarity bright
python gaussian_multiscale_features.py "input.tif" --ridge-polarity dark
python gaussian_multiscale_features.py "input.tif" --ridge-polarity both
```

- `bright`: phat hien duong go sang tren nen toi.
- `dark`: phat hien duong go toi tren nen sang.
- `both`: phat hien ca hai loai.

## 5. Vi du day du

```powershell
.\.venv\Scripts\python.exe .\gaussian_multiscale_features.py `
  "D:\data\kidney.tif" `
  --sigmas 1 2 3 5 `
  --edge-percentile 75 `
  --ridge-polarity bright `
  --output ".\results\kidney"
```

## 6. Cac file ket qua

Voi moi gia tri Sigma, chuong trinh tao cac file:

| Hau to file | Noi dung |
| --- | --- |
| `gaussian` | Anh sau khi lam tron Gaussian |
| `lx`, `ly` | Dao ham bac mot theo truc x va y |
| `gradient` | Do lon gradient |
| `edge_strength` | Gradient sau khi lam manh cuc dai cuc bo |
| `edge_mask` | Anh bien nhi phan cuoi cung |
| `lxx`, `lxy`, `lyy` | Cac dao ham bac hai dung de tao Hessian |
| `ridge` | Dap ung phat hien duong go tu tri rieng Hessian |
| `blob_log` | Dap ung blob tu Laplacian of Gaussian |

Ngoai ra:

- `original_grayscale.png`: anh xam dau vao.
- `multiscale_summary.png`: bang tong hop de so sanh cac Sigma.

Vi du `sigma_3_edge_mask.png` la anh bien nhi phan tai `sigma = 3`.

## 7. Quy trinh thuat toan

```text
Anh dau vao
    -> Chuyen anh xam
    -> Lap qua cac gia tri Sigma
    -> Lam tron va tinh dao ham Gaussian
    -> Gradient -> Non-Maximum Suppression -> Edge
    -> Hessian -> Tri rieng -> Ridge
    -> Laplacian of Gaussian -> Blob
    -> Chuan hoa va luu ket qua
```

Dao ham duoc chuan hoa theo ty le de ket qua tai cac Sigma co the so sanh voi
nhau. Phep tich chap Gaussian duoc tach thanh hai phep tich chap mot chieu
bang `cv2.sepFilter2D` de giam chi phi tinh toan.

## 8. Goi y kiem thu

Nen chon it nhat 3 den 5 anh co dac diem khac nhau:

- Anh co bien ro va it nhieu.
- Anh co nhieu chi tiet nho.
- Anh co duong/gân keo dai de kiem tra ridge.
- Anh co blob sang hoac toi.
- Anh co nhieu hoac do tuong phan thap.

Voi moi anh, giu cung bo Sigma de so sanh. Can kiem tra chuong trinh tao du
file dau ra, khong loi voi kich thuoc anh khac nhau va ket qua thay doi hop ly
khi Sigma tang.

## 9. Loi thuong gap

### `ModuleNotFoundError: No module named 'cv2'`

```powershell
python -m pip install -r requirements.txt
```

Dam bao lenh cai thu vien va lenh chay chuong trinh su dung cung mot Python.

### Khong tim thay anh

- Kiem tra lai duong dan.
- Dat duong dan co dau cach trong cap dau nhay kep.
- Tren Windows co the dung duong dan tuyet doi.

### Ket qua co qua nhieu hoac qua it bien

Dieu chinh `--edge-percentile`. Tang de giu bien manh, giam de giu them bien
yeu.

### Ridge bi nguoc mau

Thu lan luot `--ridge-polarity bright`, `dark` hoac `both` tuy vao anh.

