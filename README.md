# MNIST Quickstart (PyTorch)

Basit, hızlıca eğit–değerlendir–dağıt akışıyla **El yazısı rakam tanıma** projesi.
Confusion matrix, accuracy grafiği ve tek görsel tahmini içerir.

## Kurulum
```bash
git clone <your-repo-url>
cd mnist-quickstart
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Eğitim
```bash
python train.py --epochs 3 --batch-size 128 --lr 0.001
```
- Model: `models/mnist_cnn.pth`
- Raporlar: `outputs/metrics.json`, `outputs/acc_plot.png`, `outputs/confusion_matrix.png`

## Tek Görsel Tahmini
```bash
python predict.py --image /path/to/28x28.png
```
Renkli/gray fark etmeksizin 28x28'e ölçeklenir, normalize edilir.

## Proje Yapısı
```
mnist-quickstart
├─ train.py
├─ predict.py
├─ requirements.txt
├─ LICENSE
├─ .gitignore
└─ README.md
```

## Notlar
- Varsayılan olarak veriler `torchvision` ile otomatik indirilir.
- Eğitim süresi kısa tutulmuştur. Epoch sayısını artırarak puanı yükseltebilirsin.
- README'ye eğitim çıktılarından görsel eklemen tavsiye edilir.