# NeuroQuant 3.0 — Dinamik Sermaye Simülasyon Motoru & Matematiksel Mimarisi

NeuroQuant simülasyon robotu, seçilen herhangi bir finansal varlık üzerinde **son 6 aylık (yaklaşık 126 işlem günü)** piyasa zaman serisini adım adım oynatan, **10.000$ başlangıç sermayeli** dinamik bir kantitatif portföy motorudur. 

Bu sistem rastgele al-sat yapan basit bir osilatör değildir; her işlem günü kapanışında **5 bağımsız faktörü** matematiksel olarak modelleyip **0 ile 100 arasında bir "Bileşik Güven Skoru"** üretir ve sermayesini bu inanç düzeyine göre kademeli olarak tahsis eder.

---

## 1. Genel Kurallar & Simülasyon Çerçevesi

| Parametre | Değer / Kural | Açıklama |
| :--- | :--- | :--- |
| **Başlangıç Sermayesi** | `$10,000.00 USD` | Simülasyonun ilk günündeki nakit kasa. |
| **Test Penceresi** | Son 126 İşlem Günü (~6 Ay) | Varlığın en güncel piyasa döngüsünü kapsar. |
| **Veri Sızıntısı Koruması (Zero-Leakage)** | Walk-Forward / Out-of-Sample | Model sadece test döneminden önceki verilerle fit edilir; test günleri model için tamamen "görülmemiş" (unseen) veridir. |
| **Referans (Benchmark)** | Sabit Al-Tut (Buy & Hold) | İlk günün kapanış fiyatından 10.000$'lık alım yapıp 6 ay boyunca hiç dokunmayan pasif yatırımcı. |
| **İşlem Sürtünmesi** | `%0.10 (10 bps)` | Her pozisyon değişiminde (alım/satım nominal tutarı üzerinden) komisyon + kayma (slippage) maliyeti portföyden düşülür. |

---

## 2. Çok Faktörlü Bileşik Güven Skoru ($C$)

Her işlem günü ($t$) kapanışında portföy motoru 5 farklı alt skoru hesaplar ve ağırlıklı toplamını alır:

$$C_t = 0.35 \cdot S_{ML} + 0.20 \cdot S_{Trend} + 0.15 \cdot S_{Sektör} + 0.15 \cdot S_{Değerleme} + 0.15 \cdot S_{Hacim}$$

Elde edilen değer $[5.0, 95.0]$ aralığına kırpılır (`clip`).

```
                              ┌────────────────────────────────────────┐
                              │     GÜNLÜK PİYASA & MODEL VERİSİ       │
                              └──────────────────┬─────────────────────┘
                                                 │
            ┌────────────────┬───────────────────┼───────────────────┬────────────────┐
            ▼                ▼                   ▼                   ▼                ▼
     ┌─────────────┐  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  ┌─────────────┐
     │  %35 ML     │  │  %20 Trend  │     │  %15 Alfa   │     │  %15 Temel  │  │  %15 Hacim  │
     │  (LightGBM) │  │  (200 SMA)  │     │  (20G Rel)  │     │  (Çarpanlar)│  │ (Para Akışı)│
     └──────┬──────┘  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘  └──────┬──────┘
            │                │                   │                   │                │
            └────────────────┴───────────────────┼───────────────────┴────────────────┘
                                                 │
                                                 ▼
                              ┌────────────────────────────────────────┐
                              │      BİLEŞİK GÜVEN SKORU: C [0-100]    │
                              └──────────────────┬─────────────────────┘
                                                 │
                                                 ▼
                              ┌────────────────────────────────────────┐
                              │  HEDEF AĞIRLIK: w* (0k/2k/5k/7.5k/10k) │
                              │    + Histerezis & Bilanço Savunması    │
                              └────────────────────────────────────────┘
```

---

### A. Makine Öğrenmesi Tahmin Skoru ($S_{ML}$ — %35 Ağırlık)
* **Model Türü:** Quantile LightGBM Regresyon Kümesi.
* **Çıktı:** 5 günlük medyan getiri beklentisi ($\hat{y}_{med}$) ve %80 güven bandı ($q_{10}$ ile $q_{90}$).
* **Tahmin Belirsizliği (IQR):** 
  $$IQR = |q_{90} - q_{10}| + 10^{-6}$$
* **Formülasyon:**
  $$S_{ML} = 50.0 + 50.0 \cdot \tanh\left(\frac{\hat{y}_{med}}{IQR} \times 2.5\right)$$
* **Matematiksel Mantık:**
  - Getiri beklentisi pozitif ve belirsizlik bandı darsa ($IQR$ küçük), oran büyür; $\tanh$ fonksiyonu skoru hızla **80 – 95** aralığına taşır (Yüksek Kesinlik).
  - Getiri beklentisi negatifse veya belirsizlik bandı çok genişse oran küçülür; skor **10 – 35** aralığına iner (Düşük Güven / Risk).

---

### B. Teknik Trend & Rejim Skoru ($S_{Trend}$ — %20 Ağırlık)
* **200 Günlük SMA Mesafesi ($d_{200}$):**
  $$d_{200} = \frac{\text{Fiyat}_t - \text{SMA}_{200}(t)}{\text{SMA}_{200}(t)}$$
* **Golden Cross Oranı ($GC$):**
  $$GC = \frac{\text{SMA}_{50}(t)}{\text{SMA}_{200}(t)} - 1.0$$
* **Formülasyon:**
  $$S_{Trend} = \text{clip}\left(50.0 + (d_{200} \times 150.0) + (GC \times 100.0),\; 0.0,\; 100.0\right)$$
* **Matematiksel Mantık:** Fiyat 200 günlük uzun vadeli ortalamanın üzerindeyse ve 50 SMA 200 SMA'yı yukarı kesmişse (Golden Cross aktif) trend tam puan alır. 200 SMA altındaki varlıklar sert biçimde cezalandırılır.

---

### C. Sektörel / Piyasa Alfa Skoru ($S_{Sektör}$ — %15 Ağırlık)
* **Benchmark Seçimi:**
  - ABD Hisse Senetleri için $\to$ `SPY` (S&P 500) veya `QQQ` (Nasdaq 100).
  - Kripto Varlıklar için $\to$ `BTC-USD` (Bitcoin).
* **20 Günlük Kümülatif Rölatif Alfa ($Alfa_{20G}$):**
  $$Alfa_{20G} = \sum_{k=0}^{19} R_{\text{varlık},\; t-k} - \sum_{k=0}^{19} R_{\text{benchmark},\; t-k}$$
* **Formülasyon:**
  $$S_{Sektör} = \text{clip}\left(50.0 + (Alfa_{20G} \times 200.0),\; 0.0,\; 100.0\right)$$
* **Matematiksel Mantık:** Varlık son 1 ayda piyasa endeksinden ne kadar pozitif ayrışmışsa (momentum lideri) puanı o kadar yükselir.

---

### D. Temel Değerleme Skoru ($S_{Değerleme}$ — %15 Ağırlık)
* **Hisse Senetleri İçin:**
  - $F/K$ (Trailing P/E), İleri $F/K$ (Forward P/E), $F/DD$ (Price/Book) ve $PEG$ oranları sektörel medyanlarla kıyaslanır.
  - Çarpanlar sektörün altındaysa `AŞIRI UCUZ / İSKONTOLU` (80–100 puan), dengeliyse `MAKUL` (50–65 puan), aşırı primliyse `PRİMLİ / PAHALI` (15–35 puan).
* **Kripto Varlıklar İçin:**
  - Kripto varlıklarda şirket bilançosu olmadığından sistem hata vermez; on-chain borsa rezervleri, ağ aktivitesi ve likidite koşulları baz alınarak nötr-dengeli rejim puanı (50–60 puan) atanır.

---

### E. Para Akışı & Hacim Anomalisi Skoru ($S_{Hacim}$ — %15 Ağırlık)
* **Hacim Çarpanı ($V_{ratio}$):**
  $$V_{ratio} = \frac{\text{Hacim}_t}{\text{SMA}_{20}(\text{Hacim})_t}$$
* **Formülasyon:**
  $$S_{Hacim} = \text{clip}\left(50.0 + ((V_{ratio} - 1.0) \times 20.0),\; 10.0,\; 90.0\right)$$
* **Matematiksel Mantık:** Eğer günün işlem hacmi 20 günlük ortalamanın $1.5\times - 2.5\times$ katına fırlamışsa, kurumsal kurumsal para akışı (accumulation) tespit edilerek güven skoru yukarı çekilir.

---

## 3. Dinamik Sermaye Tahsisi & Pozisyonlama

Hesaplanan Bileşik Güven Skoru ($C_t$), portföyün **Hedef Hisse Ağırlığına ($w^*$)** dönüştürülür:

| Bileşik Güven Skoru ($C_t$) | Hedef Ağırlık ($w^*$) | Portföy Payı | Stratejik Karar & Anlamı |
| :---: | :---: | :---: | :--- |
| **$C_t < 45.0$** | **$0.00$ (%0)** | `$0$ Hisse / $10,000$ Nakit` | **Tam Nakit Koruması:** Piyasa koşulları veya model sinyali olumsuz; risk almaktan kaçın. |
| **$45.0 \le C_t < 58.0$** | **$0.20$ (%20)** | `~$2,000$ Hisse / $8,000$ Nakit` | **Düşük Riskli Pozisyon:** İlk teyit sinyali, küçük pilot alım. |
| **$58.0 \le C_t < 72.0$** | **$0.50$ (%50)** | `~$5,000$ Hisse / $5,000$ Nakit` | **Dengeli Konsolidasyon:** Trend ve değerleme makul; yarı yarıya pozisyon. |
| **$72.0 \le C_t < 84.0$** | **$0.75$ (%75)** | `~$7,500$ Hisse / $2,500$ Nakit` | **Güçlü Trend Takibi:** ML beklentisi ve hacim desteği güçlü. |
| **$C_t \ge 84.0$** | **$1.00$ (%100)** | `~$10,000$ Hisse / $0$ Nakit` | **Yüksek İnanç (High-Conviction):** Tüm faktörler azami uyumda; tam sermaye ile oyunda kal. |

### Özel Risk Kuralı: Bilanço Koruma Kalkanı (Earnings De-Risking)
Eğer analiz edilen şirket hissesinin **bilanço açıklamasına 5 gün veya daha az süre kalmışsa**:
$$\text{Eğer } t_{\text{bilanço}} \le 5 \text{ Gün ve } w^* > 0.50 \implies w^* = 0.50$$
Modelin güven skoru 95 bile olsa, bilanço piyango riskinden korunmak amacıyla ağırlık zorunlu olarak **en fazla %50** ile sınırlandırılır.

---

## 4. Gerçekçi Piyasa Koşulları & Aşırı İşlem (Churning) Önleme

Gerçek piyasada her gün al-sat yapmak yatırımcıyı komisyona ve işlem kaymalarına (slippage) boğar. Bu durumu engellemek için iki temel filtre uygulanır:

### 1. Histerezis & Bekleme Süresi Filtresi
Robot ancak ve ancak şu iki şarttan biri sağlandığında işlem yapar:
1. **Normal Rebalans:** 
   $$|w^*_{\text{yeni}} - w_{\text{mevcut}}| \ge 0.25 \quad \text{VE} \quad (t - t_{\text{son\_işlem}}) \ge 3 \text{ İş Günü}$$
   *(Ağırlık farkı en az %25 olmalı ve son işlemden bu yana en az 3 iş günü geçmiş olmalıdır.)*
2. **Acil Nakite Kaçış (Emergency De-risk):**
   $$w^*_{\text{yeni}} == 0.0 \quad \text{VE} \quad w_{\text{mevcut}} > 0.0 \quad \text{VE} \quad (t - t_{\text{son\_işlem}}) \ge 1 \text{ İş Günü}$$
   *(Eğer model çöküş öngörüp güven skorunu 45'in altına çekmişse, 3 gün beklenmez; ertesi gün derhal nakite geçilir.)*

### 2. Komisyon & Kayma (Slippage) Kesintisi
Her işlem gerçekleştiğinde, alınıp satılan nominal tutar ($|\Delta Nominal|$) üzerinden:
$$\text{Maliyet} = |\Delta Nominal| \times 0.0010 \quad (\%0.10)$$
doğrudan toplam portföy değerinden tahsil edilir.

### 3. Açıklanabilir Yapay Zeka (XAI) Kayıt Sistemi
Robotun yaptığı her alım veya satımda, o günkü karar motorunun tüm alt bileşenleri (`ML Beklentisi`, `200 SMA Uzaklığı`, `Sektörel Alfa`, `Hacim Oranı` vb.) kütüğe yazılır. Simülasyon arayüzündeki işlem satırına tıklandığında açılan çekmece bu matematiksel dökümü sunar.

---

## 5. Performans Skor Kartı (Tear-Sheet) Metrikleri

Simülasyonun 126. günü tamamlandığında aşağıdaki kurumsal risk ve getiri metrikleri hesaplanır:

### A. Net Toplam Getiri ($R_{total}$)
$$R_{total} = \frac{V_{\text{son}} - 10,000}{10,000} \times 100$$

### B. Alfa Ayrışması (Alpha Spread)
$$Alfa = R_{\text{total, AI}} - R_{\text{total, Al-Tut}}$$

### C. Yıllıklandırılmış Sharpe Oranı
Risksiz faiz oranı ($R_f$) ABD gösterge faizi olarak yıllık %4 ($0.04 / 252$) kabul edilir:
$$\text{Sharpe} = \frac{\text{Ortalama}(R_{\text{günlük}} - R_f)}{\sigma_{\text{günlük}}} \times \sqrt{252}$$

### D. Sortino Oranı (Aşağı Yönlü Risk)
Sadece negatif günlük getirilerin standart sapması ($\sigma_{\text{downside}}$) dikkate alınır (yukarı yönlü volatilite cezalandırılmaz):
$$\text{Sortino} = \frac{\text{Ortalama}(R_{\text{günlük}} - R_f)}{\sigma_{\text{downside}}} \times \sqrt{252}$$

### E. Maksimum Çekilme (Maximum Drawdown - MDD)
Portföyün tepe noktasından (Peak) yaşadığı en derin düşüş yüzdesi:
$$DD_t = \frac{V_t - \max_{\tau \le t}(V_\tau)}{\max_{\tau \le t}(V_\tau)} \times 100$$
$$MDD = \max_t |DD_t|$$

---

*Belge Tarihi: 2026-09-10 · NeuroQuant Core Engine Documentation*
