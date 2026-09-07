# 📰 NeuroQuant 3.0: Financial Broadsheet / Editorial Frontend Mimarisi

> **Tasarım Manifestosu:** "Bu bir SaaS şablonu değil; *Financial Times, Stripe Press ve Swiss Style* tipografisinin kesiştiği, yapay zekâ klişelerinden (zifiri siyah, neon yeşil, her başlığın yanına serpiştirilmiş çocuksu ikonlar ve yuvarlatılmış kart kutucukları) tamamen arındırılmış, kurumsal ağırlığı olan entelektüel bir finans gazetesi / kantitatif bülten arayüzüdür."

---

## 🧐 1. Neden Mevcut Arayüz "Dünyanın En AI Tarafından Yapılmış Sitesi" Gibi Duruyordu?

Bir yapay zekâ aracı arayüz tasarlarken neredeyse her zaman şu 4 ezberi uygular:
1. **The "SaaS-Card Kit" Tuzağı:** Sayfayı 12 farklı köşesi yuvarlatılmış (`border-radius: 12px`), altında gri gölge (`box-shadow`) olan bağımsız kutucuklara böler. Sonuç: Birbirinden kopuk, ekranı kutu çöplüğüne çeviren sıkışık bir görünüm.
2. **"İkon Kirliliği" (Icon Pollution):** Her başlığın, her butonun, her metrik kutusunun soluna otomatik olarak Lucide/Feather ikon (rozet, kalkan, alev, hedef, grafik) yapıştırır. Gerçek bir Bloomberg veya FT analizinde her cümlenin başında ikon olmaz; **tipografinin kendisi otoritedir.**
3. **Karanlık Mod / Kripto Botu Klişesi:** Zifiri siyah zemin + fosforlu yeşil/kırmızı. Finans dünyasında ciddiyet bu değildir; ciddiyet okunabilirlikte, kağıt dokusunda ve mürekkebin asaletindedir.
4. **Hiyerarşi Eksikliği:** Her kutu aynı önemdeymiş gibi ekrana serpiştirilir. Göz nerenin manşet, nerenin dipnot olduğunu anlayamaz.

---

## 🎨 2. "Financial Broadsheet" Renk Paleti & Doku Sistemi

Karanlık OLED terminal yerine, **yüzyıllık finans gazeteciliğinin ve prestijli araştırma enstitülerinin** renk paletine geçiyoruz:

| Token Adı | Renk Kodu | Kullanım Alanı | Felsefe |
| :--- | :--- | :--- | :--- |
| `--paper-base` | `#F6F3EC` | Ana Sayfa Arka Planı | Sıcak, hafif dokulu gazete kağıdı beji (Arşiv Kağıdı) |
| `--paper-card` | `#FAF8F5` | Yükseltilmiş Panel / Tablo Yüzeyi | Fildişi beyazı, kağıttan bir ton açık ferah zemin |
| `--paper-elevated`| `#EFEAE0` | Seçili Sekme / Hover Zeminleri | Doğal preslenmiş kağıt tonu |
| `--ink-primary` | `#181512` | Manşetler, Başlıklar, Ana Metinler | Derin espresso mürekkebi (Sert siyah yerine mat mürekkep) |
| `--ink-secondary`| `#57534E` | Açıklamalar, Alt Başlıklar | Taş grisi / sepya mürekkep tonu |
| `--ink-muted` | `#8C827A` | Dipnotlar, Tablo Başlıkları | Mat gazete ara metni |
| `--rule-hairline`| `#DFD7C8` | 1px İnce Ayırıcı Cetveller | Gazete sütun çizgileri (Borders yerine Rules) |
| `--rule-strong` | `#C8BFAF` | Bölüm Başlığı Altı Çift Çizgiler | Manşet altı klasik editorial cetveller |
| `--forest-gain` | `#14532D` | Yükseliş / Pozitif Getiri / Alım | British Racing Green / Derin İngiliz Orman Yeşili |
| `--forest-tint` | `rgba(20, 83, 45, 0.08)` | Pozitif Değişim Arka Planı | Narin yeşil kağıt yıkaması |
| `--madder-loss` | `#881337` | Düşüş / Negatif Getiri / Satım | Derin Kökboya / Madder Kırmızı (Göz kanatmayan vişne) |
| `--madder-tint` | `rgba(136, 19, 55, 0.08)` | Negatif Değişim Arka Planı | Narin kırmızı kağıt yıkaması |
| `--cobalt-accent`| `#1E3A8A` | Bağlantılar, Model Vurgusu | Oxford Laciverti / Kurumsal Mavi |

---

## ✍️ 3. Tipografi: Serif Otoritesi & İsviçre Disiplini

İkonların görsel kirliliğini silip yerini **yazı tipinin zarafetine** bırakıyoruz:

1. **Display & Manşetler (`Newsreader` / `Playfair Display`):**
   - 700 & 800 Bold, optik boyutlandırmalı serif.
   - İtalik vurgular: Örneğin *"The Sovereign Quant"* veya *"Algoritmik Piyasa Manşeti"* bölümlerinde İtalyan edebi italik dokunuşlar.
2. **Gövde Metinleri & Arayüz (`Inter` / `General Sans`):**
   - 400 & 500 ağırlıklarında, ferah satır yüksekliği (`line-height: 1.65`), negatif harf aralığı olmadan nefes alan temiz mizanpaj.
3. **Finansal Rakamlar (`JetBrains Mono` Tabular):**
   - Yalnızca fiyat, oran ve yüzdelerde; alt alta tam hizada duran (`font-variant-numeric: tabular-nums`) net döküm.

---

## 🏛️ 4. Sütunlu Mizanpaj & Kutu Kirliliğini Yok Etme (Broadsheet Grid)

Kutu kutu bağımsız SaaS kartları yerine, bir **gazete sayfası gibi kesintisiz cetvellerle (hairline rules)** bölünen ferah sütunlar:

```
+-----------------------------------------------------------------------------------------------+
|  NEUROQUANT FINANCIAL DESPATCH                    ISSUE NO. 3.0  |  MARKETS: OPEN  |  ISTANBUL|
+===============================================================================================+
|  [PIYASA RADARI]      [KANTITATIF TERMINAL]      [10K SIMULASYON LAB]      [PORTFOY ATOLYESI]  |
+-----------------------------------------------------------------------------------------------+
|                                                                                               |
|  THE LEAD STORY                                               MARKET MOVERS (3 SÜTUNLU CETVEL)|
|  --------------------------------------------------           --------------------------------|
|  Solana, Sektör Yörüngesinden Çıkıyor:                        1. EN YÜKSEK ALFA:              |
|  20 Günlük Ayrışmada %+12.2 Alfa Sinyali                      SOL-USD  $152.40  [+12.2% ALFA] |
|                                                               --------------------------------|
|  Model konsensüsü, küme analizinde teknoloji ve BIST          2. 200 SMA TREND LİDERİ:        |
|  hisselerinde yatay histerezis korumasına geçerken,           GARAN.IS $118.40  [+22.3% SMA]  |
|  kripto kanadında sermaye ağırlığını artırma kararı aldı.     --------------------------------|
|                                                               3. HACİM ANOMALİSİ:             |
|  [Detaylı Analizi Oku ->]                                     ASELS.IS $62.90   [1.80x HACİM] |
|                                                                                               |
+===============================================================================================+
|  PIYASA TARAYICI BÜLTENİ (THE FINANCIAL COMPASS)                                               |
|  -------------------------------------------------------------------------------------------  |
|  FİLTRE: [TÜMÜ]  [BIST 100]  [ABD TEKNOLOJİ]  [KRİPTO]             [Tabloda Ara ____________] |
|  -------------------------------------------------------------------------------------------  |
|  ŞİRKET / VARLIK         SON KÖPANIŞ     24S %      200 SMA      20G ALFA     AI SİNYALİ      |
|  -------------------------------------------------------------------------------------------  |
|  Nvidia Corporation      $230.36        +0.84%     +17.22%      +5.94%       GÜÇLÜ AL (%88.5)|
|  Türk Hava Yolları       $296.50        +0.17%     -2.07%       -4.63%       AL       (%74.2)|
|  Bitcoin (USD)           $64,250.00     +2.10%     +8.15%       +4.50%       AL       (%76.5)|
+-----------------------------------------------------------------------------------------------+
```

---

## 📈 5. Sayfa Sayfa Yeni Tasarım Kararları

### 1. Üst Başlık & Künye (The Masthead):
- SaaS navbar'ı değil; **Financial Times / WSJ benzeri prestijli bir gazete künyesi.**
- Ortalanmış veya sol hizalı serif `NEUROQUANT` logosu, altında ince zarif çift çizgi (`border-top: 1px solid; border-bottom: 1px solid`).
- Sayfa sekmeleri: Tıklanabilir, gereksiz arka plan kutusu olmayan, seçildiğinde altına zarif bir mürekkep çizgisi çeken minimalist linkler.

### 2. TradingView Mum Grafiği (Terminal):
- Koyu siyah yerine: **Sıcak bej/kemik rengi zemin (`#FAF8F5`)**, koyu antrasit grid çizgileri, **orman yeşili** boğa mumları, **madder vişne kırmızısı** ayı mumları.
- Grafiğin tepesindeki mumlar artık üst kenarlığa çarpmayacak; `%15` tavan payı (`scaleMargins.top: 0.15`) ile mumlar serbestçe dalgalanacak.
- SMA 50 çizgisi **lacivert (`#1E3A8A`)**, SMA 200 çizgisi **amber/hardal (`#B45309`)**.

### 3. 10k Simülasyon Laboratuvarı (Hedge Fund Tear-Sheet):
- Animasyon ekranı bir bilgisayar oyunu gibi değil; **Oxford/Princeton araştırma enstitüsü raporu formatında.**
- Zümrüt yeşili mürekkep çizgisi (AI Getirisi) ve koyu gri kesikli cetvel (Sabit Al-Tut).
- Sağdan kayan çekmece: Koyu siyah bir modal yerine; fildişi kağıt zeminli, serif manşetli, zarif gerekçe dökümü.

### 4. İkon ve Emoji Tasfiyesi:
- **Tüm çocuksu emojiler ve her başlığa yapıştırılmış Lucide ikonları kaldırılıyor.**
- İkon yalnızca yön gösteren minimalist oklarda (`↑`, `↓`, `→`) veya zorunlu aksiyonlarda (Oynat/Duraklat) kullanılacak.
- Başlıkların yanına yapıştırılan hedef, kalkan, alev, parıltı gibi görsel gürültüler tamamen temizleniyor.

---

## 🚀 6. Uygulama Adımları (Next Steps)

1. **`frontend/src/index.css`**: "Broadsheet Editorial" token sisteminin, bej/espresso paletinin, serif fontlarının ve hairline cetvel kurallarının yazılması.
2. **`Navigation.tsx`**: Gazete künyesi formatına dönüştürülmesi.
3. **`DashboardPage.tsx`**: Editorial manşet ve gazete tarzı finansal screener tablosuna geçiş.
4. **`TradingViewChart.tsx`**: Bej/krem arka plan temasına ve ferah ölçeklendirmeye uyarlanması.
5. **`SimulationLab.tsx`**: Tear-sheet akademik araştırma formatına kavuşturulması.
