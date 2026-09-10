# NeuroQuant — Agent Rules & Guidelines

## 1. Tarayıcı (Browser) Kullanımı Kesinlikle Yasaktır
- **İkinci bir emre kadar tarayıcı (`browser_subagent` vb.) kullanmak kesinlikle yasaktır.**
- Screenshot (ekran görüntüsü) almak veya video kaydı oluşturmak için kesinlikle tarayıcı başlatılmayacaktır.

## 2. UI ve Ekran Testleri Kullanıcıya Aittir
- Ekranda manuel veya otomatik UI testi, görsel denetim ve akış testleri ajan tarafından yapılmaz.
- Testleri bizzat kullanıcı yapar.

## 3. Test İhtiyaçlarını Bildirme Protokolü
- Ajan kod geliştirmesini ve terminal düzeyindeki derleme/kontrol işlemlerini (ör. `npm run build`, API curl kontrolleri) tamamladıktan sonra:
  - Kullanıcının arayüzde neleri denemesi gerektiğini,
  - Hangi bileşenlerin ve akışların incelenmesi gerektiğini adım adım kullanıcıya raporlar.
