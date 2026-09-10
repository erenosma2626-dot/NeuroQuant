# Kural: Tarayıcı ve UI Test Kısıtlaması

## 1. Tarayıcı (Browser) Kullanımı Kesinlikle Yasaktır
- **İkinci bir emre kadar tarayıcı (`browser_subagent`) kullanmak kesinlikle yasaktır.**
- Ekran görüntüsü (screenshot) almak, video kaydetmek ya da arayüzü kontrol etmek amacıyla tarayıcıya girilmeyecektir.

## 2. UI ve Ekran Testleri Kullanıcı Tarafından Yapılır
- Ekranda görsel testler, buton tıklama veya kullanıcı akış denemeleri ajan tarafından yapılmaz.
- Ajan sadece kod yazma, terminal üzerinden derleme (build), lint ve arka plan servis kontrollerini gerçekleştirir.

## 3. Test İhtiyaçlarını Kullanıcıya Bildirme
- Yapılan değişikliklerin ardından ajan, kullanıcının hangi adımları, senaryoları ve hisseleri test etmesi gerektiğini maddeler halinde net olarak kullanıcıya bildirir. Testleri bizzat kullanıcı yapar.
