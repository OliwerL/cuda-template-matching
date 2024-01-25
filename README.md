# Program do Wykrywania Obiektów z Użyciem CUDA

## Przegląd
Program oparty na CUDA służący do wykrywania obiektów na obrazach. Zawiera funkcjonalności przetwarzania obrazów, rotacji, zmiany rozmiaru oraz detekcji konkretnych obiektów w obrazie, wykorzystując CUDA do równoległego przetwarzania.

## Wymagania
- Zestaw narzędzi CUDA (kompatybilny z kodem)
- GPU z obsługą CUDA Compute Capability
- Kompilator C++ z obsługą CUDA
- Biblioteki: `stb_image.h`, `stb_image_write.h`, `cuda_runtime.h`, `device_launch_parameters.h`

## Funkcjonalności
- Ładowanie i zapisywanie obrazów za pomocą biblioteki stb_image
- Przetwarzanie obrazów: konwersja na skale szarości, wykrywanie przezroczystości
- Rotacja obrazów: 45, 90, 135, 180, 225, 270, 315 stopni
- Zmiana rozmiaru obrazów: powiększanie i zmniejszanie obiektów
- Detekcja obiektów: wykrywanie określonych obiektów w obrazie i zaznaczanie ich za pomocą ramki

## Instalacja
1. Zainstaluj wymaganą wersję zestawu narzędzi CUDA ze strony NVIDIA.
2. Upewnij się, że Twoje środowisko programistyczne C++ jest gotowe do pracy.
3. Dołącz wymagane biblioteki do katalogu projektu.

## Użycie
1. Załaduj obrazy przy użyciu funkcji `stbi_load`. Program obecnie pracuje z dwoma obrazami: 'litera.jpg' jako próbką i 'labedz.jpg' jako obrazem docelowym.
2. Wykonaj żądane funkcje przetwarzania obrazów (rotacja, zmiana rozmiaru itp.).
3. W celu detekcji obiektów program porównuje obraz próbki z regionami w obrazie docelowym.
4. Wykryte obiekty będą oznaczone ramką.

### Główne Funkcje
- `transparencyPixels`: Identyfikuje przezroczyste piksele w obrazie próbki.
- `grayscaleKernel`: Konwertuje obraz na skalę szarości.
- `rotate[Angle]degree`: Obraca obraz o określony kąt.
- `increase_letters` i `decrease_letters`: Zmienia rozmiar obrazu.
- `bounding_box`: Rysuje ramkę wokół wykrytych obiektów.
- `find_top_left`: Lokalizuje lewy górny róg wykrytych obiektów.

## Przykład Użycia
```cpp
// Ładowanie obrazu próbki i obrazu docelowego
uint8_t* host_imageA = stbi_load("litera.jpg", &widthA, &heightA, &channelsA, 0);
uint8_t* host_image = stbi_load("labedz.jpg", &width, &height, &channels, 0);

// Wykonanie przetwarzania obrazu (np. rotacja)
rotate90degree <<<gridSizeA, blockSize>>> (dev_one, dev_rotated90d_image, widthA, heightA);

// Detekcja obiektów w obrazie docelowym
bounding_box <<<gridSizeBB, blockSize>>> (dev_image, dev_with_boundingBox, width, height, widthA, heightA, dev_sample_check, dev_top_left, samples);
