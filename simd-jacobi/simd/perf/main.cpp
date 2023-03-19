#include <immintrin.h>
#include <iostream>
#include <fstream> 
#include <chrono>
#include <algorithm>
#include <functional>
#include <limits>
#include <cstdlib>

void dump(const float *mtx, const size_t nXArr, const size_t nYArr, const std::string filename) {
  auto out = std::ofstream(filename);
  for (auto jArr = nXArr; jArr < nXArr*(nYArr-1); jArr+=nXArr) {
    for (auto i = 1ul; i < nXArr-1; i++)
      out << mtx[i + jArr] <<  " ";
    out << "\n";  
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Wrong arguments.\n";
    return 0;
  }

  constexpr auto div8mask = 0xFFFFFFF8ul;

  const auto nX = std::stoul(std::string(argv[1]));
  const auto nY = std::stoul(std::string(argv[2]));
  const auto nT = std::stoul(std::string(argv[3]));
  const auto nXArr = (nX + 2 + 8)&div8mask;
  const auto nYArr = nY + 2; 
  const auto arrSize = nXArr*nYArr;
  const auto jArrLimit = arrSize - nXArr;
  
  constexpr auto xA = 0.0f;
  constexpr auto xB = 400.0f;
  constexpr auto yA = 0.0f;
  constexpr auto yB = 400.0f;
  
  const float hX = (xB - xA)/(nX - 1);
  const float hY = (yB - yA)/(nY - 1);
  
  constexpr auto xS1 = xA + (xB - xA)/3;
  constexpr auto yS1 = yA + (yB - yA)*(((float)2)/3);
  constexpr auto xS2 = xA + (xB - xA)*(((float)2)/3);
  constexpr auto yS2 = yA + (yB - xA)/3;
  constexpr auto r = 0.1f * std::min(xB - xA, yB - yA);
  constexpr auto r2 = r*r;
  
  auto rho = new float[arrSize];
  std::fill(rho, rho+arrSize, 0.0f);
  auto xJ = xA, xJS1 = 0.0f, xJS2 = 0.0f;
  for (auto jArr = nXArr; jArr < jArrLimit; jArr += nXArr) {
    xJS1 = (xJ - xS1)*(xJ - xS1);
    xJS2 = (xJ - xS2)*(xJ - xS2);
    
    auto yI = yA, yIS1 = 0.0f, yIS2 = 0.0f;
    for (auto i = 1ul; i < nX + 1; i++) {
      yIS1 = (yI - yS1)*(yI - yS1);
      yIS2 = (yI - yS2)*(yI - yS2);
      rho[i + jArr] = (xJS1 + yIS1 < r2) ? 0.1f  :
                      (xJS2 + yIS2 < r2) ? -0.1f :
                      0.0f;
      yI += hY;
    }
    xJ += hX;
  }
  
  const auto commonMultiplier = (1/(5/(hX*hX) + 5/(hY*hY))); 
  const auto A = commonMultiplier*0.5f*(5/(hX*hX) - 1/(hY*hY));
  const auto B = commonMultiplier*0.5f*(5/(hY*hY) - 1/(hX*hX));
  const auto C = 0.05f;
  const auto vA = _mm256_set1_ps(A);
  const auto vB = _mm256_set1_ps(B);
  const auto vC = _mm256_set1_ps(C);
  
  auto D = (float*) aligned_alloc(256, sizeof(float) * arrSize);
  std::fill(D, D+arrSize, 0.0f);
  for (auto jArr = nXArr; jArr < jArrLimit; jArr+=nXArr) {
    for (auto i = 1ul; i < nX + 1; i++) {
      D[i + jArr] = commonMultiplier*(2*rho[i + jArr] + 
        0.25f*(rho[i-1 + jArr] + rho[i+1 + jArr] + rho[i + jArr - nXArr] + rho[i + jArr + nXArr]));
    }
  }
  
  auto Phi = (float*) aligned_alloc(256, sizeof(float) * arrSize);
  auto PhiN = (float*) aligned_alloc(256, sizeof(float) * arrSize);
  std::fill(Phi, Phi+arrSize, 0.0f);
  std::fill(PhiN, PhiN+arrSize, 0.0f);

  const auto absMaskVec = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  auto start = std::chrono::high_resolution_clock::now();
  for (auto _ = 0ul; _ < nT; _++) {
    for (auto jArr = nXArr; jArr < jArrLimit; jArr+=nXArr) {
      for (auto i = 0ul; i < ((nX+1 - 8)&div8mask); i+=8) {
        auto topL    = _mm256_load_ps( Phi + i - 1 + jArr - nXArr);
        auto top     = _mm256_loadu_ps(Phi + i +     jArr - nXArr);
        auto topR    = _mm256_loadu_ps(Phi + i + 1 + jArr - nXArr);
        auto left    = _mm256_load_ps( Phi + i - 1 + jArr);
        auto vD      = _mm256_loadu_ps(D + i + jArr);
        auto right   = _mm256_loadu_ps(Phi + i + 1 + jArr);
        auto bottomL = _mm256_load_ps( Phi + i - 1 + jArr + nXArr);
        auto bottom  = _mm256_loadu_ps(Phi + i +     jArr + nXArr);
        auto bottomR = _mm256_loadu_ps(Phi + i + 1 + jArr + nXArr);

        
        auto result0  = _mm256_fmadd_ps(vC, _mm256_add_ps(_mm256_add_ps(topL, topR),
                                                       _mm256_add_ps(bottomL, bottomR)), vD);
        auto result1 = _mm256_fmadd_ps(vA, _mm256_add_ps(top, bottom), result0);
        auto result  = _mm256_fmadd_ps(vB, _mm256_add_ps(left, right), result1);
        
        _mm256_storeu_ps(PhiN + i + jArr, result);
      }
      for (auto i = (nX+1)&div8mask; i < nX+1; i++) {
        PhiN[i + jArr] = A*(Phi[i + jArr-nXArr] + Phi[i + jArr+nXArr]) + B*(Phi[i-1 + jArr] + Phi[i+1 + jArr])
                      + C*(Phi[i-1 + jArr-nXArr] + Phi[i-1 + jArr+nXArr] + Phi[i+1 + jArr-nXArr] + Phi[i+1 + jArr+nXArr])
                      + D[i + jArr];
      }
    }


//    auto vMax = _mm256_set1_ps(std::numeric_limits<float>::min());    
//    for (auto i = nXArr; i < ((arrSize-8)&div8mask); i+=8) {
//      auto phi  = _mm256_load_ps(Phi + i);
//      auto phin = _mm256_load_ps(PhiN + i);
//      auto sub  = _mm256_sub_ps(phi, phin);
//      auto dist = _mm256_and_ps(sub, absMaskVec);
//      vMax = _mm256_max_ps(vMax, dist);
//    }
//    auto vMaxShuffled = _mm256_shuffle_ps(vMax, vMax, _MM_SHUFFLE(2, 3, 0, 1));
//    vMax = _mm256_max_ps(vMax, vMaxShuffled);
//    vMaxShuffled = _mm256_shuffle_ps(vMax, vMax, _MM_SHUFFLE(1, 0, 3, 2));
//    vMax = _mm256_max_ps(vMax, vMaxShuffled);
//    auto max = std::max(vMax[0], vMax[4]);
//    for (auto i = arrSize&div8mask; i < arrSize; i++) 
//      max = std::max<float>(max, std::abs(Phi[i] - PhiN[i]));
//    std::cout << max << "\n";    
    std::swap(PhiN, Phi);
  }
  auto end = std::chrono::high_resolution_clock::now();


  std::cout << std::chrono::duration<double>(end - start).count() << "\n";
 
  delete[] rho;
  free(D);
  free(Phi);
  free(PhiN);
    
  return 0;
}
