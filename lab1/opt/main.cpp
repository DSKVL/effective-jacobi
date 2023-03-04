#include <iostream>
#include <chrono>
#include <fstream> 
#include <algorithm>
#include <functional>
#include <cmath>
#include <limits>
#include <span>
#include <numeric>

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

  const auto nX = std::stoul(std::string(argv[1]));
  const auto nY = std::stoul(std::string(argv[2]));
  const auto nT = std::stoul(std::string(argv[3]));
  const auto nXArr = nX + 2;
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

  auto D = new float[arrSize];
  std::fill(D, D+arrSize, 0.0f);
  for (auto jArr = nXArr; jArr < jArrLimit; jArr+=nXArr) {
    for (auto i = 1ul; i < nX + 1; i++) {
      D[i + jArr] = commonMultiplier*(2*rho[i + jArr] + 0.25f*(rho[i-1 + jArr] + rho[i+1 + jArr] + rho[i + jArr - nXArr] + rho[i + jArr + nXArr]));
    }
  }
  
  auto Phi = new float[arrSize];
  auto PhiN = new float[arrSize];
  std::fill(Phi, Phi+arrSize, 0.0f);
  std::fill(PhiN, PhiN+arrSize, 0.0f);

  auto start = std::chrono::high_resolution_clock::now();
  for (auto _ = 0ul; _ < nT; _++) {
    for (auto jArr = nXArr; jArr < jArrLimit; jArr+=nXArr)
      for (auto i = 1ul; i < nX + 1; i++) {
        PhiN[i + jArr] = A*(Phi[i + jArr-nXArr] + Phi[i + jArr+nXArr]) + B*(Phi[i-1 + jArr] + Phi[i+1 + jArr])
                       + C*(Phi[i-1 + jArr-nXArr] + Phi[i-1 + jArr+nXArr] + Phi[i+1 + jArr-nXArr] + Phi[i+1 + jArr+nXArr])
                       + D[i + jArr];
      }
    
    
    std::cout << std::transform_reduce(Phi,  Phi+arrSize,
                                       PhiN,
                                       0.0f,
                                       [](const auto& a, const auto& b){return std::max(a, b);},
                                       [](const auto& a, const auto& b){return std::abs(a-b);}) << "\n";
    
    std::swap(PhiN, Phi);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << std::chrono::duration<double>(end - start).count() << std::endl;
 
  delete[] rho;
  delete[] D;
  delete[] Phi;
  delete[] PhiN;
    
  return 0;
}
