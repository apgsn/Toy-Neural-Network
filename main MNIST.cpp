#include <fstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include "Net.h"
#include <SFML/Graphics.hpp>

void readInvertingMSB(std::ifstream &f, unsigned &v){
  f.read((char*)&v, sizeof(v));
  v = (v << 24) | ((v >> 8 & 255) << 16) | ((v >> 16 & 255) << 8) | (v >> 24);
}
void importMNIST(std::vector< std::vector <float> > &targetDB, std::string filePath, unsigned &imgW, unsigned &imgH){
  std::ifstream file(filePath);
  if(!file.is_open()) throw "Error while opening file : " + filePath + "\nExecution terminated.\n";
  unsigned magicNumber, nOfImages;
  readInvertingMSB(file, magicNumber);
  readInvertingMSB(file, nOfImages);
  if(filePath.find("idx3-ubyte") != std::string::npos){ // read images
    readInvertingMSB(file, imgW);
    readInvertingMSB(file, imgH);
    for(unsigned i = 0; i < nOfImages; i++){
      targetDB.push_back(std::vector<float>());
      for(unsigned j = 0; j < imgW * imgH; j++){
        unsigned char val = 0;
        file.read((char*)&val, sizeof(val));
        targetDB[i].push_back((float)val / 255.0);
      }
    }
  } else { // read labels
    for(unsigned i = 0; i < nOfImages; i++){
      targetDB.push_back(std::vector<float>(10, 0.0));
      unsigned char val = 0;
      file.read((char*)&val, sizeof(val));
      targetDB[i][val] = 1.0;
    }
  }
  std::cout << "Import image data (" << filePath << ") : Done\n";
  file.close();
}
unsigned indexOfMax(std::vector<float> &v){
  return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}
bool inField(int x, int y, std::vector<std::vector<sf::VertexArray> > &f){
	return x >= 0 && y >= 0 && x < f.size() && y < f[0].size();
}
void shiftDrawing(std::vector<std::vector<sf::VertexArray> > &f, unsigned dir){
  for(unsigned i = 0; i < f.size(); i++)
    for(unsigned j = 0; j < f[0].size(); j++)
      for(unsigned k = 0; k < 4; k++)
        switch(dir){
          case 0: // down
          if(j == f[0].size() - 1) f[i][0][k].color = sf::Color::Black;
          else f[i][f[0].size() - 1 - j][k].color = f[i][f[0].size() - 2 - j][k].color;
          break;
          case 1: // up
          if(j == f[0].size() - 1) f[i][f[0].size() - 1][k].color = sf::Color::Black;
          else f[i][j][k].color = f[i][j + 1][k].color;
          break;
          case 2: // left
          if(i == f.size() - 1) f[f.size() - 1][j][k].color = sf::Color::Black;
          else f[i][j][k].color = f[i + 1][j][k].color;
          break;
          case 3: // right
          if(i == f.size() - 1) f[0][j][k].color = sf::Color::Black;
          else f[f.size() - 1 - i][j][k].color = f[f.size() - 2 - i][j][k].color;
          break;
        }
}

int main(){
  std::srand(std::time(NULL));
  const unsigned FREQ_SHOW_SET = 100, TRAINING_SETS = 10000, TEST_SETS = 2000;
  const bool SHOW_IMG = true;
  unsigned imgWidth, imgHeight;

  std::cout << "Neural Network : MNIST Database training\nPress Enter to start ";
  std::getchar();

  std::cout << "\nTraining session : Loading data...\n";
  std::vector< std::vector <float> > trainingInputs, trainingOutputs;
  try {
    importMNIST(trainingInputs, "../MNIST/train-images.idx3-ubyte", imgWidth, imgHeight);
    importMNIST(trainingOutputs, "../MNIST/train-labels.idx1-ubyte", imgWidth, imgHeight);
    std::cout << std::endl;
  } catch (const std::string e) {
    std::cerr << e;
    return 1;
  }

  const float ALPHA = .2;
  const std::vector<unsigned> TOPOLOGY = {imgWidth * imgHeight, 100, 10};
  Net net(TOPOLOGY, ALPHA);

  for(unsigned i = 0; i < TRAINING_SETS; i++){
    unsigned r = std::rand()%trainingInputs.size();
    net.forwardPropagation(trainingInputs[r]);
    net.backPropagation(trainingOutputs[r]);
    if(!((i + 1)%FREQ_SHOW_SET)){
      std::cout << "#" << i + 1 << " / " << TRAINING_SETS << std::endl;
      std::vector<float> results = net.output(2);
      for(unsigned j = 0; j < results.size(); j++){
        std::cout << "[" << j << "]" << " output: " << std::fixed << std::abs(results[j]);
        std::cout << (indexOfMax(trainingOutputs[r]) == j ? " < Target" : "");
        std::cout << (indexOfMax(results) == j ? " < Output" : "") << std::endl;
      }
      std::cout << std::endl;
    }
  }

  char ans;
  while (ans != 'q') {
    std::cout << "\n---------------------\n";
    std::cout << "Test against MNIST: 1\nDraw number: 2\nQuit: Q\n";
    std::cout << "---------------------\n";
    std::cin >> ans;
    ans = std::tolower(ans);
    std::cin.ignore(256, '\n');
    switch (ans) {

      case '1' : { // MNIST test section
        std::cout << "\nTest session : Loading data...\n";
        std::vector< std::vector <float> > testInputs, testOutputs;
        try {
          importMNIST(testInputs, "../MNIST/t10k-images.idx3-ubyte", imgWidth, imgHeight);
          importMNIST(testOutputs, "../MNIST/t10k-labels.idx1-ubyte", imgWidth, imgHeight);
          std::cout << std::endl;
        } catch (const std::string e) {
          std::cerr << e;
          return 1;
        }

        unsigned errors = 0;
        for(unsigned i = 0; i < TEST_SETS; i++){
          unsigned r = std::rand()%testInputs.size();
          // put testInputs[r] in a 2D matrix
          // convolute
          // output goes into forwardPropagation
          net.forwardPropagation(testInputs[r]);
          std::vector<float> results = net.output(2);
          std::cout << std::endl << "Target: " << indexOfMax(testOutputs[r]) << " - Output: " << indexOfMax(results);
          if(indexOfMax(results) - indexOfMax(testOutputs[r])) {
            errors++;
            std::cout << " Error!";
            if(SHOW_IMG) for(unsigned j = 0; j < imgWidth * imgHeight; j++){
              std::cout << (testInputs[r][j] ? "#" : " ");
              if(!(j%imgWidth)) std::cout << std::endl;
            }
          }
        }
        float accuracy = 100.0 * (1.0 - (float)errors / TEST_SETS);
        std::cout << "\n\nAccuracy: " << std::fixed << std::setprecision(2) << accuracy << "%\n";
        break;
      }

      case '2' : { // SFML drawing section
        const int DETAIL = 4, CELLSIZE = 6, BRUSH_SIZE = 5, BRUSH_INTENSITY = 7;
        sf::RenderWindow window(sf::VideoMode(imgWidth * DETAIL * CELLSIZE, imgHeight * DETAIL * CELLSIZE), "MNIST NN test", sf::Style::Titlebar | sf::Style::Close);
        std::vector<std::vector<sf::VertexArray> > field
        (imgWidth * DETAIL, std::vector<sf::VertexArray>(imgHeight * DETAIL, sf::VertexArray(sf::Quads, 4)));
        bool edited, reset = true;
        std::cout << "\nTest session : Draw a number\n";
        while (window.isOpen()){

          if(reset){
            reset = edited = false;
            for(unsigned i = 0; i < field.size(); i++){
              for(unsigned j = 0; j < field[0].size(); j++){
                for(unsigned k = 0; k < 4; k++){
                  field[i][j][k].color = sf::Color::Black;
                  field[i][j][k].position = sf::Vector2f(((k % 2 == k / 2) + i) * CELLSIZE, (k / 2 + j) * CELLSIZE);
                }
              }
            }
          }

          sf::Event event;
        	while(window.pollEvent(event)){
            switch (event.type){
              case sf::Event::MouseButtonPressed :
                if(event.mouseButton.button == sf::Mouse::Right){
                  reset = true;
                  std::cout << "Screen cleared\n";
                }
              break;
              case sf::Event::KeyPressed :
                edited = true;
                if(sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) shiftDrawing(field, 1);
                if(sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) shiftDrawing(field, 0);
                if(sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) shiftDrawing(field, 2);
                if(sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) shiftDrawing(field, 3);
              break;
              case sf::Event::Closed :
      					window.close();
      				break;
            }
          }

          sf::Vector2i mPos = sf::Mouse::getPosition(window);
          mPos /= (int)CELLSIZE;
            if(sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)){
              for(int i = -BRUSH_SIZE; i <= BRUSH_SIZE; i++)
                for(int j = -BRUSH_SIZE; j <= BRUSH_SIZE; j++)
                  for(unsigned k = 0; k < 4; k++){
                    int sfoc = std::max(0, BRUSH_INTENSITY * (BRUSH_SIZE + BRUSH_SIZE - std::abs(i) - std::abs(j)));
                    if(inField(mPos.x + i, mPos.y + j, field)){
                      field[mPos.x + i][mPos.y + j][k].color += sf::Color(sfoc, sfoc, sfoc);
                      edited = true;
                    }
                  }

          }

          if(edited && !reset){
            edited = false;
            std::vector<float> testInputs(imgWidth * imgHeight);
            for(unsigned i = 0; i < imgHeight; i++){
              for(unsigned j = 0; j < imgWidth; j++){
                float sum = 0;
                for(unsigned k = 0; k < DETAIL; k++){
                  for(unsigned l = 0; l < DETAIL; l++){
                    sum += field[i * DETAIL + k][j * DETAIL + l][0].color.r;
                  }
                }
                testInputs[j * imgWidth + i] = sum / (255.0 * DETAIL * DETAIL);
              }
            }
            net.forwardPropagation(testInputs);
            std::vector<float> results = net.output(2);
            std::cout << "Guess: " << indexOfMax(results);
            std::cout << " (" << std::fixed << std::setprecision(2) << results[indexOfMax(results)]*100 << "%)\n";
          }

          window.clear();
          for(auto &i : field)
            for(auto &c : i)
              window.draw(c);
          window.display();
        }
        break;
      }

      case 'q' :
        std::cout << "\nProgram Terminated.\n";
      break;

      default :
        std::cout << "\nError. Please Retry.\n";
      break;
    }
  }
  return 0;
}
