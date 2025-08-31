CXX = g++

CXXFLAGS = -O3 -mavx2 -mfma -Wall -fopenmp -march=native -mtune=native -funroll-loops -ffast-math

LDFLAGS = -fopenmp

TARGET = matmul_avx_fp4

SRCS = $(wildcard *.cpp)

OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: run
run: all
	./$(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)
