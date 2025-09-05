CXX = g++

CXXFLAGS = -O3 -mavx2 -mfma -Wall -fopenmp -march=native -mtune=native -funroll-loops -ffast-math

LDFLAGS = -fopenmp

TARGET = matmul_avx_fp4

CORE_SRCS = fp4_conversion.cpp matmul_opt.cpp
MAIN_SRCS = main.cpp $(CORE_SRCS)

MAIN_OBJS = $(MAIN_SRCS:.cpp=.o)

.PHONY: all
all: $(TARGET)

$(TARGET): $(MAIN_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: run
run: $(TARGET)
	./$(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
